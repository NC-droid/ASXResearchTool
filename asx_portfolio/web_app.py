"""FastAPI web app — interactive ETF screener + ASX 200 stocks + combined portfolio simulator.

Run with::

    uvicorn asx_portfolio.web_app:app --host 127.0.0.1 --port 8000

or use the bundled ``start.command`` launcher.

Endpoints:
  * ``GET /``                   — ETF Screener SPA
  * ``GET /stocks``             — ASX 200 Stock Screener SPA
  * ``GET /combined``           — Combined Portfolio Simulator SPA
  * ``POST /api/screen``        — body: ScreenRequest; returns ETF picks + filtered ranking
  * ``POST /api/screen-stocks`` — body: StockScreenRequest; returns ASX 200 stock picks
  * ``POST /api/simulate``      — body: SimulateRequest; runs Monte Carlo simulation
  * ``GET /api/categories``     — returns the list of available ETF asset categories
  * ``GET /api/sectors``        — returns the list of available stock sectors
  * ``GET /api/refresh``        — force-refresh the ETF Yahoo cache
  * ``GET /api/health``         — diagnostic info on cache state
"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

# ETF modules — always required, imported at top level
from . import etf_fetcher, etf_scorer

# Stock + numpy modules imported lazily inside routes so a missing dep
# never prevents the ETF page from starting up.

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("asx_portfolio.web")

app = FastAPI(title="ASX ETF + Stock Screener", version="0.4.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class ScreenRequest(BaseModel):
    risk_tolerance: str = Field(default="medium")
    horizon: str = Field(default="medium")
    categories: list[str] = Field(default_factory=list)
    max_mer: float | None = Field(default=None)
    min_dividend_yield_annual: float = Field(default=0.0)
    top_n: int = Field(default=3)
    force_refresh: bool = Field(default=False)
    target_quarterly_dividend_aud: float = Field(default=0.0)


class StockScreenRequest(BaseModel):
    risk_tolerance: str = Field(default="medium")
    horizon: str = Field(default="medium")
    sectors: list[str] = Field(default_factory=list)
    min_dividend_yield: float = Field(default=0.0)
    min_market_cap_aud_m: float = Field(default=0.0)
    top_n: int = Field(default=5)
    force_refresh: bool = Field(default=False)


class SimulateRequest(BaseModel):
    holdings: list[dict] = Field(...)
    initial_amount: float = Field(default=10_000.0)
    years: int = Field(default=10)
    sim_count: int = Field(default=500)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _safe(x: Any) -> Any:
    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x):
            return None
    return x


def _row_to_payload(row, include_history: bool = True) -> dict[str, Any]:
    payload = {k: _safe(v) for k, v in row.items() if k not in ("history_dates", "history_prices", "pillar_weights")}
    if include_history:
        payload["history_dates"] = row.get("history_dates", [])
        payload["history_prices"] = row.get("history_prices", [])
    return payload


def _row_to_stock_payload(d: dict, include_history: bool = True) -> dict[str, Any]:
    out = {k: _safe(v) for k, v in d.items() if k not in ("history_dates", "history_prices", "pillar_weights")}
    if include_history:
        out["history_dates"] = d.get("history_dates", [])
        out["history_prices"] = d.get("history_prices", [])
    return out


def _build_dividend_plan(picks: list[dict], target_quarterly_aud: float) -> dict[str, Any] | None:
    if not picks or target_quarterly_aud <= 0:
        return None

    target_annual = target_quarterly_aud * 4.0
    yields = [(p.get("dividend_yield") or 0.0) for p in picks]
    sum_yields = sum(yields)
    n = len(picks)

    if sum_yields <= 0:
        return {
            "feasible": False,
            "target_quarterly_aud": target_quarterly_aud,
            "target_annual_aud": target_annual,
            "total_capital_aud": None,
            "allocations": [],
            "warnings": [
                "None of the current picks pay a distribution, so no allocation can produce the target.",
                "Try lowering the risk tolerance or setting a minimum yield to favour income-producing ETFs.",
            ],
        }

    per_etf = target_annual / sum_yields
    total = per_etf * n
    allocations = []
    zero_yield_count = 0
    for p, y in zip(picks, yields):
        annual = per_etf * y
        allocations.append({
            "ticker": p.get("ticker"),
            "name": p.get("name"),
            "yield": y,
            "investment_aud": round(per_etf, 2),
            "expected_annual_aud": round(annual, 2),
            "expected_quarterly_aud": round(annual / 4.0, 2),
            "expected_monthly_aud": round(annual / 12.0, 2),
        })
        if y <= 0:
            zero_yield_count += 1

    warnings = []
    if zero_yield_count:
        warnings.append(
            f"{zero_yield_count} of your {n} picks pay no distributions — "
            "they're held for capital growth and do not contribute to your income target."
        )
    avg_yield = sum_yields / n
    if avg_yield < 0.02:
        warnings.append(
            f"Average yield across the picks is only {avg_yield*100:.1f}% p.a., "
            "which makes the required capital large. Consider tightening the minimum-yield filter."
        )

    return {
        "feasible": True,
        "target_quarterly_aud": target_quarterly_aud,
        "target_annual_aud": target_annual,
        "total_capital_aud": round(total, 2),
        "average_yield": avg_yield,
        "allocations": allocations,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# ETF endpoints (original — untouched)
# ---------------------------------------------------------------------------


@app.get("/api/categories")
def categories() -> dict[str, Any]:
    df = etf_fetcher.load_etf_universe()
    cats = sorted(df["category"].dropna().unique().tolist())
    return {"categories": cats, "count_per_category": df["category"].value_counts().to_dict()}


@app.get("/api/health")
def health() -> dict[str, Any]:
    cs = etf_fetcher.cache_status()
    return {"ok": True, "cache": cs}


@app.get("/api/refresh")
def refresh() -> dict[str, Any]:
    started = time.time()
    data = etf_fetcher.get_etf_data(force_refresh=True)
    return {
        "refreshed": True,
        "etfs_fetched": len(data),
        "elapsed_seconds": round(time.time() - started, 2),
    }


@app.post("/api/screen")
def screen(req: ScreenRequest) -> dict[str, Any]:
    started = time.time()
    try:
        etfs = etf_fetcher.get_etf_data(force_refresh=req.force_refresh)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    inputs = etf_scorer.ScreenInputs(
        risk_tolerance=req.risk_tolerance,
        horizon=req.horizon,
        categories=req.categories,
        max_mer=req.max_mer,
        min_dividend_yield_annual=req.min_dividend_yield_annual,
        top_n=req.top_n,
    )
    scored = etf_scorer.score(etfs, inputs)
    if scored is None or scored.empty:
        return {
            "picks": [],
            "all_filtered": [],
            "rationales": [],
            "pillar_weights": etf_scorer._pillar_weights(inputs.normalised()),
            "candidate_count": 0,
            "dividend_plan": None,
            "fetched_at": etf_fetcher.cache_status().get("age_seconds"),
            "elapsed_seconds": round(time.time() - started, 2),
        }

    top = etf_scorer.select_top(scored, n=inputs.normalised().top_n)
    rationales = [etf_scorer.build_rationale(row) for _, row in top.iterrows()]
    pillar_weights = top.iloc[0].get("pillar_weights", {}) if len(top) else etf_scorer._pillar_weights(inputs.normalised())

    pick_payloads = [_row_to_payload(top.iloc[i].to_dict(), include_history=True) for i in range(len(top))]
    dividend_plan = _build_dividend_plan(pick_payloads, req.target_quarterly_dividend_aud)

    return {
        "picks": pick_payloads,
        "all_filtered": [_row_to_payload(scored.iloc[i].to_dict(), include_history=False) for i in range(len(scored))],
        "rationales": rationales,
        "pillar_weights": pillar_weights,
        "candidate_count": len(scored),
        "dividend_plan": dividend_plan,
        "elapsed_seconds": round(time.time() - started, 2),
    }


# ---------------------------------------------------------------------------
# ASX 200 Stocks endpoints (v2 — lazy imports)
# ---------------------------------------------------------------------------


@app.get("/api/sectors")
def get_sectors() -> dict[str, Any]:
    try:
        from . import stock_fetcher
        stocks = stock_fetcher.get_stock_data(force_refresh=False)
        sectors = sorted({s.sector for s in stocks if s.sector})
        return {"sectors": sectors}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/screen-stocks")
def screen_stocks(req: StockScreenRequest) -> dict[str, Any]:
    started = time.time()
    try:
        from . import stock_fetcher, stock_scorer
    except ImportError as exc:
        raise HTTPException(status_code=500, detail=f"Stock modules not available: {exc}")

    try:
        stocks = stock_fetcher.get_stock_data(force_refresh=req.force_refresh)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    inputs = stock_scorer.StockScreenInputs(
        risk_tolerance=req.risk_tolerance,
        horizon=req.horizon,
        sectors=req.sectors,
        min_dividend_yield=req.min_dividend_yield,
        min_market_cap_aud_m=req.min_market_cap_aud_m,
        top_n=req.top_n,
    )
    scored = stock_scorer.score(stocks, inputs)

    if scored is None or scored.empty:
        return {
            "picks": [],
            "all_filtered": [],
            "rationales": [],
            "pillar_weights": stock_scorer._pillar_weights(inputs.normalised()),
            "candidate_count": 0,
            "elapsed_seconds": round(time.time() - started, 2),
        }

    top = stock_scorer.select_top(scored, n=inputs.normalised().top_n)
    rationales = [stock_scorer.build_rationale(row) for _, row in top.iterrows()]
    pillar_weights = top.iloc[0].get("pillar_weights", {}) if len(top) else stock_scorer._pillar_weights(inputs.normalised())

    return {
        "picks": [_row_to_stock_payload(top.iloc[i].to_dict(), include_history=True) for i in range(len(top))],
        "all_filtered": [_row_to_stock_payload(scored.iloc[i].to_dict(), include_history=False) for i in range(len(scored))],
        "rationales": rationales,
        "pillar_weights": pillar_weights,
        "candidate_count": len(scored),
        "elapsed_seconds": round(time.time() - started, 2),
    }


# ---------------------------------------------------------------------------
# Monte Carlo simulation endpoint (v2 — lazy numpy import)
# ---------------------------------------------------------------------------


@app.post("/api/simulate")
def simulate_portfolio(req: SimulateRequest) -> dict[str, Any]:
    try:
        import numpy as np
    except ImportError as exc:
        raise HTTPException(status_code=500, detail=f"numpy not available: {exc}")

    holdings = req.holdings
    if not holdings:
        raise HTTPException(status_code=400, detail="No holdings provided")

    total_alloc = sum(h.get("allocation", 0) for h in holdings)
    if abs(total_alloc - 100.0) > 0.5:
        raise HTTPException(status_code=400, detail=f"Allocations must sum to 100 (got {total_alloc})")

    weighted_return = sum(
        (h.get("annualReturn", 0) / 100) * (h.get("allocation", 0) / 100)
        for h in holdings
    )
    weighted_vol = math.sqrt(sum(
        ((h.get("volatility", 15) / 100) * (h.get("allocation", 0) / 100)) ** 2
        for h in holdings
    ))

    months = req.years * 12
    monthly_return = weighted_return / 12
    monthly_vol = weighted_vol / math.sqrt(12)
    initial = req.initial_amount
    sim_count = min(req.sim_count, 1000)

    rng = np.random.default_rng()
    z = rng.standard_normal((sim_count, months))
    monthly_returns = monthly_return + monthly_vol * z
    growth = np.cumprod(1 + monthly_returns, axis=1)
    paths = initial * np.hstack([np.ones((sim_count, 1)), growth])

    final_vals = paths[:, -1]
    sorted_idx = np.argsort(final_vals)
    step = max(1, months // 60)

    def _path_to_list(path: np.ndarray) -> list[float]:
        return [round(float(v), 2) for v in path[::step]]

    return {
        "p10": _path_to_list(paths[sorted_idx[int(sim_count * 0.10)]]),
        "p50": _path_to_list(paths[sorted_idx[int(sim_count * 0.50)]]),
        "p90": _path_to_list(paths[sorted_idx[int(sim_count * 0.90)]]),
        "mean_final": round(float(np.mean(final_vals)), 2),
        "prob_profit": round(float(np.mean(final_vals > initial)), 4),
        "weighted_return_pct": round(weighted_return * 100, 2),
        "weighted_vol_pct": round(weighted_vol * 100, 2),
        "months": months,
        "step": step,
        "initial_amount": initial,
    }


# ---------------------------------------------------------------------------
# Page routes
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
def landing() -> str:
    template_path = Path(__file__).parent / "templates" / "landing.html"
    return template_path.read_text(encoding="utf-8")


@app.get("/etfs", response_class=HTMLResponse)
def index() -> str:
    template_path = Path(__file__).parent / "templates" / "etf_app.html"
    return template_path.read_text(encoding="utf-8")


@app.get("/stocks", response_class=HTMLResponse)
def stocks_page() -> str:
    template_path = Path(__file__).parent / "templates" / "stocks_app.html"
    return template_path.read_text(encoding="utf-8")


@app.get("/combined", response_class=HTMLResponse)
def combined_page() -> str:
    template_path = Path(__file__).parent / "templates" / "combined_app.html"
    return template_path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Entrypoint for `python -m asx_portfolio.web_app`
# ---------------------------------------------------------------------------


def run() -> None:
    import uvicorn
    uvicorn.run("asx_portfolio.web_app:app", host="127.0.0.1", port=8000, log_level="info")


if __name__ == "__main__":
    run()
