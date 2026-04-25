"""FastAPI web app — interactive ETF screener.

Run with::

    uvicorn asx_portfolio.web_app:app --host 127.0.0.1 --port 8000

or use the bundled ``start.command`` launcher.

Endpoints:
  * ``GET /``                — the SPA UI
  * ``POST /api/screen``     — body: ScreenRequest; returns top picks + filtered ranking
  * ``GET /api/categories``  — returns the list of available asset categories
  * ``GET /api/refresh``     — force-refresh the Yahoo cache
  * ``GET /api/health``      — diagnostic info on cache state
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

from . import etf_fetcher, etf_scorer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("asx_portfolio.web")

app = FastAPI(title="ASX ETF Screener", version="0.3.0")
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


def _build_dividend_plan(picks: list[dict], target_quarterly_aud: float) -> dict[str, Any] | None:
    """Compute the equal-weight investment per pick to hit the target quarterly dividend.

    Returns None when target_quarterly_aud <= 0 or no picks. Otherwise returns a dict
    with per-pick allocations and totals. ETFs with zero yield are flagged so the
    user knows they'd be invested in for growth, not income.
    """
    if not picks or target_quarterly_aud <= 0:
        return None

    target_annual = target_quarterly_aud * 4.0
    yields = [(p.get("dividend_yield") or 0.0) for p in picks]
    sum_yields = sum(yields)
    n = len(picks)

    if sum_yields <= 0:
        # No income from any pick — can't size a dividend portfolio.
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

    # Equal-weight allocation:
    #   per_etf_capital = target_annual / sum_yields
    #   total_capital   = per_etf_capital * n
    per_etf = target_annual / sum_yields
    total = per_etf * n

    allocations = []
    zero_yield_count = 0
    for p, y in zip(picks, yields):
        annual = per_etf * y
        allocations.append(
            {
                "ticker": p.get("ticker"),
                "name": p.get("name"),
                "yield": y,
                "investment_aud": round(per_etf, 2),
                "expected_annual_aud": round(annual, 2),
                "expected_quarterly_aud": round(annual / 4.0, 2),
                "expected_monthly_aud": round(annual / 12.0, 2),
            }
        )
        if y <= 0:
            zero_yield_count += 1

    warnings = []
    if zero_yield_count:
        warnings.append(
            f"{zero_yield_count} of your {n} picks pay no distributions — "
            "they're held for capital growth and do not contribute to your income target. "
            "Their share of the portfolio is dead weight from a dividend perspective."
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


# ---------------------------------------------------------------------------
# Endpoints
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


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    template_path = Path(__file__).parent / "templates" / "etf_app.html"
    return template_path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Entrypoint for `python -m asx_portfolio.web_app`
# ---------------------------------------------------------------------------


def run() -> None:
    import uvicorn
    uvicorn.run("asx_portfolio.web_app:app", host="127.0.0.1", port=8000, log_level="info")


if __name__ == "__main__":
    run()
