"""
NEW ROUTES TO ADD TO web_app.py
================================

Add these imports at the top of web_app.py (after existing imports):

    from . import stock_fetcher, stock_scorer

Then add these Pydantic models and route handlers anywhere after the existing ones.
"""

# ─── NEW PYDANTIC MODELS ─────────────────────────────────────────────────────

class StockScreenRequest(BaseModel):
    risk_tolerance: str = Field(default="medium")
    horizon: str = Field(default="medium")
    sectors: list[str] = Field(default_factory=list)
    min_dividend_yield: float = Field(default=0.0)
    min_market_cap_aud_m: float = Field(default=0.0)
    top_n: int = Field(default=5)
    force_refresh: bool = Field(default=False)


class SimulateRequest(BaseModel):
    holdings: list[dict] = Field(...)   # [{ticker, name, allocation, annualReturn, volatility, source}]
    initial_amount: float = Field(default=10_000.0)
    years: int = Field(default=10)
    sim_count: int = Field(default=500)


# ─── NEW ROUTES ──────────────────────────────────────────────────────────────

@app.get("/stocks", response_class=HTMLResponse)
def stocks_page() -> str:
    template_path = Path(__file__).parent / "templates" / "stocks_app.html"
    return template_path.read_text(encoding="utf-8")


@app.get("/combined", response_class=HTMLResponse)
def combined_page() -> str:
    template_path = Path(__file__).parent / "templates" / "combined_app.html"
    return template_path.read_text(encoding="utf-8")


@app.get("/api/sectors")
def get_sectors() -> dict:
    """Return unique sectors in the fetched stock universe."""
    try:
        stocks = stock_fetcher.get_stock_data(force_refresh=False)
        sectors = sorted({s.sector for s in stocks if s.sector})
        return {"sectors": sectors}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/screen-stocks")
def screen_stocks(req: StockScreenRequest) -> dict:
    started = time.time()
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

    def _row_to_stock_payload(d: dict, include_history: bool = False) -> dict:
        def _clean(v):
            if isinstance(v, float) and math.isnan(v):
                return None
            return v
        out = {k: _clean(v) for k, v in d.items()
               if not isinstance(v, (list, dict)) or k in ("history_dates", "history_prices")}
        if not include_history:
            out.pop("history_dates", None)
            out.pop("history_prices", None)
        return out

    pick_payloads = [_row_to_stock_payload(top.iloc[i].to_dict(), include_history=True) for i in range(len(top))]

    return {
        "picks": pick_payloads,
        "all_filtered": [_row_to_stock_payload(scored.iloc[i].to_dict(), include_history=False) for i in range(len(scored))],
        "rationales": rationales,
        "pillar_weights": pillar_weights,
        "candidate_count": len(scored),
        "elapsed_seconds": round(time.time() - started, 2),
    }


@app.post("/api/simulate")
def simulate_portfolio(req: SimulateRequest) -> dict:
    """Monte Carlo simulation for a mixed ETF + stocks portfolio."""
    import random
    import numpy as np

    holdings = req.holdings
    if not holdings:
        raise HTTPException(status_code=400, detail="No holdings provided")

    total_alloc = sum(h.get("allocation", 0) for h in holdings)
    if abs(total_alloc - 100.0) > 0.5:
        raise HTTPException(status_code=400, detail=f"Allocations must sum to 100 (got {total_alloc})")

    # Weighted portfolio stats
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

    # Box-Muller Monte Carlo
    rng = np.random.default_rng()
    # shape: (sim_count, months)
    z = rng.standard_normal((sim_count, months))
    monthly_returns = monthly_return + monthly_vol * z  # (sim_count, months)
    # Cumulative product from initial
    growth = np.cumprod(1 + monthly_returns, axis=1)
    paths = initial * np.hstack([np.ones((sim_count, 1)), growth])  # (sim_count, months+1)

    final_vals = paths[:, -1]
    sorted_idx = np.argsort(final_vals)

    def _path_to_list(path: np.ndarray, step: int = 1) -> list[float]:
        return [round(float(v), 2) for v in path[::step]]

    step = max(1, months // 60)

    p10 = _path_to_list(paths[sorted_idx[int(sim_count * 0.10)]], step)
    p50 = _path_to_list(paths[sorted_idx[int(sim_count * 0.50)]], step)
    p90 = _path_to_list(paths[sorted_idx[int(sim_count * 0.90)]], step)

    mean_final = float(np.mean(final_vals))
    prob_profit = float(np.mean(final_vals > initial))

    return {
        "p10": p10,
        "p50": p50,
        "p90": p90,
        "mean_final": round(mean_final, 2),
        "prob_profit": round(prob_profit, 4),
        "weighted_return_pct": round(weighted_return * 100, 2),
        "weighted_vol_pct": round(weighted_vol * 100, 2),
        "months": months,
        "step": step,
        "initial_amount": initial,
    }
