"""Build the static site against synthetic data so we can verify the pipeline.

Generates a CSV + Excel screen output and a portfolio_state.json with
plausible value history, then invokes site_builder.build_site and
sanity-checks the produced HTML.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from asx_portfolio import reporter, scorer, universe, site_builder  # noqa: E402
from asx_portfolio.fetcher import StockData  # noqa: E402


def fake_stock_data(ticker: str, seed: int) -> StockData:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp.today(), periods=252 * 5)
    drift = rng.normal(0.0004, 0.001)
    daily_vol = rng.uniform(0.01, 0.025)
    returns = rng.normal(drift, daily_vol, size=len(dates))
    prices = 50 * np.exp(np.cumsum(returns))
    history = pd.DataFrame({"Close": prices}, index=dates)
    info = {
        "currentPrice": float(prices[-1]),
        "marketCap": float(rng.uniform(1e9, 3e11)),
        "trailingPE": float(rng.uniform(8, 35)),
        "forwardPE": float(rng.uniform(8, 30)),
        "priceToBook": float(rng.uniform(0.7, 8)),
        "freeCashflow": float(rng.uniform(5e7, 5e9)),
        "revenueGrowth": float(rng.uniform(-0.05, 0.25)),
        "earningsGrowth": float(rng.uniform(-0.10, 0.30)),
        "earningsQuarterlyGrowth": float(rng.uniform(-0.10, 0.40)),
        "returnOnEquity": float(rng.uniform(0.02, 0.30)),
        "returnOnAssets": float(rng.uniform(0.01, 0.18)),
        "grossMargins": float(rng.uniform(0.15, 0.65)),
        "debtToEquity": float(rng.uniform(10, 200)),
        "dividendYield": float(rng.uniform(0.0, 0.07)),
        "payoutRatio": float(rng.uniform(0.0, 0.95)),
        "beta": float(rng.uniform(0.4, 1.8)),
    }
    return StockData(ticker=ticker, info=info, history=history)


def main() -> int:
    project_root = Path(__file__).resolve().parent.parent
    reports_dir = project_root / "reports"
    site_dir = project_root / "site"
    state_path = project_root / "portfolio_state.json"

    # --- Generate a screen output -------------------------------------------
    uni = universe.load_asx200().head(40)
    fetched = {row.ticker: fake_stock_data(row.ticker, seed=i) for i, row in enumerate(uni.itertuples())}
    name_lookup = dict(zip(uni["ticker"], uni["name"]))
    sector_lookup = dict(zip(uni["ticker"], uni["sector"]))
    metrics = [
        scorer.metrics_from_stockdata(sd, name=name_lookup[t], sector=sector_lookup[t])
        for t, sd in fetched.items()
    ]
    scored = scorer.score_universe(metrics)
    picks = scorer.select_top_picks(scored, n=5, min_projected_return=0.10)
    rationales = [scorer.build_rationale(row) for _, row in picks.iterrows()]

    # Write today's CSV + Excel
    csv_path, xlsx_path = reporter.make_default_paths(reports_dir)
    reporter.write_csv(picks, csv_path)
    reporter.write_excel(picks=picks, full_screen=scored, rationales=rationales, out_path=xlsx_path)

    # Also write a few historical pick CSVs to populate the archive
    for days_ago in (7, 14, 21):
        ds = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")
        archive_path = reports_dir / f"asx_picks_{ds}.csv"
        reporter.write_csv(picks, archive_path)

    # --- Generate a portfolio_state.json with value history -----------------
    today = datetime.now()
    positions = []
    cash = 10_000.0
    per = 2_000.0
    for _, row in picks.iterrows():
        sd = fetched[row["ticker"]]
        price = float(sd.history["Close"].iloc[-1])
        shares = per / price
        cost = shares * price
        cash -= cost
        positions.append({
            "ticker": row["ticker"],
            "shares": shares,
            "avg_cost": price,
            "entry_date": (today - timedelta(days=30)).strftime("%Y-%m-%d"),
            "name": row.get("name", ""),
            "sector": row.get("sector", ""),
        })
    # 30 days of value history with mild drift up
    history = []
    val = 10_000.0
    rng = np.random.default_rng(7)
    for i in range(30, -1, -1):
        d = (today - timedelta(days=i)).strftime("%Y-%m-%d")
        val *= (1 + rng.normal(0.0008, 0.012))
        history.append({
            "date": d, "total_value": round(val, 2), "cash": round(cash, 2),
            "invested_value": round(val - cash, 2),
            "total_return_pct": (val - 10_000.0) / 10_000.0,
        })

    state = {
        "initial_capital": 10_000.0,
        "cash": round(cash, 2),
        "created_at": (today - timedelta(days=30)).strftime("%Y-%m-%d"),
        "last_rebalance": (today - timedelta(days=30)).strftime("%Y-%m-%d"),
        "rebalance_interval_days": 30,
        "transaction_cost": 0.0,
        "positions": positions,
        "value_history": history,
        "rebalance_history": [{
            "date": (today - timedelta(days=30)).strftime("%Y-%m-%d"),
            "picks": [p["ticker"] for p in positions],
            "cost_basis": round(sum(p["shares"] * p["avg_cost"] for p in positions), 2),
            "cash_after": round(cash, 2),
        }],
    }
    state_path.write_text(json.dumps(state, indent=2))

    # --- Build the site ------------------------------------------------------
    index = site_builder.build_site(site_dir=site_dir, reports_dir=reports_dir, state_path=state_path)

    # Verify all 4 pages exist and contain expected markers
    expected = {
        "index.html":     ["Today's Picks", "composite_score" not in "x" or "Composite", "Full ASX 200"],
        "portfolio.html": ["Paper Portfolio", "Equity curve", "Holdings"],
        "archive.html":   ["Picks Archive"],
        "backtest.html":  ["Backtest"],
    }
    for fname, markers in expected.items():
        path = site_dir / fname
        assert path.exists(), f"missing {fname}"
        text = path.read_text()
        for m in markers:
            if isinstance(m, str):
                assert m in text, f"{fname} missing marker: {m!r}"
        assert len(text) > 1000, f"{fname} unexpectedly short ({len(text)} chars)"

    print(f"OK — site built at {index}")
    print(f"  index.html      ({(site_dir/'index.html').stat().st_size:,} bytes)")
    print(f"  portfolio.html  ({(site_dir/'portfolio.html').stat().st_size:,} bytes)")
    print(f"  archive.html    ({(site_dir/'archive.html').stat().st_size:,} bytes)")
    print(f"  backtest.html   ({(site_dir/'backtest.html').stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
