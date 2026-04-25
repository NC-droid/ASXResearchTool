"""Static site generator for the ASX portfolio project.

Reads:
  * the latest screen output (CSV + Excel in reports/)
  * portfolio_state.json (paper portfolio)
  * the most recent backtest output

Writes:
  * site/index.html       (today's picks)
  * site/portfolio.html   (paper portfolio dashboard)
  * site/archive.html     (every historical day's picks)
  * site/backtest.html    (most recent backtest)

The site is fully self-contained — uses Tailwind via CDN and Plotly via CDN.
Open site/index.html directly in a browser, or serve the folder with
``python -m http.server`` from inside it.
"""

from __future__ import annotations

import json
import logging
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from . import portfolio as portfolio_mod
from . import scorer

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_PICK_FILE_RE = re.compile(r"asx_picks_(\d{4}-\d{2}-\d{2})\.csv$")
_BACKTEST_FILE_RE = re.compile(r"backtest_(\d{4}-\d{2}-\d{2})\.xlsx$")


def _safe_num(v: Any) -> float | None:
    """Convert to a JSON-safe number; None for NaN/inf/empty."""
    try:
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except (TypeError, ValueError):
        return None


def _row_to_dict(row: pd.Series, fields: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for f in fields:
        if f in row.index:
            v = row[f]
            if isinstance(v, (int, float)):
                out[f] = _safe_num(v)
            else:
                out[f] = "" if pd.isna(v) else v
    return out


def _list_pick_files(reports_dir: Path) -> list[tuple[str, Path]]:
    """Return [(date_str, csv_path), ...] sorted descending by date."""
    out = []
    for p in reports_dir.glob("asx_picks_*.csv"):
        m = _PICK_FILE_RE.search(p.name)
        if m:
            out.append((m.group(1), p))
    out.sort(key=lambda pair: pair[0], reverse=True)
    return out


def _latest_backtest(reports_dir: Path) -> Path | None:
    files = []
    for p in reports_dir.glob("backtest_*.xlsx"):
        m = _BACKTEST_FILE_RE.search(p.name)
        if m:
            files.append((m.group(1), p))
    if not files:
        return None
    files.sort(key=lambda pair: pair[0], reverse=True)
    return files[0][1]


# ---------------------------------------------------------------------------
# Page builders
# ---------------------------------------------------------------------------


PICK_FIELDS = [
    "ticker",
    "name",
    "sector",
    "price",
    "composite_score",
    "projected_return",
    "dividend_yield",
    "pe",
    "roe",
    "revenue_growth",
    "return_12m",
    "score_value",
    "score_growth",
    "score_quality",
    "score_income",
    "score_momentum",
    "score_risk",
]


def _today_picks_context(reports_dir: Path, min_projected_return: float) -> dict[str, Any] | None:
    files = _list_pick_files(reports_dir)
    if not files:
        return None
    pick_date, csv_path = files[0]
    today_csv = pd.read_csv(csv_path)

    # The CSV summary doesn't have all pillar fields; if today's xlsx exists, prefer it.
    xlsx_path = csv_path.with_suffix(".xlsx")
    if xlsx_path.exists():
        try:
            picks_df = pd.read_excel(xlsx_path, sheet_name="Top Picks")
            full_df = pd.read_excel(xlsx_path, sheet_name="All Scores")
        except Exception:
            picks_df = today_csv
            full_df = today_csv
    else:
        picks_df = today_csv
        full_df = today_csv

    picks = []
    for _, row in picks_df.iterrows():
        d = _row_to_dict(row, PICK_FIELDS + ["rationale"])
        if not d.get("rationale"):
            d["rationale"] = scorer.build_rationale(row) if "score_value" in row.index else ""
        picks.append(d)

    full_screen = [_row_to_dict(row, PICK_FIELDS) for _, row in full_df.iterrows()]

    return {
        "active": "picks",
        "pick_date": pick_date,
        "picks": picks,
        "full_screen": full_screen,
        "min_projected_return": min_projected_return,
    }


def _portfolio_context(state_path: Path, reports_dir: Path) -> dict[str, Any]:
    if not state_path.exists():
        return {"active": "portfolio", "state": None}

    state = portfolio_mod.PortfolioState.load(state_path)
    if state is None:
        return {"active": "portfolio", "state": None}

    # Reconstruct holdings view from state + last-known prices in pick files
    last_prices: dict[str, float] = {}
    for _, csv_path in _list_pick_files(reports_dir):
        try:
            df = pd.read_csv(csv_path)
            for _, r in df.iterrows():
                t = r.get("ticker")
                p = r.get("price")
                if t and isinstance(p, (int, float)) and not math.isnan(p) and t not in last_prices:
                    last_prices[t] = float(p)
        except Exception:
            continue

    holdings: list[dict[str, Any]] = []
    total = state.cash
    for pos in state.positions:
        price = last_prices.get(pos.ticker, pos.avg_cost)
        market_value = pos.shares * price
        pnl = market_value - pos.shares * pos.avg_cost
        pnl_pct = pnl / (pos.shares * pos.avg_cost) if pos.avg_cost > 0 else 0.0
        holdings.append(
            {
                "ticker": pos.ticker,
                "name": pos.name,
                "sector": pos.sector,
                "shares": pos.shares,
                "avg_cost": pos.avg_cost,
                "price": price,
                "market_value": market_value,
                "weight_pct": 0.0,
                "unrealized_pnl": pnl,
                "unrealized_pnl_pct": pnl_pct,
            }
        )
        total += market_value
    for h in holdings:
        h["weight_pct"] = h["market_value"] / total if total > 0 else 0.0

    latest = state.value_history[-1] if state.value_history else {
        "total_value": total,
        "total_return_pct": (total - state.initial_capital) / state.initial_capital if state.initial_capital else 0.0,
    }

    days_held = (datetime.now() - datetime.strptime(state.created_at, "%Y-%m-%d")).days or 1
    total_return = latest.get("total_return_pct", 0.0)
    annualised = (1 + total_return) ** (365.0 / days_held) - 1 if total_return > -1 else 0.0

    return {
        "active": "portfolio",
        "state": state,
        "holdings": holdings,
        "latest": latest,
        "days_held": days_held,
        "annualised": annualised,
        "rebalance_days": state.rebalance_interval_days,
        "rebalances": state.rebalance_history,
        "value_history_json": json.dumps(state.value_history),
    }


def _archive_context(reports_dir: Path) -> dict[str, Any]:
    archive: list[dict[str, Any]] = []
    for date_str, csv_path in _list_pick_files(reports_dir):
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        picks = [_row_to_dict(row, PICK_FIELDS) for _, row in df.iterrows()]
        archive.append({"date": date_str, "picks": picks})
    return {"active": "archive", "archive": archive}


def _backtest_context(reports_dir: Path) -> dict[str, Any]:
    path = _latest_backtest(reports_dir)
    if path is None:
        return {"active": "backtest", "backtest": None}

    try:
        summary = pd.read_excel(path, sheet_name="Summary")
        equity = pd.read_excel(path, sheet_name="Equity Curve")
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not read backtest %s: %s", path, exc)
        return {"active": "backtest", "backtest": None}

    # Summary is presented as Metric/Value rows of strings. We need numeric metrics
    # for the cards. We re-derive them from the equity curve (which has dates +
    # strategy + benchmark columns).
    if "date" in [c.lower() for c in equity.columns]:
        date_col = [c for c in equity.columns if c.lower() == "date"][0]
        equity = equity.rename(columns={date_col: "date"}).set_index("date")
    else:
        # Most likely first column is the date index
        equity = equity.set_index(equity.columns[0])
    equity.index = pd.to_datetime(equity.index, errors="coerce")
    equity = equity.dropna(how="all")

    strat = equity["strategy"].dropna()
    bench = equity["benchmark"].dropna() if "benchmark" in equity.columns else pd.Series(dtype=float)

    def _cagr(s):
        if len(s) < 2:
            return 0.0
        years = (s.index[-1] - s.index[0]).days / 365.25
        return (s.iloc[-1] / s.iloc[0]) ** (1 / years) - 1 if years > 0 else 0.0

    def _vol(s):
        r = s.pct_change(fill_method=None).dropna()
        return float(r.std() * (252 ** 0.5)) if not r.empty else 0.0

    def _maxdd(s):
        running = s.cummax()
        return float((s / running - 1).min()) if not s.empty else 0.0

    def _sharpe(s, rf=0.03):
        v = _vol(s)
        return (_cagr(s) - rf) / v if v > 0 else 0.0

    metrics = {
        "strategy_cagr": _cagr(strat),
        "benchmark_cagr": _cagr(bench) if not bench.empty else 0.0,
        "strategy_vol": _vol(strat),
        "benchmark_vol": _vol(bench) if not bench.empty else 0.0,
        "strategy_sharpe": _sharpe(strat),
        "benchmark_sharpe": _sharpe(bench) if not bench.empty else 0.0,
        "strategy_max_dd": _maxdd(strat),
        "benchmark_max_dd": _maxdd(bench) if not bench.empty else 0.0,
        "total_return_strategy": (strat.iloc[-1] / strat.iloc[0] - 1) if len(strat) > 1 else 0.0,
        "total_return_benchmark": (bench.iloc[-1] / bench.iloc[0] - 1) if len(bench) > 1 else 0.0,
        "years": (strat.index[-1] - strat.index[0]).days / 365.25 if len(strat) > 1 else 0.0,
    }

    metric_rows = [(str(r["Metric"]), str(r["Value"])) for _, r in summary.iterrows()]

    chart_data = {
        "dates": [d.strftime("%Y-%m-%d") for d in equity.index],
        "strategy": [_safe_num(v) for v in equity["strategy"].tolist()],
        "benchmark": [_safe_num(v) for v in equity["benchmark"].tolist()] if "benchmark" in equity.columns else [],
    }

    return {
        "active": "backtest",
        "backtest": True,
        "metrics": metrics,
        "metric_rows": metric_rows,
        "chart_data_json": json.dumps(chart_data),
        "backtest_file": str(path.name),
    }


# ---------------------------------------------------------------------------
# Build entry point
# ---------------------------------------------------------------------------


def build_site(
    site_dir: Path,
    reports_dir: Path,
    state_path: Path,
    min_projected_return: float = 0.10,
) -> Path:
    """Generate the static site. Returns the path to ``index.html``."""
    try:
        from jinja2 import Environment, PackageLoader, select_autoescape
    except ImportError as exc:
        raise RuntimeError("jinja2 is required to build the site. pip install jinja2") from exc

    env = Environment(
        loader=PackageLoader("asx_portfolio", "templates"),
        autoescape=select_autoescape(["html"]),
    )

    site_dir.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    pages = {
        "index.html": ("picks.html", _today_picks_context(reports_dir, min_projected_return) or {"active": "picks", "picks": [], "full_screen": [], "pick_date": "—", "min_projected_return": min_projected_return}),
        "portfolio.html": ("portfolio.html", _portfolio_context(state_path, reports_dir)),
        "archive.html": ("archive.html", _archive_context(reports_dir)),
        "backtest.html": ("backtest.html", _backtest_context(reports_dir)),
    }

    for out_name, (template_name, ctx) in pages.items():
        ctx["generated_at"] = generated_at
        template = env.get_template(template_name)
        html = template.render(**ctx)
        (site_dir / out_name).write_text(html, encoding="utf-8")
        log.info("Wrote %s", site_dir / out_name)

    return site_dir / "index.html"
