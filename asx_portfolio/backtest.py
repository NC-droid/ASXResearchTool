"""Price-based backtest of momentum + risk pillars."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def download_price_history(tickers: list[str], start: str = "2016-01-01", end: str | None = None, include_benchmark: bool = True):
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("yfinance is required for backtesting") from exc
    all_syms = list(tickers) + (["^AXJO"] if include_benchmark else [])
    log.info("Downloading %d price series from %s ...", len(all_syms), start)
    df = yf.download(all_syms, start=start, end=end, auto_adjust=True, progress=False, group_by="column", threads=True)
    if isinstance(df.columns, pd.MultiIndex):
        if "Close" in df.columns.get_level_values(0):
            closes = df["Close"].copy()
        else:
            closes = df.xs("Close", axis=1, level=-1).copy()
    else:
        closes = df[["Close"]].rename(columns={"Close": all_syms[0]})
    closes = closes.dropna(how="all")
    benchmark = closes["^AXJO"].dropna() if "^AXJO" in closes.columns else pd.Series(dtype=float)
    if "^AXJO" in closes.columns:
        closes = closes.drop(columns=["^AXJO"])
    usable = [c for c in closes.columns if closes[c].dropna().shape[0] >= 260]
    closes = closes[usable]
    return closes, benchmark


def _pct_change_over(close_panel: pd.DataFrame, days: int, asof: pd.Timestamp) -> pd.Series:
    sub = close_panel.loc[:asof].tail(days + 1)
    if len(sub) < days + 1:
        return pd.Series(float("nan"), index=close_panel.columns)
    return sub.iloc[-1] / sub.iloc[0] - 1.0


def _rolling_vol(close_panel: pd.DataFrame, days: int, asof: pd.Timestamp) -> pd.Series:
    sub = close_panel.loc[:asof].tail(days + 1)
    if len(sub) < 21:
        return pd.Series(float("nan"), index=close_panel.columns)
    return sub.pct_change(fill_method=None).dropna(how="all").std() * math.sqrt(252)


def _rolling_max_dd(close_panel: pd.DataFrame, days: int, asof: pd.Timestamp) -> pd.Series:
    sub = close_panel.loc[:asof].tail(days + 1)
    if len(sub) < 21:
        return pd.Series(float("nan"), index=close_panel.columns)
    return (sub / sub.cummax() - 1.0).min()


def _rolling_beta(close_panel: pd.DataFrame, benchmark: pd.Series, days: int, asof: pd.Timestamp) -> pd.Series:
    sub_p = close_panel.loc[:asof].tail(days + 1)
    sub_b = benchmark.loc[:asof].tail(days + 1)
    if len(sub_p) < 60 or len(sub_b) < 60:
        return pd.Series(float("nan"), index=close_panel.columns)
    p_ret = sub_p.pct_change(fill_method=None).dropna(how="all")
    b_ret = sub_b.pct_change(fill_method=None).dropna()
    aligned = p_ret.join(b_ret.rename("bench"), how="inner")
    bench_var = aligned["bench"].var()
    if not bench_var or np.isnan(bench_var):
        return pd.Series(float("nan"), index=close_panel.columns)
    return pd.Series({c: aligned[c].cov(aligned["bench"]) / bench_var for c in close_panel.columns if c in aligned.columns})


def _percentile_rank(s: pd.Series, higher_is_better: bool = True) -> pd.Series:
    if higher_is_better:
        return s.rank(pct=True).fillna(0.5) * 100.0
    return (1 - s.rank(pct=True).fillna(0.5)) * 100.0


def score_on_date(close_panel: pd.DataFrame, benchmark: pd.Series, asof: pd.Timestamp) -> pd.DataFrame:
    ret_6m = _pct_change_over(close_panel, 126, asof)
    ret_12m = _pct_change_over(close_panel, 252, asof)
    ret_1m = _pct_change_over(close_panel, 21, asof).abs()
    vol = _rolling_vol(close_panel, 252, asof)
    mdd = _rolling_max_dd(close_panel, 252, asof).abs()
    beta = _rolling_beta(close_panel, benchmark, 252, asof)
    momentum = _percentile_rank(ret_6m, True) * 0.5 + _percentile_rank(ret_12m, True) * 0.4 + _percentile_rank(ret_1m, False) * 0.1
    risk = _percentile_rank(vol, False) * 0.45 + _percentile_rank(mdd, False) * 0.35 + _percentile_rank((beta - 1.0).abs(), False) * 0.20
    composite = momentum * 0.6 + risk * 0.4
    df = pd.DataFrame({
        "ticker": close_panel.columns,
        "ret_6m": ret_6m.reindex(close_panel.columns).values,
        "ret_12m": ret_12m.reindex(close_panel.columns).values,
        "vol": vol.reindex(close_panel.columns).values,
        "max_drawdown": (-mdd).reindex(close_panel.columns).values,
        "beta": beta.reindex(close_panel.columns).values,
        "momentum": momentum.reindex(close_panel.columns).values,
        "risk": risk.reindex(close_panel.columns).values,
        "composite": composite.reindex(close_panel.columns).values,
    }).dropna(subset=["ret_12m", "vol"])
    return df.sort_values("composite", ascending=False).reset_index(drop=True)


@dataclass
class BacktestResult:
    equity_curve: pd.DataFrame
    trades: pd.DataFrame
    metrics: dict[str, float] = field(default_factory=dict)


def _rebalance_dates(price_index: pd.DatetimeIndex, interval_days: int = 30) -> list[pd.Timestamp]:
    if len(price_index) == 0:
        return []
    target = price_index[0] + pd.Timedelta(days=300)
    out = []
    while target <= price_index[-1]:
        snapped = price_index[price_index >= target]
        if len(snapped) == 0:
            break
        out.append(snapped[0])
        target = snapped[0] + pd.Timedelta(days=interval_days)
    return out


def run_backtest(close_panel: pd.DataFrame, benchmark: pd.Series, top_n: int = 5, rebalance_interval_days: int = 30, initial_capital: float = 10_000.0) -> BacktestResult:
    if close_panel.empty or benchmark.empty:
        raise ValueError("need price data and benchmark")
    dates = _rebalance_dates(close_panel.index, rebalance_interval_days)
    log.info("Backtest: %d rebalances", len(dates))
    equity = initial_capital
    points = []
    trades = []
    holdings: dict[str, float] = {}
    rebal_set = set(dates)
    prev_close: pd.Series | None = None
    for day in close_panel.index[close_panel.index >= dates[0]]:
        close_row = close_panel.loc[day]
        if holdings:
            mv = 0.0
            for t, sh in holdings.items():
                price = close_row.get(t, np.nan)
                if pd.isna(price) and prev_close is not None:
                    price = prev_close.get(t, np.nan)
                if not pd.isna(price):
                    mv += sh * price
            equity = mv
        points.append({"date": day, "strategy": equity, "benchmark": float(benchmark.loc[day]) if day in benchmark.index else np.nan})
        if day in rebal_set:
            scored = score_on_date(close_panel, benchmark, day)
            picks = scored.head(top_n)["ticker"].tolist()
            new_h: dict[str, float] = {}
            per = equity / max(len(picks), 1)
            for t in picks:
                price = close_row.get(t, np.nan)
                if not pd.isna(price):
                    new_h[t] = per / price
            holdings = new_h
            trades.append({"date": day, "picks": picks, "equity": round(equity, 2)})
        prev_close = close_row
    eq = pd.DataFrame(points).set_index("date")
    first_b = eq["benchmark"].dropna().iloc[0]
    eq["benchmark"] = eq["benchmark"] / first_b * initial_capital
    eq = eq.dropna(subset=["strategy", "benchmark"])
    metrics = compute_metrics(eq["strategy"], eq["benchmark"])
    log.info("Backtest: strategy CAGR=%.2f%%, benchmark CAGR=%.2f%%", metrics["strategy_cagr"]*100, metrics["benchmark_cagr"]*100)
    return BacktestResult(equity_curve=eq, trades=pd.DataFrame(trades), metrics=metrics)


def compute_metrics(strategy: pd.Series, benchmark: pd.Series) -> dict[str, float]:
    def cagr(s):
        if s.empty:
            return 0.0
        years = (s.index[-1] - s.index[0]).days / 365.25
        return (s.iloc[-1] / s.iloc[0]) ** (1 / years) - 1 if years > 0 else 0.0

    def vol(s):
        r = s.pct_change(fill_method=None).dropna()
        return float(r.std() * math.sqrt(252)) if not r.empty else 0.0

    def maxdd(s):
        return float((s / s.cummax() - 1).min()) if not s.empty else 0.0

    def sharpe(s, rf=0.03):
        v = vol(s)
        return (cagr(s) - rf) / v if v > 0 else 0.0

    sm = strategy.resample("ME").last().pct_change().dropna()
    bm = benchmark.resample("ME").last().pct_change().dropna()
    aligned = pd.concat([sm, bm], axis=1, join="inner")
    aligned.columns = ["strat", "bench"]
    hit = float((aligned["strat"] > aligned["bench"]).mean()) if not aligned.empty else 0.0
    return {
        "strategy_cagr": cagr(strategy),
        "benchmark_cagr": cagr(benchmark),
        "strategy_vol": vol(strategy),
        "benchmark_vol": vol(benchmark),
        "strategy_sharpe": sharpe(strategy),
        "benchmark_sharpe": sharpe(benchmark),
        "strategy_max_dd": maxdd(strategy),
        "benchmark_max_dd": maxdd(benchmark),
        "hit_rate_monthly": hit,
        "total_return_strategy": strategy.iloc[-1] / strategy.iloc[0] - 1 if len(strategy) else 0.0,
        "total_return_benchmark": benchmark.iloc[-1] / benchmark.iloc[0] - 1 if len(benchmark) else 0.0,
        "years": (strategy.index[-1] - strategy.index[0]).days / 365.25 if len(strategy) else 0.0,
    }


def write_backtest_report(result: BacktestResult, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        ("Period (years)", f"{result.metrics['years']:.2f}"),
        ("Strategy CAGR", f"{result.metrics['strategy_cagr']*100:.2f}%"),
        ("Benchmark CAGR (ASX 200)", f"{result.metrics['benchmark_cagr']*100:.2f}%"),
        ("Strategy total return", f"{result.metrics['total_return_strategy']*100:.2f}%"),
        ("Benchmark total return", f"{result.metrics['total_return_benchmark']*100:.2f}%"),
        ("Strategy ann. vol", f"{result.metrics['strategy_vol']*100:.2f}%"),
        ("Benchmark ann. vol", f"{result.metrics['benchmark_vol']*100:.2f}%"),
        ("Strategy Sharpe (rf=3%)", f"{result.metrics['strategy_sharpe']:.2f}"),
        ("Benchmark Sharpe (rf=3%)", f"{result.metrics['benchmark_sharpe']:.2f}"),
        ("Strategy max drawdown", f"{result.metrics['strategy_max_dd']*100:.2f}%"),
        ("Benchmark max drawdown", f"{result.metrics['benchmark_max_dd']*100:.2f}%"),
        ("Monthly hit-rate vs benchmark", f"{result.metrics['hit_rate_monthly']*100:.1f}%"),
    ]
    summary_df = pd.DataFrame(rows, columns=["Metric", "Value"])
    caveat_df = pd.DataFrame({"Caveats": [
        "This backtest uses only the price-based pillars of the screen (momentum + risk).",
        "Fundamental pillars are not tested historically (no point-in-time fundamentals).",
        "Past performance does not guarantee future returns.",
    ]})
    try:
        with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
            summary_df.to_excel(writer, sheet_name="Summary", index=False)
            result.equity_curve.to_excel(writer, sheet_name="Equity Curve")
            result.trades.to_excel(writer, sheet_name="Trades", index=False)
            caveat_df.to_excel(writer, sheet_name="Caveats", index=False)
            wb = writer.book
            n = len(result.equity_curve)
            if n > 1:
                chart = wb.add_chart({"type": "line"})
                chart.add_series({"name": "Strategy", "categories": ["Equity Curve", 1, 0, n, 0], "values": ["Equity Curve", 1, 1, n, 1], "line": {"color": "#2E86AB"}})
                chart.add_series({"name": "ASX 200", "categories": ["Equity Curve", 1, 0, n, 0], "values": ["Equity Curve", 1, 2, n, 2], "line": {"color": "#A23B72"}})
                chart.set_title({"name": "Backtest: $10,000 Growth"})
                writer.sheets["Equity Curve"].insert_chart("E2", chart, {"x_scale": 1.4, "y_scale": 1.3})
    except ImportError:
        with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
            summary_df.to_excel(writer, sheet_name="Summary", index=False)
            result.equity_curve.to_excel(writer, sheet_name="Equity Curve")
            result.trades.to_excel(writer, sheet_name="Trades", index=False)
            caveat_df.to_excel(writer, sheet_name="Caveats", index=False)
    log.info("Wrote backtest report: %s", out_path)
    return out_path
