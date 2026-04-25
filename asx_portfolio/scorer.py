"""Fundamental scoring engine for long-term ASX picks (six-pillar composite)."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

DEFAULT_WEIGHTS = {
    "value": 0.22,
    "growth": 0.22,
    "quality": 0.22,
    "income": 0.10,
    "momentum": 0.14,
    "risk": 0.10,
}


@dataclass
class TickerMetrics:
    ticker: str
    name: str = ""
    sector: str = ""
    price: float = float("nan")
    market_cap: float = float("nan")
    pe: float = float("nan")
    forward_pe: float = float("nan")
    pb: float = float("nan")
    earnings_yield: float = float("nan")
    fcf_yield: float = float("nan")
    revenue_growth: float = float("nan")
    earnings_growth: float = float("nan")
    eps_forward_growth: float = float("nan")
    roe: float = float("nan")
    roa: float = float("nan")
    gross_margin: float = float("nan")
    debt_to_equity: float = float("nan")
    dividend_yield: float = float("nan")
    payout_ratio: float = float("nan")
    return_1m: float = float("nan")
    return_6m: float = float("nan")
    return_12m: float = float("nan")
    volatility_1y: float = float("nan")
    max_drawdown_1y: float = float("nan")
    beta: float = float("nan")
    notes: list[str] = field(default_factory=list)


def _safe(d: dict, key: str, default: float = float("nan")) -> float:
    val = d.get(key)
    if val is None:
        return default
    try:
        v = float(val)
        if math.isinf(v) or math.isnan(v):
            return default
        return v
    except (TypeError, ValueError):
        return default


def _pct_return(history: pd.DataFrame, days: int) -> float:
    if history is None or history.empty or "Close" not in history.columns:
        return float("nan")
    close = history["Close"].dropna()
    if len(close) < 2:
        return float("nan")
    end = close.iloc[-1]
    target_idx = max(0, len(close) - days - 1)
    start = close.iloc[target_idx]
    if start == 0 or pd.isna(start):
        return float("nan")
    return float(end / start - 1.0)


def _max_drawdown(history: pd.DataFrame, days: int = 252) -> float:
    if history is None or history.empty or "Close" not in history.columns:
        return float("nan")
    close = history["Close"].dropna().tail(days)
    if len(close) < 2:
        return float("nan")
    return float((close / close.cummax() - 1.0).min())


def _annualized_volatility(history: pd.DataFrame, days: int = 252) -> float:
    if history is None or history.empty or "Close" not in history.columns:
        return float("nan")
    close = history["Close"].dropna().tail(days)
    if len(close) < 20:
        return float("nan")
    return float(close.pct_change().dropna().std() * math.sqrt(252))


def metrics_from_stockdata(sd, name: str = "", sector: str = "") -> TickerMetrics:
    info = sd.info or {}
    history = sd.history
    m = TickerMetrics(ticker=sd.ticker, name=name or info.get("longName", ""), sector=sector or info.get("sector", ""))
    m.price = _safe(info, "currentPrice", _safe(info, "regularMarketPrice"))
    m.market_cap = _safe(info, "marketCap")
    m.pe = _safe(info, "trailingPE")
    m.forward_pe = _safe(info, "forwardPE")
    m.pb = _safe(info, "priceToBook")
    if not math.isnan(m.pe) and m.pe > 0:
        m.earnings_yield = 1.0 / m.pe
    elif not math.isnan(m.forward_pe) and m.forward_pe > 0:
        m.earnings_yield = 1.0 / m.forward_pe
    fcf = _safe(info, "freeCashflow")
    if not math.isnan(fcf) and not math.isnan(m.market_cap) and m.market_cap > 0:
        m.fcf_yield = fcf / m.market_cap
    m.revenue_growth = _safe(info, "revenueGrowth")
    m.earnings_growth = _safe(info, "earningsGrowth")
    m.eps_forward_growth = _safe(info, "earningsQuarterlyGrowth")
    m.roe = _safe(info, "returnOnEquity")
    m.roa = _safe(info, "returnOnAssets")
    m.gross_margin = _safe(info, "grossMargins")
    m.debt_to_equity = _safe(info, "debtToEquity")
    if not math.isnan(m.debt_to_equity) and m.debt_to_equity > 5:
        m.debt_to_equity /= 100.0
    m.dividend_yield = _safe(info, "dividendYield")
    if not math.isnan(m.dividend_yield) and m.dividend_yield > 1:
        m.dividend_yield /= 100.0
    m.payout_ratio = _safe(info, "payoutRatio")
    m.beta = _safe(info, "beta")
    m.return_1m = _pct_return(history, 21)
    m.return_6m = _pct_return(history, 126)
    m.return_12m = _pct_return(history, 252)
    m.volatility_1y = _annualized_volatility(history)
    m.max_drawdown_1y = _max_drawdown(history)
    return m


def _percentile_rank(s: pd.Series, higher_is_better: bool = True) -> pd.Series:
    if higher_is_better:
        ranked = s.rank(pct=True) * 100.0
    else:
        ranked = (1 - s.rank(pct=True)) * 100.0
    return ranked.fillna(50.0)


def _winsorize(s: pd.Series, lower: float = 0.02, upper: float = 0.98) -> pd.Series:
    if s.dropna().empty:
        return s
    return s.clip(lower=s.quantile(lower), upper=s.quantile(upper))


def score_universe(metrics: list[TickerMetrics], weights: dict[str, float] | None = None) -> pd.DataFrame:
    if not metrics:
        return pd.DataFrame()
    weights = {**DEFAULT_WEIGHTS, **(weights or {})}
    df = pd.DataFrame([m.__dict__ for m in metrics])

    value_score = (
        _percentile_rank(_winsorize(df["earnings_yield"]), True) * 0.5
        + _percentile_rank(_winsorize(df["fcf_yield"]), True) * 0.3
        + _percentile_rank(_winsorize(df["pb"]), False) * 0.2
    )
    growth_score = (
        _percentile_rank(_winsorize(df["revenue_growth"]), True) * 0.4
        + _percentile_rank(_winsorize(df["earnings_growth"]), True) * 0.3
        + _percentile_rank(_winsorize(df["eps_forward_growth"]), True) * 0.3
    )
    quality_score = (
        _percentile_rank(_winsorize(df["roe"]), True) * 0.35
        + _percentile_rank(_winsorize(df["roa"]), True) * 0.25
        + _percentile_rank(_winsorize(df["gross_margin"]), True) * 0.2
        + _percentile_rank(_winsorize(df["debt_to_equity"]), False) * 0.2
    )
    payout_penalty = df["payout_ratio"].apply(lambda x: 0.0 if pd.isna(x) else (1.0 if 0 <= x <= 0.85 else 0.5))
    income_score = _percentile_rank(_winsorize(df["dividend_yield"]), True) * payout_penalty
    momentum_score = (
        _percentile_rank(_winsorize(df["return_6m"]), True) * 0.5
        + _percentile_rank(_winsorize(df["return_12m"]), True) * 0.4
        + _percentile_rank(_winsorize(df["return_1m"].abs()), False) * 0.1
    )
    beta_dist = (df["beta"] - 1.0).abs()
    risk_score = (
        _percentile_rank(_winsorize(df["volatility_1y"]), False) * 0.45
        + _percentile_rank(_winsorize(df["max_drawdown_1y"].abs()), False) * 0.35
        + _percentile_rank(_winsorize(beta_dist), False) * 0.20
    )

    df["score_value"] = value_score
    df["score_growth"] = growth_score
    df["score_quality"] = quality_score
    df["score_income"] = income_score
    df["score_momentum"] = momentum_score
    df["score_risk"] = risk_score

    df["composite_score"] = (
        df["score_value"] * weights["value"]
        + df["score_growth"] * weights["growth"]
        + df["score_quality"] * weights["quality"]
        + df["score_income"] * weights["income"]
        + df["score_momentum"] * weights["momentum"]
        + df["score_risk"] * weights["risk"]
    )

    def projected_return(row: pd.Series) -> float:
        dy = row["dividend_yield"] if not pd.isna(row["dividend_yield"]) else 0.0
        gs = [row.get("revenue_growth"), row.get("earnings_growth"), row.get("eps_forward_growth")]
        gs = [g for g in gs if g is not None and not pd.isna(g)]
        sustainable = max(min(np.mean(gs), 0.15), -0.05) if gs else 0.02
        ey = row.get("earnings_yield")
        reversion = 0.0
        if ey is not None and not pd.isna(ey):
            reversion = max(min((ey - 0.06) * 0.5, 0.04), -0.04)
        return float(dy + sustainable + reversion)

    df["projected_return"] = df.apply(projected_return, axis=1)
    return df.sort_values("composite_score", ascending=False).reset_index(drop=True)


def select_top_picks(
    scored: pd.DataFrame,
    n: int = 5,
    min_projected_return: float = 0.10,
    require_positive_growth: bool = True,
    diversify_by_sector: bool = True,
) -> pd.DataFrame:
    if scored is None or scored.empty:
        return scored
    df = scored[scored["projected_return"] >= min_projected_return].copy()
    if require_positive_growth:
        df = df[(df["revenue_growth"].fillna(0) > -0.05) | (df["earnings_growth"].fillna(0) > -0.05)]
    if df.empty:
        df = scored.copy()
    if diversify_by_sector and "sector" in df.columns:
        picks = []
        sec_count: dict[str, int] = {}
        for _, row in df.sort_values("composite_score", ascending=False).iterrows():
            sec = row.get("sector") or "Unknown"
            if sec_count.get(sec, 0) >= 2:
                continue
            picks.append(row)
            sec_count[sec] = sec_count.get(sec, 0) + 1
            if len(picks) >= n:
                break
        return pd.DataFrame(picks).reset_index(drop=True)
    return df.sort_values("composite_score", ascending=False).head(n).reset_index(drop=True)


def build_rationale(row: pd.Series) -> str:
    bits = []
    if row.get("score_value", 0) >= 70:
        bits.append("attractively valued")
    if row.get("score_growth", 0) >= 70:
        bits.append("strong growth profile")
    if row.get("score_quality", 0) >= 70:
        bits.append("high-quality balance sheet & returns")
    if row.get("score_income", 0) >= 70 and not pd.isna(row.get("dividend_yield", float("nan"))):
        bits.append(f"solid {row['dividend_yield']*100:.1f}% yield")
    if row.get("score_momentum", 0) >= 70:
        bits.append("positive medium-term momentum")
    if row.get("score_risk", 0) >= 70:
        bits.append("contained volatility/drawdowns")
    if not bits:
        bits.append("balanced overall scorecard with no single weak pillar")
    return (
        f"{row.get('name') or row['ticker']} ({row['ticker']}, {row.get('sector') or 'n/a'}): "
        + ", ".join(bits)
        + f". Projected long-term return ≈ {row['projected_return']*100:.1f}% p.a."
    )
