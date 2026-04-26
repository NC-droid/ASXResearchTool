"""ASX 200 stock scoring engine — mirrors etf_scorer.py.

Pillars (each cross-sectionally percentile-ranked across the candidate set):
  * **Value**     — earnings yield, FCF yield, P/B (lower P/B = better)
  * **Growth**    — revenue growth, earnings growth, forward EPS growth
  * **Quality**   — ROE, ROA, gross margin, debt/equity (lower = better)
  * **Income**    — dividend yield (with payout-ratio sanity penalty)
  * **Momentum**  — 6m + 12m return; 1m extreme penalty
  * **Risk**      — volatility, max drawdown, distance of beta from 1.0

Default pillar weights match config.yaml in the fundamental screener:
  Value 22%, Growth 22%, Quality 22%, Income 10%, Momentum 14%, Risk 10%

Pillar weights shift slightly based on risk_tolerance and horizon, same
logic as etf_scorer so the two pages feel consistent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .stock_fetcher import StockData


# ---------------------------------------------------------------------------
# Default weights (match scorer.py)
# ---------------------------------------------------------------------------

DEFAULT_WEIGHTS: dict[str, float] = {
    "value": 0.22,
    "growth": 0.22,
    "quality": 0.22,
    "income": 0.10,
    "momentum": 0.14,
    "risk": 0.10,
}


# ---------------------------------------------------------------------------
# User inputs
# ---------------------------------------------------------------------------

@dataclass
class StockScreenInputs:
    risk_tolerance: str = "medium"   # "low" | "medium" | "high"
    horizon: str = "medium"          # "short" | "medium" | "long"
    sectors: list[str] = field(default_factory=list)   # empty = all
    min_dividend_yield: float = 0.0
    min_market_cap_aud_m: float = 0.0   # filter out micro-caps
    top_n: int = 5

    def normalised(self) -> "StockScreenInputs":
        out = StockScreenInputs(
            risk_tolerance=(self.risk_tolerance or "medium").lower(),
            horizon=(self.horizon or "medium").lower(),
            sectors=[s for s in (self.sectors or []) if s],
            min_dividend_yield=max(self.min_dividend_yield or 0.0, 0.0),
            min_market_cap_aud_m=max(self.min_market_cap_aud_m or 0.0, 0.0),
            top_n=max(int(self.top_n or 5), 1),
        )
        if out.risk_tolerance not in ("low", "medium", "high"):
            out.risk_tolerance = "medium"
        if out.horizon not in ("short", "medium", "long"):
            out.horizon = "medium"
        return out


# ---------------------------------------------------------------------------
# Pillar weight logic
# ---------------------------------------------------------------------------

def _pillar_weights(inputs: StockScreenInputs) -> dict[str, float]:
    w = dict(DEFAULT_WEIGHTS)

    # Risk tolerance shifts: low risk → emphasise quality + risk pillar
    if inputs.risk_tolerance == "low":
        w["quality"] += 0.06
        w["risk"] += 0.06
        w["momentum"] -= 0.06
        w["growth"] -= 0.06
    elif inputs.risk_tolerance == "high":
        w["growth"] += 0.08
        w["momentum"] += 0.04
        w["risk"] -= 0.06
        w["quality"] -= 0.06

    # Horizon shifts: short horizon → value + income; long → growth
    if inputs.horizon == "short":
        w["value"] += 0.06
        w["income"] += 0.04
        w["growth"] -= 0.06
        w["momentum"] -= 0.04
    elif inputs.horizon == "long":
        w["growth"] += 0.06
        w["quality"] += 0.04
        w["value"] -= 0.06
        w["income"] -= 0.04

    # Income boost if user requested yield
    if inputs.min_dividend_yield > 0:
        w["income"] += 0.08
        w["value"] -= 0.04
        w["growth"] -= 0.04

    # Normalise to sum = 1
    total = sum(w.values())
    return {k: max(0.0, v / total) for k, v in w.items()}


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _winsorize(s: pd.Series, p: float = 0.05) -> pd.Series:
    lo, hi = s.quantile(p), s.quantile(1 - p)
    return s.clip(lo, hi)


def _percentile_rank(s: pd.Series, higher_better: bool = True) -> pd.Series:
    ranked = s.rank(pct=True, na_option="bottom") * 100
    return ranked if higher_better else (100 - ranked)


def _fillna_median(s: pd.Series) -> pd.Series:
    return s.fillna(s.median())


# ---------------------------------------------------------------------------
# Main scoring function
# ---------------------------------------------------------------------------

def score(stocks: list[StockData], inputs: StockScreenInputs) -> pd.DataFrame:
    """Score and filter stocks, returning a sorted DataFrame."""
    inputs = inputs.normalised()

    rows = []
    for s in stocks:
        rows.append({
            "ticker": s.ticker,
            "name": s.name,
            "sector": s.sector,
            "industry": s.industry,
            "price": s.price,
            "market_cap": s.market_cap,
            "avg_volume": s.avg_volume,
            # Valuation
            "pe_ratio": s.pe_ratio,
            "pb_ratio": s.pb_ratio,
            "ev_ebitda": s.ev_ebitda,
            "earnings_yield": s.earnings_yield,
            "fcf_yield": s.fcf_yield,
            # Growth
            "revenue_growth": s.revenue_growth,
            "earnings_growth": s.earnings_growth,
            "forward_eps_growth": s.forward_eps_growth,
            # Quality
            "roe": s.roe,
            "roa": s.roa,
            "gross_margin": s.gross_margin,
            "debt_equity": s.debt_equity,
            # Income
            "dividend_yield": s.dividend_yield if not math.isnan(s.dividend_yield) else 0.0,
            "payout_ratio": s.payout_ratio,
            # Momentum
            "return_1m": s.return_1m,
            "return_6m": s.return_6m,
            "return_1y": s.return_1y,
            "return_3y": s.return_3y,
            "return_5y": s.return_5y,
            # Risk
            "volatility_1y": s.volatility_1y,
            "max_drawdown_1y": s.max_drawdown_1y,
            "beta": s.beta,
            "sharpe_1y": s.sharpe_1y,
            "projected_return": s.projected_return,
            # Sparkline
            "history_dates": s.history_dates,
            "history_prices": s.history_prices,
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # ---- Hard filters -------------------------------------------------------
    if inputs.sectors:
        df = df[df["sector"].isin(inputs.sectors)]
    if inputs.min_dividend_yield > 0:
        df = df[df["dividend_yield"].fillna(0.0) >= inputs.min_dividend_yield]
    if inputs.min_market_cap_aud_m > 0:
        df = df[df["market_cap"].fillna(0.0) >= inputs.min_market_cap_aud_m * 1e6]

    if df.empty:
        return df.reset_index(drop=True)

    # ---- Pillar 1: Value -----------------------------------------------------
    earnings_y = _percentile_rank(_winsorize(_fillna_median(df["earnings_yield"])), True)
    fcf_y = _percentile_rank(_winsorize(_fillna_median(df["fcf_yield"])), True)
    pb_score = _percentile_rank(_winsorize(_fillna_median(df["pb_ratio"])), False)  # lower P/B = better
    value_score = (earnings_y * 0.4 + fcf_y * 0.35 + pb_score * 0.25)

    # ---- Pillar 2: Growth ---------------------------------------------------
    rev_g = _percentile_rank(_winsorize(_fillna_median(df["revenue_growth"])), True)
    earn_g = _percentile_rank(_winsorize(_fillna_median(df["earnings_growth"])), True)
    fwd_g = _percentile_rank(_winsorize(_fillna_median(df["forward_eps_growth"])), True)
    growth_score = (rev_g * 0.35 + earn_g * 0.40 + fwd_g * 0.25)

    # ---- Pillar 3: Quality --------------------------------------------------
    roe_s = _percentile_rank(_winsorize(_fillna_median(df["roe"])), True)
    roa_s = _percentile_rank(_winsorize(_fillna_median(df["roa"])), True)
    gm_s = _percentile_rank(_winsorize(_fillna_median(df["gross_margin"])), True)
    de_s = _percentile_rank(_winsorize(_fillna_median(df["debt_equity"])), False)  # lower D/E = better
    quality_score = (roe_s * 0.30 + roa_s * 0.25 + gm_s * 0.25 + de_s * 0.20)

    # ---- Pillar 4: Income ---------------------------------------------------
    div_raw = df["dividend_yield"].fillna(0.0)
    # Payout ratio penalty: heavily penalise payout > 90% (unsustainable)
    payout = df["payout_ratio"].fillna(0.6).clip(0, 1)
    penalty = (payout > 0.9).astype(float) * 20
    income_score = _percentile_rank(_winsorize(div_raw), True) - penalty

    # ---- Pillar 5: Momentum -------------------------------------------------
    m6 = _percentile_rank(_winsorize(_fillna_median(df["return_6m"])), True)
    m12 = _percentile_rank(_winsorize(_fillna_median(df["return_1y"])), True)
    # Mean-reversion penalty on extreme 1m returns (both directions)
    m1_abs = df["return_1m"].fillna(0).abs()
    m1_penalty = (m1_abs > m1_abs.quantile(0.9)).astype(float) * 15
    momentum_score = (m6 * 0.45 + m12 * 0.55) - m1_penalty

    # ---- Pillar 6: Risk -----------------------------------------------------
    vol_s = _percentile_rank(_winsorize(_fillna_median(df["volatility_1y"])), False)
    dd_s = _percentile_rank(_winsorize(_fillna_median(df["max_drawdown_1y"])), True)  # less negative = better
    # Beta distance from 1.0 (closer to market = safer for risk pillar)
    beta_dist = (df["beta"].fillna(1.0) - 1.0).abs()
    beta_s = _percentile_rank(_winsorize(beta_dist), False)
    risk_score = (vol_s * 0.40 + dd_s * 0.40 + beta_s * 0.20)

    # ---- Composite ---------------------------------------------------------
    df["score_value"] = value_score
    df["score_growth"] = growth_score
    df["score_quality"] = quality_score
    df["score_income"] = income_score
    df["score_momentum"] = momentum_score
    df["score_risk"] = risk_score

    w = _pillar_weights(inputs)
    df["composite_score"] = (
        value_score * w["value"]
        + growth_score * w["growth"]
        + quality_score * w["quality"]
        + income_score * w["income"]
        + momentum_score * w["momentum"]
        + risk_score * w["risk"]
    )
    df["pillar_weights"] = [w] * len(df)

    return df.sort_values("composite_score", ascending=False).reset_index(drop=True)


def select_top(scored: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    if scored is None or scored.empty:
        return scored
    return scored.head(n).reset_index(drop=True)


def build_rationale(row: pd.Series) -> str:
    bits = []
    if row.get("score_value", 0) >= 70:
        bits.append("attractively valued")
    if row.get("score_growth", 0) >= 70:
        bits.append("strong growth outlook")
    if row.get("score_quality", 0) >= 70:
        bits.append("high-quality business")
    if row.get("score_momentum", 0) >= 70:
        bits.append("positive price momentum")
    if row.get("score_risk", 0) >= 70:
        bits.append("low relative risk")
    if row.get("score_income", 0) >= 70 and row.get("dividend_yield", 0) > 0:
        dy = row["dividend_yield"] * 100
        bits.append(f"{dy:.1f}% dividend yield")
    if not bits:
        bits.append("balanced scorecard")
    sector = row.get("sector", "")
    sector_str = f" ({sector})" if sector else ""
    return f"{row.get('name') or row['ticker']} ({row['ticker']}){sector_str}: " + ", ".join(bits) + "."
