"""ETF risk-adjusted scoring engine, parameterised by user inputs.

The scorer takes a list of ``ETFData`` plus a ``ScreenInputs`` describing the
user's preferences, applies hard filters (asset categories, max MER, min
yield), then computes a composite score whose pillar weights vary based on
risk tolerance and investment horizon.

Pillars (each cross-sectionally percentile-ranked across the candidate set):
  * **Return**     — blended 1y / 3y / 5y return (weighting depends on horizon)
  * **Risk-Adj**   — Sharpe ratio (1y + 3y blend)
  * **Stability**  — volatility + max drawdown (lower = better)
  * **Cost**       — expense ratio (lower = better)
  * **Liquidity**  — AUM (higher = better)
  * **Income**     — dividend yield

Pillar weights flex with the inputs. A "low risk / short horizon" investor
weights stability + cost much higher; "high risk / long horizon" weights
return higher. Income receives extra weight only if the user actually set
a min-yield filter, since otherwise yield isn't a stated preference.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd

from .etf_fetcher import ETFData


# ---------------------------------------------------------------------------
# User inputs
# ---------------------------------------------------------------------------


@dataclass
class ScreenInputs:
    risk_tolerance: str = "medium"          # "low" | "medium" | "high"
    horizon: str = "medium"                 # "short" (1-3y) | "medium" (3-7y) | "long" (7+y)
    categories: list[str] = field(default_factory=list)  # empty = all categories
    max_mer: float | None = None            # e.g. 0.005 for 0.50% p.a.
    min_dividend_yield_annual: float = 0.0  # 0.04 for 4%
    top_n: int = 3

    def normalised(self) -> "ScreenInputs":
        out = ScreenInputs(
            risk_tolerance=(self.risk_tolerance or "medium").lower(),
            horizon=(self.horizon or "medium").lower(),
            categories=[c for c in (self.categories or []) if c],
            max_mer=self.max_mer,
            min_dividend_yield_annual=max(self.min_dividend_yield_annual or 0.0, 0.0),
            top_n=max(int(self.top_n or 3), 1),
        )
        if out.risk_tolerance not in ("low", "medium", "high"):
            out.risk_tolerance = "medium"
        if out.horizon not in ("short", "medium", "long"):
            out.horizon = "medium"
        return out


def _pillar_weights(inputs: ScreenInputs) -> dict[str, float]:
    """Pillar weights tuned by risk tolerance, horizon, and whether income is asked for."""
    # Baseline (medium risk, medium horizon)
    w = {
        "return": 0.30,
        "risk_adj": 0.25,
        "stability": 0.15,
        "cost": 0.15,
        "liquidity": 0.10,
        "income": 0.05,
    }
    # Risk tolerance shifts return vs stability
    if inputs.risk_tolerance == "low":
        w["return"] -= 0.10
        w["risk_adj"] += 0.03
        w["stability"] += 0.10
        w["cost"] += 0.02
        w["liquidity"] += 0.02
        w["income"] -= 0.07
    elif inputs.risk_tolerance == "high":
        w["return"] += 0.10
        w["risk_adj"] -= 0.02
        w["stability"] -= 0.07
        w["cost"] -= 0.02
        w["income"] += 0.01

    # Horizon shifts long-run return weighting (but not the pillar weight itself)
    # — handled in _return_blend below.

    # If user asks for any yield, boost income materially
    if inputs.min_dividend_yield_annual > 0:
        bump = 0.08
        # take it proportionally from return + risk_adj
        w["return"] -= bump * 0.6
        w["risk_adj"] -= bump * 0.4
        w["income"] += bump

    # Renormalise to sum to 1.0 in case rounding drift
    total = sum(w.values())
    return {k: v / total for k, v in w.items()}


def _return_blend(inputs: ScreenInputs, ret_1y: pd.Series, ret_3y: pd.Series, ret_5y: pd.Series) -> pd.Series:
    """Blend the 1y/3y/5y annualised returns into a single return signal."""
    if inputs.horizon == "short":
        a, b, c = 0.60, 0.30, 0.10
    elif inputs.horizon == "long":
        a, b, c = 0.10, 0.35, 0.55
    else:  # medium
        a, b, c = 0.30, 0.40, 0.30
    blended = ret_1y * a + ret_3y * b + ret_5y * c
    # When 3y/5y missing for newer ETFs, fall back to whatever's available
    fallback = ret_1y.where(ret_5y.isna() & ret_3y.isna(), other=blended)
    return blended.fillna(fallback)


def _percentile_rank(s: pd.Series, higher_is_better: bool = True) -> pd.Series:
    if higher_is_better:
        return (s.rank(pct=True) * 100.0).fillna(50.0)
    return ((1 - s.rank(pct=True)) * 100.0).fillna(50.0)


def _winsorize(s: pd.Series, lower: float = 0.02, upper: float = 0.98) -> pd.Series:
    if s.dropna().empty:
        return s
    return s.clip(lower=s.quantile(lower), upper=s.quantile(upper))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def score(etfs: Iterable[ETFData], inputs: ScreenInputs) -> pd.DataFrame:
    """Apply filters + scoring; return a DataFrame sorted descending by composite."""
    inputs = inputs.normalised()

    rows = []
    for e in etfs:
        rows.append({
            "ticker": e.ticker,
            "name": e.name,
            "category": e.category,
            "subcategory": e.subcategory,
            "mer": e.mer,
            "distribution_freq": e.distribution_freq,
            "price": e.price,
            "aum": e.aum,
            "dividend_yield": e.dividend_yield if not math.isnan(e.dividend_yield) else 0.0,
            "return_1m": e.return_1m,
            "return_3m": e.return_3m,
            "return_6m": e.return_6m,
            "return_1y": e.return_1y,
            "return_3y": e.return_3y,
            "return_5y": e.return_5y,
            "volatility_1y": e.volatility_1y,
            "max_drawdown_1y": e.max_drawdown_1y,
            "max_drawdown_3y": e.max_drawdown_3y,
            "sharpe_1y": e.sharpe_1y,
            "sharpe_3y": e.sharpe_3y,
            "history_dates": e.history_dates,
            "history_prices": e.history_prices,
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # ---- Hard filters --------------------------------------------------------
    if inputs.categories:
        df = df[df["category"].isin(inputs.categories)]
    if inputs.max_mer is not None:
        df = df[df["mer"].fillna(1.0) <= inputs.max_mer]
    if inputs.min_dividend_yield_annual > 0:
        df = df[df["dividend_yield"].fillna(0.0) >= inputs.min_dividend_yield_annual]

    if df.empty:
        return df.reset_index(drop=True)

    # ---- Pillar inputs -------------------------------------------------------
    return_blend = _return_blend(inputs, df["return_1y"], df["return_3y"], df["return_5y"])
    sharpe_blend = (df["sharpe_1y"].fillna(0) * 0.4 + df["sharpe_3y"].fillna(0) * 0.6)
    stability_signal = -((df["volatility_1y"].fillna(df["volatility_1y"].median())) * 0.5
                         + df["max_drawdown_1y"].abs().fillna(df["max_drawdown_1y"].abs().median()) * 0.5)

    return_score = _percentile_rank(_winsorize(return_blend), True)
    risk_adj_score = _percentile_rank(_winsorize(sharpe_blend), True)
    stability_score = _percentile_rank(_winsorize(stability_signal), True)
    cost_score = _percentile_rank(_winsorize(df["mer"]), False)
    liquidity_score = _percentile_rank(_winsorize(df["aum"]), True)
    income_score = _percentile_rank(_winsorize(df["dividend_yield"]), True)

    df["score_return"] = return_score
    df["score_risk_adj"] = risk_adj_score
    df["score_stability"] = stability_score
    df["score_cost"] = cost_score
    df["score_liquidity"] = liquidity_score
    df["score_income"] = income_score

    w = _pillar_weights(inputs)
    df["composite_score"] = (
        return_score * w["return"]
        + risk_adj_score * w["risk_adj"]
        + stability_score * w["stability"]
        + cost_score * w["cost"]
        + liquidity_score * w["liquidity"]
        + income_score * w["income"]
    )
    df["pillar_weights"] = [w] * len(df)

    return df.sort_values("composite_score", ascending=False).reset_index(drop=True)


def select_top(scored: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    if scored is None or scored.empty:
        return scored
    return scored.head(n).reset_index(drop=True)


def build_rationale(row: pd.Series) -> str:
    bits = []
    if row.get("score_return", 0) >= 70:
        bits.append("strong returns")
    if row.get("score_risk_adj", 0) >= 70:
        bits.append("excellent risk-adjusted return")
    if row.get("score_stability", 0) >= 70:
        bits.append("low volatility & drawdowns")
    if row.get("score_cost", 0) >= 70:
        bits.append("very low fees")
    if row.get("score_liquidity", 0) >= 70:
        bits.append("highly liquid")
    if row.get("score_income", 0) >= 70 and row.get("dividend_yield", 0) > 0:
        bits.append(f"{row['dividend_yield']*100:.1f}% yield")
    if not bits:
        bits.append("balanced overall scorecard")
    return f"{row.get('name') or row['ticker']} ({row['ticker']}, {row.get('category','')}): " + ", ".join(bits) + "."
