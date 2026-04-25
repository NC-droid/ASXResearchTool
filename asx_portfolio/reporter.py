"""Report generator — Excel + CSV outputs for the daily picks."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)


SUMMARY_COLS = [
    "ticker", "name", "sector", "price", "composite_score", "projected_return",
    "dividend_yield", "pe", "roe", "revenue_growth", "return_12m",
]
DETAIL_COLS = SUMMARY_COLS + [
    "market_cap", "forward_pe", "pb", "earnings_yield", "fcf_yield",
    "earnings_growth", "eps_forward_growth", "roa", "gross_margin",
    "debt_to_equity", "payout_ratio", "return_1m", "return_6m",
    "volatility_1y", "max_drawdown_1y", "beta",
    "score_value", "score_growth", "score_quality",
    "score_income", "score_momentum", "score_risk",
]


def _ensure_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df[[c for c in cols if c in df.columns]].copy()


def write_csv(picks: pd.DataFrame, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _ensure_cols(picks, SUMMARY_COLS).to_csv(out_path, index=False)
    log.info("Wrote CSV: %s", out_path)
    return out_path


def write_excel(picks: pd.DataFrame, full_screen: pd.DataFrame, rationales: list[str], out_path: Path, methodology_text: str = "") -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    picks_view = _ensure_cols(picks, DETAIL_COLS)
    if rationales and len(rationales) == len(picks_view):
        picks_view.insert(loc=min(3, len(picks_view.columns)), column="rationale", value=rationales)
    full_view = _ensure_cols(full_screen, DETAIL_COLS)
    methodology_df = pd.DataFrame({"Methodology": (methodology_text or _default_methodology()).split("\n")})

    try:
        with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
            picks_view.to_excel(writer, sheet_name="Top Picks", index=False)
            full_view.to_excel(writer, sheet_name="All Scores", index=False)
            methodology_df.to_excel(writer, sheet_name="Methodology", index=False)
            wb = writer.book
            pct = wb.add_format({"num_format": "0.0%"})
            num = wb.add_format({"num_format": "#,##0.00"})
            bold = wb.add_format({"bold": True, "bg_color": "#F4F4F4"})
            for sheet, frame in (("Top Picks", picks_view), ("All Scores", full_view)):
                ws = writer.sheets[sheet]
                ws.set_row(0, None, bold)
                for i, c in enumerate(frame.columns):
                    fmt = pct if c in {"projected_return","dividend_yield","earnings_yield","fcf_yield","revenue_growth","earnings_growth","eps_forward_growth","roe","roa","gross_margin","payout_ratio","return_1m","return_6m","return_12m","volatility_1y","max_drawdown_1y"} else (num if c in {"price","pe","forward_pe","pb","debt_to_equity","beta","composite_score","score_value","score_growth","score_quality","score_income","score_momentum","score_risk"} else None)
                    width = 60 if c == "rationale" else (28 if c == "name" else 16)
                    ws.set_column(i, i, width, fmt)
                ws.freeze_panes(1, 0)
    except ImportError:
        with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
            picks_view.to_excel(writer, sheet_name="Top Picks", index=False)
            full_view.to_excel(writer, sheet_name="All Scores", index=False)
            methodology_df.to_excel(writer, sheet_name="Methodology", index=False)
    log.info("Wrote Excel: %s", out_path)
    return out_path


def _default_methodology() -> str:
    return (
        "ASX 200 Daily Picks — Methodology\n\n"
        "Universe: S&P/ASX 200 constituents.\n\n"
        "Scoring pillars (cross-sectional percentile rank, 0–100):\n"
        "  Value    — earnings yield, FCF yield, P/B\n"
        "  Growth   — revenue/earnings growth, forward EPS growth\n"
        "  Quality  — ROE, ROA, gross margin, debt/equity\n"
        "  Income   — dividend yield, payout-ratio sanity check\n"
        "  Momentum — 6m + 12m return, with 1m extreme penalty\n"
        "  Risk     — volatility, max drawdown, beta distance from 1\n\n"
        "Composite = weighted sum of pillars (default: value 22%, growth 22%, quality 22%, income 10%, momentum 14%, risk 10%).\n\n"
        "Projected long-term annualised return = dividend_yield + sustainable_growth + valuation_reversion.\n\n"
        "This is a transparent diagnostic, NOT a forecast.\n"
        "Picks are filtered to projected return ≥ 10% p.a. and limited to two per sector for diversification.\n\n"
        "Disclaimer: Personal research only. NOT financial advice."
    )


def make_default_paths(reports_dir: Path) -> tuple[Path, Path]:
    stamp = datetime.now().strftime("%Y-%m-%d")
    return (reports_dir / f"asx_picks_{stamp}.csv", reports_dir / f"asx_picks_{stamp}.xlsx")
