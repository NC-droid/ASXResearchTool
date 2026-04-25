"""ASX 200 ticker universe (with .AX suffix)."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

_WIKIPEDIA_URL = "https://en.wikipedia.org/wiki/S%26P/ASX_200"


def _read_static_csv() -> pd.DataFrame:
    csv_path = Path(__file__).parent / "data" / "asx200_tickers.csv"
    df = pd.read_csv(csv_path)
    df = df.drop_duplicates(subset="ticker").reset_index(drop=True)
    return df


def load_asx200(refresh: bool = False) -> pd.DataFrame:
    """Return DataFrame with columns ticker, name, sector. Falls back to static CSV."""
    static = _read_static_csv()
    if not refresh:
        return static
    try:
        tables = pd.read_html(_WIKIPEDIA_URL)
        candidate = None
        for tbl in tables:
            cols = [str(c).lower() for c in tbl.columns]
            if any("code" in c or "ticker" in c or "symbol" in c for c in cols):
                candidate = tbl
                break
        if candidate is None:
            return static
        rename_map = {}
        for c in candidate.columns:
            lc = str(c).lower()
            if "code" in lc or "ticker" in lc or "symbol" in lc:
                rename_map[c] = "code"
            elif "company" in lc or "name" in lc:
                rename_map[c] = "name"
            elif "sector" in lc:
                rename_map[c] = "sector"
        candidate = candidate.rename(columns=rename_map)
        if "code" not in candidate.columns:
            return static
        candidate["ticker"] = candidate["code"].astype(str).str.upper().str.strip() + ".AX"
        keep = [c for c in ["ticker", "name", "sector"] if c in candidate.columns]
        live = candidate[keep].drop_duplicates(subset="ticker")
        merged = live.merge(static, on="ticker", how="left", suffixes=("", "_static"))
        if "name" in merged.columns and "name_static" in merged.columns:
            merged["name"] = merged["name"].fillna(merged["name_static"])
        if "sector" in merged.columns and "sector_static" in merged.columns:
            merged["sector"] = merged["sector"].fillna(merged["sector_static"])
        cols = [c for c in ["ticker", "name", "sector"] if c in merged.columns]
        return merged[cols].reset_index(drop=True)
    except Exception as exc:  # noqa: BLE001
        log.warning("Wikipedia refresh failed (%s); using static list", exc)
        return static


def tickers(refresh: bool = False) -> list[str]:
    return load_asx200(refresh=refresh)["ticker"].tolist()
