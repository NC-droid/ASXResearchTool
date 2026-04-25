"""Fetch market data + fundamentals for ASX tickers (Yahoo primary)."""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Iterable

import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class StockData:
    ticker: str
    info: dict[str, Any] = field(default_factory=dict)
    history: pd.DataFrame | None = None
    announcements: list[dict[str, Any]] = field(default_factory=list)
    av_overview: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.history is not None and not self.history.empty


def _yf():
    try:
        import yfinance as yf
        return yf
    except ImportError as exc:
        raise RuntimeError("yfinance is required. pip install yfinance") from exc


def fetch_yahoo(ticker: str, period: str = "5y") -> StockData:
    yf = _yf()
    sd = StockData(ticker=ticker)
    try:
        t = yf.Ticker(ticker)
        try:
            sd.info = dict(t.info or {})
        except Exception:
            sd.info = {}
        hist = t.history(period=period, auto_adjust=True)
        if hist is None or hist.empty:
            sd.error = "no price history"
            return sd
        sd.history = hist
    except Exception as exc:  # noqa: BLE001
        sd.error = f"yahoo fetch failed: {exc}"
    return sd


_AV_BASE = "https://www.alphavantage.co/query"


def fetch_alpha_vantage_overview(ticker: str, api_key: str | None = None) -> dict[str, Any]:
    key = api_key or os.environ.get("ALPHAVANTAGE_API_KEY")
    if not key:
        return {}
    try:
        import requests
    except ImportError:
        return {}
    av_symbol = f"ASX:{ticker.split('.')[0]}"
    try:
        r = requests.get(_AV_BASE, params={"function": "OVERVIEW", "symbol": av_symbol, "apikey": key}, timeout=20)
        if r.status_code != 200:
            return {}
        data = r.json() or {}
        return data if data and "Symbol" in data else {}
    except Exception:
        return {}


def fetch_asx_announcements(ticker: str, limit: int = 5) -> list[dict[str, Any]]:
    try:
        import requests
    except ImportError:
        return []
    code = ticker.split(".")[0]
    url = f"https://asx.api.markitdigital.com/asx-research/1.0/companies/{code}/announcements?count={limit}&market_sensitive=false"
    try:
        r = requests.get(url, headers={"User-Agent": "asx-portfolio/0.2 (research)"}, timeout=15)
        if r.status_code != 200:
            return []
        items = (r.json() or {}).get("data", {}).get("items", []) or []
        return [
            {
                "date": i.get("document_date") or i.get("released_at"),
                "title": i.get("header") or i.get("subject"),
                "market_sensitive": bool(i.get("market_sensitive")),
            }
            for i in items[:limit]
        ]
    except Exception:
        return []


def fetch_universe(
    tickers: Iterable[str],
    period: str = "5y",
    workers: int = 8,
    progress_every: int = 25,
) -> dict[str, StockData]:
    tickers = list(tickers)
    results: dict[str, StockData] = {}
    log.info("Fetching Yahoo data for %d tickers (workers=%d, period=%s)", len(tickers), workers, period)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch_yahoo, t, period): t for t in tickers}
        completed = 0
        for fut in as_completed(futures):
            t = futures[fut]
            try:
                sd = fut.result()
            except Exception:
                continue
            completed += 1
            if completed % progress_every == 0:
                log.info("  ...%d/%d done", completed, len(tickers))
            if sd.ok:
                results[t] = sd
    log.info("Fetched usable data for %d / %d tickers", len(results), len(tickers))
    return results


def enrich_with_alpha_vantage(selected: dict[str, StockData], api_key: str | None = None, sleep_between: float = 13.0) -> None:
    for ticker, sd in selected.items():
        sd.av_overview = fetch_alpha_vantage_overview(ticker, api_key=api_key)
        time.sleep(sleep_between)


def enrich_with_announcements(selected: dict[str, StockData], limit: int = 5) -> None:
    for ticker, sd in selected.items():
        sd.announcements = fetch_asx_announcements(ticker, limit=limit)
