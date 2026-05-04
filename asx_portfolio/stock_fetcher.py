"""ASX 200 stock data fetcher — mirrors etf_fetcher.py.

Fetches fundamental + price data for ASX 200 constituents via yfinance,
caches results in-memory with a configurable TTL, and exposes a
``get_stock_data()`` function consumed by stock_scorer and web_app.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class StockData:
    ticker: str
    name: str = ""
    sector: str = ""
    industry: str = ""

    # Price / market data
    price: float = float("nan")
    market_cap: float = float("nan")
    avg_volume: float = float("nan")

    # Valuation
    pe_ratio: float = float("nan")
    pb_ratio: float = float("nan")
    ev_ebitda: float = float("nan")
    earnings_yield: float = float("nan")   # 1 / PE
    fcf_yield: float = float("nan")

    # Growth
    revenue_growth: float = float("nan")   # YoY %
    earnings_growth: float = float("nan")  # YoY %
    forward_eps_growth: float = float("nan")

    # Quality
    roe: float = float("nan")
    roa: float = float("nan")
    gross_margin: float = float("nan")
    debt_equity: float = float("nan")

    # Income
    dividend_yield: float = float("nan")
    payout_ratio: float = float("nan")

    # Price momentum / risk
    return_1m: float = float("nan")
    return_3m: float = float("nan")
    return_6m: float = float("nan")
    return_1y: float = float("nan")
    return_3y: float = float("nan")
    return_5y: float = float("nan")
    volatility_1y: float = float("nan")
    max_drawdown_1y: float = float("nan")
    beta: float = float("nan")
    sharpe_1y: float = float("nan")

    # Projected return (Gordon-style)
    projected_return: float = float("nan")

    # Sparkline
    history_dates: list[str] = field(default_factory=list)
    history_prices: list[float] = field(default_factory=list)

    fetched_at: float = 0.0
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error and not math.isnan(self.price)


# ---------------------------------------------------------------------------
# ASX 200 ticker universe (static list — refreshed via universe.py if needed)
# ---------------------------------------------------------------------------

_TICKERS_CSV = Path(__file__).parent / "data" / "asx200_tickers.csv"

# FIX: removed NCM.AX (Newcrest — delisted after Newmont merger); replaced with
# current ASX 200 constituents as of 2025.
_FALLBACK_TICKERS: list[dict[str, str]] = [
    {"ticker": "BHP.AX",  "name": "BHP Group",                      "sector": "Materials"},
    {"ticker": "CBA.AX",  "name": "Commonwealth Bank",              "sector": "Financials"},
    {"ticker": "CSL.AX",  "name": "CSL Limited",                    "sector": "Healthcare"},
    {"ticker": "NAB.AX",  "name": "National Australia Bank",        "sector": "Financials"},
    {"ticker": "WBC.AX",  "name": "Westpac Banking",                "sector": "Financials"},
    {"ticker": "ANZ.AX",  "name": "ANZ Group",                      "sector": "Financials"},
    {"ticker": "WES.AX",  "name": "Wesfarmers",                     "sector": "Consumer Discretionary"},
    {"ticker": "RIO.AX",  "name": "Rio Tinto",                      "sector": "Materials"},
    {"ticker": "MQG.AX",  "name": "Macquarie Group",                "sector": "Financials"},
    {"ticker": "WOW.AX",  "name": "Woolworths Group",               "sector": "Consumer Staples"},
    {"ticker": "TLS.AX",  "name": "Telstra Group",                  "sector": "Communication Services"},
    {"ticker": "FMG.AX",  "name": "Fortescue Ltd",                  "sector": "Materials"},
    {"ticker": "GMG.AX",  "name": "Goodman Group",                  "sector": "Real Estate"},
    {"ticker": "TCL.AX",  "name": "Transurban Group",               "sector": "Industrials"},
    {"ticker": "ALL.AX",  "name": "Aristocrat Leisure",             "sector": "Consumer Discretionary"},
    {"ticker": "REA.AX",  "name": "REA Group",                      "sector": "Technology"},
    {"ticker": "COL.AX",  "name": "Coles Group",                    "sector": "Consumer Staples"},
    {"ticker": "QAN.AX",  "name": "Qantas Airways",                 "sector": "Industrials"},
    {"ticker": "AGL.AX",  "name": "AGL Energy",                     "sector": "Utilities"},
    {"ticker": "AMP.AX",  "name": "AMP Limited",                    "sector": "Financials"},
    {"ticker": "ASX.AX",  "name": "ASX Limited",                    "sector": "Financials"},
    {"ticker": "CPU.AX",  "name": "Computershare",                  "sector": "Technology"},
    {"ticker": "IAG.AX",  "name": "Insurance Australia Group",      "sector": "Financials"},
    {"ticker": "MIN.AX",  "name": "Mineral Resources",              "sector": "Materials"},
    {"ticker": "NXT.AX",  "name": "NextDC",                         "sector": "Technology"},
    {"ticker": "PLS.AX",  "name": "Pilbara Minerals",               "sector": "Materials"},
    {"ticker": "QBE.AX",  "name": "QBE Insurance Group",            "sector": "Financials"},
    {"ticker": "RMD.AX",  "name": "ResMed Inc",                     "sector": "Healthcare"},
    {"ticker": "SCG.AX",  "name": "Scentre Group",                  "sector": "Real Estate"},
    {"ticker": "SEK.AX",  "name": "Seek Limited",                   "sector": "Technology"},
    {"ticker": "SHL.AX",  "name": "Sonic Healthcare",               "sector": "Healthcare"},
    {"ticker": "SUN.AX",  "name": "Suncorp Group",                  "sector": "Financials"},
    {"ticker": "XRO.AX",  "name": "Xero Limited",                   "sector": "Technology"},
    {"ticker": "WDS.AX",  "name": "Woodside Energy",                "sector": "Energy"},
    {"ticker": "MPL.AX",  "name": "Medibank Private",               "sector": "Healthcare"},
    {"ticker": "NST.AX",  "name": "Northern Star Resources",        "sector": "Materials"},
    {"ticker": "TWE.AX",  "name": "Treasury Wine Estates",          "sector": "Consumer Staples"},
    {"ticker": "STO.AX",  "name": "Santos Limited",                 "sector": "Energy"},
    {"ticker": "NEM.AX",  "name": "Newmont Corporation (ASX CDI)",  "sector": "Materials"},  # replaced NCM.AX
    {"ticker": "ORA.AX",  "name": "Orora Limited",                  "sector": "Materials"},
]


def _load_tickers() -> list[dict[str, str]]:
    """Load tickers from CSV if available, else use fallback list."""
    if _TICKERS_CSV.exists():
        try:
            df = pd.read_csv(_TICKERS_CSV)
            rows = []
            for _, row in df.iterrows():
                t = str(row.get("ticker", "")).strip()
                if not t.endswith(".AX"):
                    t = t + ".AX"
                rows.append({
                    "ticker": t,
                    "name": str(row.get("name", t)),
                    "sector": str(row.get("sector", "Unknown")),
                })
            if rows:
                return rows
        except Exception:
            pass
    return _FALLBACK_TICKERS


# ---------------------------------------------------------------------------
# Fetch helpers (mirrors etf_fetcher.py)
# ---------------------------------------------------------------------------

def _safe(d: dict, *keys: str, default: float = float("nan")) -> float:
    for k in keys:
        v = d.get(k)
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
    return default


def _pct_window(close: pd.Series, days: int) -> float:
    close = close.dropna()
    if len(close) < max(2, days // 2):
        return float("nan")
    start = close.iloc[-min(days, len(close))]
    end = close.iloc[-1]
    if start == 0:
        return float("nan")
    return float((end - start) / start)


def _annualised_window(close: pd.Series, days: int) -> float:
    close = close.dropna()
    if len(close) < max(10, days // 2):
        return float("nan")
    start = close.iloc[-min(days, len(close))]
    end = close.iloc[-1]
    years = min(days, len(close)) / 252
    if start <= 0 or years <= 0:
        return float("nan")
    return float((end / start) ** (1 / years) - 1)


def _vol(close: pd.Series, days: int) -> float:
    close = close.dropna()
    if len(close) < max(10, days // 4):
        return float("nan")
    returns = close.pct_change().dropna().tail(days)
    return float(returns.std() * (252 ** 0.5))


def _max_drawdown(close: pd.Series, days: int) -> float:
    close = close.dropna().tail(days)
    if len(close) < 5:
        return float("nan")
    roll_max = close.cummax()
    drawdowns = (close - roll_max) / roll_max
    return float(drawdowns.min())


def _sharpe(ret: float, vol: float, rf: float = 0.04) -> float:
    if math.isnan(ret) or math.isnan(vol) or vol == 0:
        return float("nan")
    return (ret - rf) / vol


def _projected_return(sd: StockData) -> float:
    """Gordon Growth Model proxy: div_yield + sustainable_growth + val_reversion."""
    div = sd.dividend_yield if not math.isnan(sd.dividend_yield) else 0.0
    roe = sd.roe if not math.isnan(sd.roe) else 0.0
    payout = sd.payout_ratio if not math.isnan(sd.payout_ratio) else 0.5
    payout = max(0.0, min(1.0, payout))
    sustainable_g = roe * (1 - payout)
    val_rev = 0.0
    if not math.isnan(sd.pe_ratio) and sd.pe_ratio > 0:
        fair_pe = 15.0
        val_rev = (math.log(fair_pe / sd.pe_ratio)) / 5
    return div + sustainable_g + val_rev


def fetch_stock(row: dict[str, str], period: str = "5y") -> StockData:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("yfinance is required: pip install yfinance") from exc

    sd = StockData(
        ticker=row["ticker"],
        name=row.get("name", ""),
        sector=row.get("sector", ""),
    )
    try:
        t = yf.Ticker(sd.ticker)
        try:
            info = dict(t.info or {})
        except Exception:
            info = {}

        sd.name = info.get("longName") or info.get("shortName") or sd.name
        sd.sector = info.get("sector") or sd.sector
        sd.industry = info.get("industry") or ""

        sd.price = _safe(info, "currentPrice", "regularMarketPrice")
        sd.market_cap = _safe(info, "marketCap")
        sd.avg_volume = _safe(info, "averageVolume", "averageDailyVolume10Day")

        # Valuation
        sd.pe_ratio = _safe(info, "trailingPE", "forwardPE")
        sd.pb_ratio = _safe(info, "priceToBook")
        sd.ev_ebitda = _safe(info, "enterpriseToEbitda")
        if not math.isnan(sd.pe_ratio) and sd.pe_ratio > 0:
            sd.earnings_yield = 1.0 / sd.pe_ratio
        fcf = _safe(info, "freeCashflow")
        mc = sd.market_cap
        if not math.isnan(fcf) and not math.isnan(mc) and mc > 0:
            sd.fcf_yield = fcf / mc

        # Growth
        sd.revenue_growth = _safe(info, "revenueGrowth")
        sd.earnings_growth = _safe(info, "earningsGrowth")
        sd.forward_eps_growth = _safe(info, "earningsQuarterlyGrowth")

        # Quality
        sd.roe = _safe(info, "returnOnEquity")
        sd.roa = _safe(info, "returnOnAssets")
        sd.gross_margin = _safe(info, "grossMargins")
        sd.debt_equity = _safe(info, "debtToEquity")
        if not math.isnan(sd.debt_equity):
            sd.debt_equity = sd.debt_equity / 100.0  # yfinance returns as %, normalise

        # Income
        raw_dy = _safe(info, "dividendYield", "trailingAnnualDividendYield")
        if not math.isnan(raw_dy) and raw_dy > 1.0:
            raw_dy = raw_dy / 100.0
        sd.dividend_yield = raw_dy
        sd.payout_ratio = _safe(info, "payoutRatio")

        # Price history
        hist = t.history(period=period, auto_adjust=True)
        if hist is None or hist.empty:
            sd.error = "no price history"
            return sd
        close = hist["Close"]
        if math.isnan(sd.price):
            sd.price = float(close.dropna().iloc[-1])

        sd.return_1m = _pct_window(close, 21)
        sd.return_3m = _pct_window(close, 63)
        sd.return_6m = _pct_window(close, 126)
        sd.return_1y = _pct_window(close, 252)
        sd.return_3y = _annualised_window(close, 252 * 3)
        sd.return_5y = _annualised_window(close, 252 * 5)
        sd.volatility_1y = _vol(close, 252)
        sd.max_drawdown_1y = _max_drawdown(close, 252)
        sd.beta = _safe(info, "beta")
        sd.sharpe_1y = _sharpe(sd.return_1y, sd.volatility_1y)

        sd.projected_return = _projected_return(sd)

        # Sparkline
        weekly = close.dropna().resample("W").last().tail(160)
        sd.history_dates = [d.strftime("%Y-%m-%d") for d in weekly.index]
        sd.history_prices = [round(float(v), 4) for v in weekly.values]

        sd.fetched_at = time.time()
    except Exception as exc:
        sd.error = f"yahoo fetch failed: {exc}"
    return sd


# ---------------------------------------------------------------------------
# Cached universe fetch
# ---------------------------------------------------------------------------

_cache_lock = threading.Lock()
_cache: dict[str, StockData] = {}
_cache_timestamp: float = 0.0
_cache_refreshing: bool = False  # FIX: prevents concurrent Yahoo refresh storms
_CACHE_TTL = 4 * 3600  # 4 hours


def get_stock_data(force_refresh: bool = False, workers: int = 8) -> list[StockData]:
    """Return the full list of fetched stocks, refreshing the cache if stale.

    Only one refresh runs at a time — a second concurrent caller that finds
    the cache stale will receive the (slightly stale) cached data rather than
    triggering a duplicate Yahoo fetch.
    """
    global _cache, _cache_timestamp, _cache_refreshing

    with _cache_lock:
        age = time.time() - _cache_timestamp
        if _cache and not force_refresh and age < _CACHE_TTL:
            return list(_cache.values())
        # FIX: if another thread is already refreshing, return stale data
        if _cache_refreshing:
            return list(_cache.values())
        _cache_refreshing = True

    try:
        tickers = _load_tickers()
        results: dict[str, StockData] = {}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(fetch_stock, row): row["ticker"] for row in tickers}
            for fut in as_completed(futures):
                ticker = futures[fut]
                try:
                    sd = fut.result()
                    if sd.ok:
                        results[ticker] = sd
                except Exception:
                    pass

        with _cache_lock:
            _cache = results
            _cache_timestamp = time.time()
        return list(results.values())
    finally:
        with _cache_lock:
            _cache_refreshing = False


def cache_status() -> dict[str, Any]:
    with _cache_lock:
        age = time.time() - _cache_timestamp if _cache_timestamp else None
        return {
            "cached_count": len(_cache),
            "age_seconds": round(age, 1) if age is not None else None,
            "ttl_seconds": _CACHE_TTL,
        }
