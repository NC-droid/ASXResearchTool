"""Fetch ETF data from Yahoo Finance with in-memory + disk TTL cache.

The web app calls ``get_etf_data()`` on every screen request. Results are
cached in memory for ``CACHE_TTL_SECONDS`` and persisted to disk so a server
restart serves data instantly while a background Yahoo refresh runs.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

log = logging.getLogger(__name__)

CACHE_TTL_SECONDS = 60 * 60          # 1 hour — in-memory freshness window
DISK_CACHE_STALE_HOURS = 24          # disk cache considered too old after 24 h
DEFAULT_HISTORY_PERIOD = "5y"

_DISK_CACHE_PATH = Path(__file__).parent / "cache" / "etf_cache.json"


@dataclass
class ETFData:
    ticker: str
    name: str
    category: str
    subcategory: str
    mer: float
    distribution_freq: str
    price: float = float("nan")
    market_cap: float = float("nan")
    aum: float = float("nan")
    return_1m: float = float("nan")
    return_3m: float = float("nan")
    return_6m: float = float("nan")
    return_1y: float = float("nan")
    return_3y: float = float("nan")  # annualised CAGR over the window
    return_5y: float = float("nan")
    volatility_1y: float = float("nan")
    max_drawdown_1y: float = float("nan")
    max_drawdown_3y: float = float("nan")
    sharpe_1y: float = float("nan")
    sharpe_3y: float = float("nan")
    dividend_yield: float = float("nan")
    fetched_at: float = 0.0
    history_dates: list[str] = field(default_factory=list)
    history_prices: list[float] = field(default_factory=list)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and not math.isnan(self.price)


# ---------------------------------------------------------------------------
# Universe loading
# ---------------------------------------------------------------------------


def load_etf_universe() -> pd.DataFrame:
    csv_path = Path(__file__).parent / "data" / "asx_etfs.csv"
    df = pd.read_csv(csv_path)
    return df.drop_duplicates(subset="ticker").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Per-ticker fetch
# ---------------------------------------------------------------------------


def _safe(d: dict, key: str, default: float = float("nan")) -> float:
    v = d.get(key)
    if v is None:
        return default
    try:
        f = float(v)
        return default if math.isnan(f) or math.isinf(f) else f
    except (TypeError, ValueError):
        return default


def _pct_window(close: pd.Series, days: int) -> float:
    s = close.dropna()
    if len(s) < days + 1:
        return float("nan")
    return float(s.iloc[-1] / s.iloc[-days - 1] - 1.0)


def _annualised_window(close: pd.Series, days: int) -> float:
    """CAGR over the trailing window — the right metric for multi-year comparisons."""
    s = close.dropna()
    if len(s) < days + 1:
        return float("nan")
    total_return = float(s.iloc[-1] / s.iloc[-days - 1] - 1.0)
    years = days / 252.0
    if years <= 0 or 1 + total_return <= 0:
        return float("nan")
    return (1 + total_return) ** (1 / years) - 1


def _max_drawdown(close: pd.Series, days: int) -> float:
    s = close.dropna().tail(days)
    if len(s) < 21:
        return float("nan")
    return float((s / s.cummax() - 1).min())


def _vol(close: pd.Series, days: int = 252) -> float:
    s = close.dropna().tail(days)
    if len(s) < 20:
        return float("nan")
    return float(s.pct_change().dropna().std() * math.sqrt(252))


def _sharpe(annualised_return: float, vol: float, rf: float = 0.03) -> float:
    if math.isnan(annualised_return) or math.isnan(vol) or vol <= 0:
        return float("nan")
    return (annualised_return - rf) / vol


def _normalise_yield(y: float) -> float:
    """Yahoo sometimes returns yield as a percentage (e.g. 4.5 for 4.5%) and
    sometimes as a decimal (0.045). Anything > 1 is treated as a percent."""
    if math.isnan(y):
        return y
    return y / 100.0 if y > 1 else y


def _resolve_yield(ticker_obj, info: dict, price: float) -> float:
    """Get the trailing dividend yield for an ETF using a chain of fallbacks.

    Yahoo Finance's primary ``yield`` field is unreliable for ASX-listed ETFs —
    it's frequently None even when the fund clearly pays distributions
    (e.g. VEU.AX). This helper tries the primary field, two alternates, and
    finally falls back to summing the last 12 months of actual distributions
    from the ticker's dividend history. As a last resort we return NaN so
    the caller can decide how to display "unknown".

    Returns the yield as a decimal (0.04 means 4.00% p.a.), or NaN if
    nothing usable is available.
    """
    # 1) Primary fund yield
    y = _safe(info, "yield", float("nan"))
    if not math.isnan(y) and y > 0:
        return _normalise_yield(y)

    # 2) Trailing annual yield (alternate field name)
    y = _safe(info, "trailingAnnualDividendYield", float("nan"))
    if not math.isnan(y) and y > 0:
        return _normalise_yield(y)

    # 3) Trailing annual rate (dollars per share) divided by price
    rate = _safe(info, "trailingAnnualDividendRate", float("nan"))
    if not math.isnan(rate) and rate > 0 and not math.isnan(price) and price > 0:
        return rate / price

    # 4) Last resort: compute from actual dividend history (most reliable)
    try:
        divs = ticker_obj.dividends
    except Exception:  # noqa: BLE001
        divs = None
    if divs is not None and not divs.empty:
        try:
            cutoff = pd.Timestamp.now(tz=divs.index.tz) - pd.Timedelta(days=365)
            recent_total = float(divs[divs.index >= cutoff].sum())
            if recent_total > 0 and not math.isnan(price) and price > 0:
                return recent_total / price
        except Exception as exc:  # noqa: BLE001
            log.debug("Dividend history fallback failed: %s", exc)

    return float("nan")


def fetch_one(row: pd.Series, period: str = DEFAULT_HISTORY_PERIOD) -> ETFData:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("yfinance is required: pip install yfinance") from exc

    etf = ETFData(
        ticker=row["ticker"],
        name=row.get("name", ""),
        category=row.get("category", ""),
        subcategory=row.get("subcategory", ""),
        mer=float(row.get("mer", 0.0)) if pd.notna(row.get("mer", float("nan"))) else 0.0,
        distribution_freq=str(row.get("distribution_freq", "")),
    )
    try:
        t = yf.Ticker(etf.ticker)
        try:
            info = dict(t.info or {})
        except Exception:
            info = {}
        etf.price = _safe(info, "currentPrice", _safe(info, "regularMarketPrice"))
        etf.aum = _safe(info, "totalAssets", _safe(info, "marketCap"))
        etf.dividend_yield = _resolve_yield(t, info, etf.price)

        hist = t.history(period=period, auto_adjust=True)
        if hist is None or hist.empty:
            etf.error = "no price history"
            return etf
        close = hist["Close"]
        if math.isnan(etf.price):
            etf.price = float(close.dropna().iloc[-1])

        etf.return_1m = _pct_window(close, 21)
        etf.return_3m = _pct_window(close, 63)
        etf.return_6m = _pct_window(close, 126)
        etf.return_1y = _pct_window(close, 252)
        etf.return_3y = _annualised_window(close, 252 * 3)
        etf.return_5y = _annualised_window(close, 252 * 5)

        etf.volatility_1y = _vol(close, 252)
        etf.max_drawdown_1y = _max_drawdown(close, 252)
        etf.max_drawdown_3y = _max_drawdown(close, 252 * 3)

        etf.sharpe_1y = _sharpe(etf.return_1y, etf.volatility_1y)
        etf.sharpe_3y = _sharpe(etf.return_3y, _vol(close, 252 * 3))

        # Compress price history for the sparkline (~120 weekly points)
        weekly = close.dropna().resample("W").last().tail(160)
        etf.history_dates = [d.strftime("%Y-%m-%d") for d in weekly.index]
        etf.history_prices = [round(float(v), 4) for v in weekly.values]

        etf.fetched_at = time.time()
    except Exception as exc:  # noqa: BLE001
        etf.error = f"yahoo fetch failed: {exc}"
    return etf


# ---------------------------------------------------------------------------
# Disk cache — survives server restarts
# ---------------------------------------------------------------------------

def _nan_safe(v: Any) -> Any:
    """Convert NaN/Inf floats to None so json.dumps doesn't choke."""
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


def _etf_to_dict(etf: ETFData) -> dict:
    d = dataclasses.asdict(etf)
    return {k: _nan_safe(v) for k, v in d.items()}


def _etf_from_dict(d: dict) -> ETFData:
    # Replace JSON null → NaN for float fields
    fields = {f.name: f for f in dataclasses.fields(ETFData)}
    kwargs: dict = {}
    for name, fld in fields.items():
        val = d.get(name)
        if val is None and fld.default is dataclasses.MISSING:
            # required field — use empty string as fallback
            val = ""
        elif val is None:
            # optional float field — restore NaN
            origin = getattr(fld.type, "__origin__", None)
            if fld.default == float("nan") or (isinstance(fld.default, float) and math.isnan(fld.default)):
                val = float("nan")
        kwargs[name] = val
    return ETFData(**kwargs)


def _save_disk_cache(data: dict[str, ETFData], timestamp: float) -> None:
    """Write in-memory cache to disk. Silently ignores errors."""
    try:
        _DISK_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "timestamp": timestamp,
            "version": 1,
            "data": {k: _etf_to_dict(v) for k, v in data.items()},
        }
        tmp = _DISK_CACHE_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, allow_nan=False))
        tmp.replace(_DISK_CACHE_PATH)
        log.info("ETF disk cache written: %d entries", len(data))
    except Exception as exc:  # noqa: BLE001
        log.warning("ETF disk cache write failed: %s", exc)


def _load_disk_cache() -> tuple[dict[str, ETFData], float]:
    """Load disk cache. Returns ({}, 0.0) if missing, corrupt, or too old."""
    try:
        if not _DISK_CACHE_PATH.exists():
            return {}, 0.0
        payload = json.loads(_DISK_CACHE_PATH.read_text())
        timestamp = float(payload.get("timestamp", 0))
        age_hours = (time.time() - timestamp) / 3600
        if age_hours > DISK_CACHE_STALE_HOURS:
            log.info("ETF disk cache too old (%.1f h) — ignoring", age_hours)
            return {}, 0.0
        data = {k: _etf_from_dict(v) for k, v in payload.get("data", {}).items()}
        log.info("ETF disk cache loaded: %d entries, age %.1f h", len(data), age_hours)
        return data, timestamp
    except Exception as exc:  # noqa: BLE001
        log.warning("ETF disk cache load failed: %s", exc)
        return {}, 0.0


# ---------------------------------------------------------------------------
# In-memory cache — pre-seeded from disk on import
# ---------------------------------------------------------------------------

_cache_lock = threading.Lock()
_cache: dict[str, ETFData] = {}
_cache_timestamp: float = 0.0
_cache_refreshing: bool = False  # prevents concurrent Yahoo refresh storms

# Seed in-memory cache from disk at import time so the first request is fast
_disk_data, _disk_ts = _load_disk_cache()
if _disk_data:
    _cache.update(_disk_data)
    _cache_timestamp = _disk_ts
    log.info("ETF in-memory cache pre-seeded from disk (%d entries)", len(_cache))


def get_etf_data(force_refresh: bool = False, workers: int = 8) -> list[ETFData]:
    """Return the full list of fetched ETFs, refreshing the cache if stale.

    Only one refresh runs at a time — a second concurrent caller that finds
    the cache stale will receive the (slightly stale) cached data rather than
    triggering a duplicate Yahoo fetch.
    """
    global _cache_timestamp, _cache_refreshing

    with _cache_lock:
        is_stale = (time.time() - _cache_timestamp) > CACHE_TTL_SECONDS
        if _cache and not force_refresh and not is_stale:
            return list(_cache.values())
        if _cache_refreshing:
            log.debug("Cache refresh already in progress, returning stale data")
            return list(_cache.values())
        _cache_refreshing = True

    log.info("Refreshing ETF cache (force=%s, stale=%s)", force_refresh, is_stale)
    try:
        universe = load_etf_universe()
        results: dict[str, ETFData] = {}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(fetch_one, row): row["ticker"] for _, row in universe.iterrows()}
            for fut in as_completed(futures):
                ticker = futures[fut]
                try:
                    etf = fut.result()
                except Exception as exc:  # noqa: BLE001
                    log.debug("Unhandled fetch error for %s: %s", ticker, exc)
                    continue
                if etf.ok:
                    results[ticker] = etf
                else:
                    log.debug("Drop %s: %s", ticker, etf.error)

        now = time.time()
        with _cache_lock:
            _cache.clear()
            _cache.update(results)
            _cache_timestamp = now
        _save_disk_cache(results, now)          # persist to disk after every successful refresh
        log.info("Cache refreshed: %d / %d ETFs usable", len(results), len(universe))
        return list(results.values())
    finally:
        with _cache_lock:
            _cache_refreshing = False


def cache_status() -> dict[str, Any]:
    with _cache_lock:
        age = time.time() - _cache_timestamp if _cache_timestamp else None
    return {
        "size": len(_cache),
        "age_seconds": round(age, 1) if age is not None else None,
        "stale": age is None or age > CACHE_TTL_SECONDS,
        "ttl_seconds": CACHE_TTL_SECONDS,
        "disk_cache_exists": _DISK_CACHE_PATH.exists(),
    }
