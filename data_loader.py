"""
Production Yahoo Finance data loader with atomic parquet caching.

Contract:
    DataLoader.get_data(..., wide=True, field="Adj Close")
    -> DataFrame indexed by dates, columns=tickers.

Raw Close is retained for liquidity/trade-price checks; Adj Close is used
for total-return features and portfolio P&L. yfinance's current API documents
`auto_adjust=False` as disabling automatic OHLC adjustment, and `actions=True`
as including dividends/splits. We also request `repair=True` to let yfinance
repair documented price inconsistencies where possible.
"""
from __future__ import annotations

import argparse
import inspect
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import pandas as pd

try:
    import yfinance as yf
except ImportError as exc:
    yf = None
    _YF_IMPORT_ERROR = exc

import config
from utils import _safe_symbol, get_logger

logger = get_logger("data_loader")

VALID_INTERVALS = {
    "1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h",
    "1d", "5d", "1wk", "1mo", "3mo",
}
INTRADAY_MAX_LOOKBACK_DAYS = {
    "1m": 7, "2m": 60, "5m": 60, "15m": 60, "30m": 60,
    "60m": 730, "90m": 60, "1h": 730,
}
_TAIL_TOLERANCE = pd.Timedelta(days=4)
_DAY = pd.Timedelta(days=1)

# Messages yfinance records when Yahoo genuinely has nothing for a symbol/range.
# Anything else (rate limit, HTTP 403/429, JSON decode error, timeout) is transient.
_GENUINE_EMPTY_MARKERS = ("delisted", "no price data found", "no data found")


def _pop_yf_error(symbol: str) -> str:
    """yfinance does NOT raise on most failures: it returns an empty frame and stores the reason here."""
    try:
        import yfinance.shared as _shared
        return str(_shared._ERRORS.pop(symbol, "") or "")
    except Exception:  # noqa: BLE001
        return ""


def _is_genuine_empty(err: str) -> bool:
    low = err.lower()
    return (not low) or any(m in low for m in _GENUINE_EMPTY_MARKERS)


def _supported_download_kwargs() -> dict:
    """Only pass kwargs the installed yfinance understands (old versions reject multi_level_index)."""
    wanted = {
        "auto_adjust": False,
        "actions": True,
        "repair": config.YFINANCE_REPAIR,
        "progress": False,
        "multi_level_index": False,
        "threads": False,
    }
    try:
        params = set(inspect.signature(yf.download).parameters)
    except (TypeError, ValueError):
        return wanted
    return {k: v for k, v in wanted.items() if k in params}


def _require_yfinance():
    if yf is None:
        raise ImportError(
            "yfinance is required for live data downloads. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from _YF_IMPORT_ERROR

def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        if len(df.columns.levels) >= 2 and len(set(df.columns.get_level_values(-1))) == 1:
            df.columns = df.columns.get_level_values(0)
        elif "Close" in set(df.columns.get_level_values(0)):
            df.columns = df.columns.get_level_values(0)
        else:
            df.columns = ["_".join(str(x) for x in tup if str(x) != "nan") for tup in df.columns]
    return df

@dataclass
class DataLoader:
    data_dir: Path = dc_field(default_factory=lambda: config.DATA_DIR)
    max_retries: int = 3
    retry_backoff: float = 2.0
    max_workers: int = 4  # Yahoo rate-limits aggressively; 8 parallel workers triggers silent empty responses

    def __post_init__(self):
        self.failures: Dict[str, str] = {}
        self.data_dir = Path(self.data_dir)
        self.raw_dir = self.data_dir / config.RAW_SUBDIR
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def get_data(
        self,
        symbols: Sequence[str],
        start: str,
        end: Optional[str] = None,
        interval: str = "1d",
        force_refresh: bool = False,
        wide: bool = False,
        field: str = "Adj Close",
    ) -> Dict[str, pd.DataFrame] | pd.DataFrame:
        symbols = self._normalize_symbols(symbols)
        self._validate_interval(interval)
        results = self._run_parallel(
            lambda sym: self._get_one(sym, start, end, interval, force_refresh), symbols
        )
        data = {s: df for s, df in zip(symbols, results) if df is not None and not df.empty}
        n_fail = sum(1 for s in symbols if s in self.failures)
        logger.info(f"loaded {len(data)}/{len(symbols)} symbols ({n_fail} transient failures, "
                    f"{len(symbols) - len(data) - n_fail} with no data)")
        if symbols and not data:
            sample = next(iter(self.failures.values()), "Yahoo returned no data for any symbol")
            raise RuntimeError(
                f"No price data could be loaded for any of {len(symbols)} symbols. "
                f"This is a download problem, not missing stocks. Typical cause: Yahoo rate-limit/403 or an outdated "
                f"yfinance (pip install -U yfinance). First error: {sample}"
            )
        return self._to_wide(data, field) if wide else data

    def update(self, symbols: Sequence[str], interval: str = "1d") -> None:
        symbols = self._normalize_symbols(symbols)
        self._validate_interval(interval)
        self._run_parallel(lambda sym: self._update_one(sym, interval), symbols)

    def _run_parallel(self, fn: Callable[[str], Optional[pd.DataFrame]], symbols: Sequence[str]) -> list:
        def guarded(sym):
            try:
                return fn(sym)
            except Exception as exc:  # noqa: BLE001
                logger.error(f"{sym} failed: {exc!r}")
                self.failures[sym] = repr(exc)
                return None

        if self.max_workers <= 1 or len(symbols) <= 1:
            return [guarded(s) for s in symbols]
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(symbols))) as pool:
            return list(pool.map(guarded, symbols))

    def _get_one(self, symbol, start, end, interval, force_refresh):
        path = self._cache_path(symbol, interval)
        start_ts, target = pd.Timestamp(start), self._target(end)
        cached = None
        if path.exists() and not force_refresh:
            cached = self._read_cache(path)
            if cached.empty:
                cached = None

        if cached is not None:
            cov_start, cov_through = self._coverage(cached, path)
            if cov_start <= start_ts:
                if cov_through < target:
                    cached = self._extend_tail(symbol, cached, path, end, interval, target)
                return self._slice(cached, start, end)
            logger.info(f"{symbol} cache starts after requested start; re-downloading")

        df = self._download_with_retry(symbol, start, None, interval)
        if df is None:
            logger.warning(f"{symbol} download FAILED (transient); not caching, will retry next run")
            return self._slice(cached, start, end) if cached is not None else None
        if df.empty:
            logger.warning(f"{symbol} Yahoo has no data for this symbol/range, skipping")
            return self._slice(cached, start, end) if cached is not None else None

        self._write_cache(df, path, start_ts, self._target(None))
        return self._slice(df, start, end)

    def _update_one(self, symbol, interval):
        path = self._cache_path(symbol, interval)
        if not path.exists():
            logger.warning(f"{symbol} no cache; call get_data() first")
            return
        cached = self._read_cache(path)
        if cached.empty:
            return
        self._extend_tail(symbol, cached, path, None, interval, pd.Timestamp.today().normalize())

    def _extend_tail(self, symbol, cached, path, end, interval, target):
        cov_start, cov_through = self._coverage(cached, path)
        last = self._naive(cached.index.max())
        new = self._download_with_retry(symbol, last.strftime("%Y-%m-%d"), end, interval)
        if new is None:
            # transient failure: leave coverage untouched so the next run retries the tail
            return cached
        if new.empty:
            # genuine empty (delisted / no newer bars): only then mark the tail as covered
            if target - last > pd.Timedelta(days=config.MAX_STALENESS_DAYS):
                self._write_meta(path, cov_start, max(cov_through, target))
            return cached
        combined = pd.concat([cached, new])
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
        self._write_cache(combined, path, cov_start, max(cov_through, target))
        logger.info(f"{symbol} appended {len(combined) - len(cached)} rows")
        return combined

    def _download_with_retry(self, symbol, start, end, interval):
        """Returns a non-empty frame, an EMPTY frame (Yahoo genuinely has nothing), or None (transient failure)."""
        _require_yfinance()
        self._check_intraday_lookback(start, interval)
        kwargs = _supported_download_kwargs()
        last_err = ""
        for attempt in range(1, self.max_retries + 1):
            try:
                _pop_yf_error(symbol)
                df = yf.download(symbol, start=start, end=end, interval=interval, **kwargs)
            except TypeError:
                raise  # programming/version error: never mask it as "no data"
            except Exception as exc:  # noqa: BLE001
                last_err = repr(exc)
                logger.warning(f"{symbol} download raised (attempt {attempt}): {exc}")
                self._sleep_backoff(attempt)
                continue
            if df is None:
                df = pd.DataFrame()
            df = _flatten_columns(df)
            if not df.empty:
                idx = pd.DatetimeIndex(pd.to_datetime(df.index))
                df.index = idx.tz_localize(None) if idx.tz is not None else idx
                df = df[~df.index.duplicated(keep="last")].sort_index()
                df.index.name = "Date"
                self.failures.pop(symbol, None)
                return df
            err = _pop_yf_error(symbol)
            if _is_genuine_empty(err):
                self.failures.pop(symbol, None)
                df.index.name = "Date"
                return df
            last_err = err
            logger.warning(f"{symbol} empty response with error (attempt {attempt}): {err[:120]}")
            self._sleep_backoff(attempt)
        self.failures[symbol] = last_err or "unknown download failure"
        logger.error(f"{symbol} all {self.max_retries} download attempts failed: {last_err[:200]}")
        return None

    def _sleep_backoff(self, attempt):
        time.sleep(self.retry_backoff * attempt + random.uniform(0, 1.0))

    @staticmethod
    def _check_intraday_lookback(start, interval):
        max_days = INTRADAY_MAX_LOOKBACK_DAYS.get(interval)
        if max_days is None:
            return
        requested = (pd.Timestamp.today() - pd.Timestamp(start)).days
        if requested > max_days:
            logger.warning(
                f"interval={interval} supports ~{max_days} days; requested {requested}. "
                "Yahoo may truncate the response."
            )

    def _cache_path(self, symbol, interval):
        return self.raw_dir / interval / f"{_safe_symbol(symbol)}.parquet"

    @staticmethod
    def _meta_path(path):
        return path.with_suffix(".json")

    def _read_cache(self, path):
        df = pd.read_parquet(path)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        idx = pd.DatetimeIndex(df.index)
        df.index = idx.tz_localize(None) if idx.tz is not None else idx
        df = _flatten_columns(df)
        return df[~df.index.duplicated(keep="last")].sort_index()

    def _write_cache(self, df, path, start, through):
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp.parquet")
        df.to_parquet(tmp, index=True)
        tmp.replace(path)
        self._write_meta(path, start, through)

    def _write_meta(self, path, start, through):
        payload = {"start": pd.Timestamp(start).strftime("%Y-%m-%d"),
                   "through": pd.Timestamp(through).strftime("%Y-%m-%d")}
        self._meta_path(path).write_text(json.dumps(payload))

    def _coverage(self, df, path):
        try:
            m = json.loads(self._meta_path(path).read_text())
            return pd.Timestamp(m["start"]), pd.Timestamp(m["through"])
        except (OSError, ValueError, KeyError, TypeError):
            return self._naive(df.index.min()) - _TAIL_TOLERANCE, self._naive(df.index.max()) + _TAIL_TOLERANCE

    @staticmethod
    def _naive(ts):
        ts = pd.Timestamp(ts)
        return ts.tz_localize(None) if ts.tzinfo is not None else ts

    @staticmethod
    def _target(end):
        yesterday = pd.Timestamp.today().normalize() - _DAY
        return min(pd.Timestamp(end) - _DAY, yesterday) if end else yesterday

    @staticmethod
    def _slice(df, start, end):
        idx = pd.DatetimeIndex(df.index)
        lo = pd.Timestamp(start)
        hi = pd.Timestamp(end) if end else None
        if idx.tz is not None:
            lo = lo.tz_localize(idx.tz) if lo.tzinfo is None else lo
            hi = (hi.tz_localize(idx.tz) if hi is not None and hi.tzinfo is None else hi)
        mask = idx >= lo
        if hi is not None:
            mask &= idx < hi
        return df.loc[mask].copy()

    @staticmethod
    def _to_wide(data, field):
        series = {}
        for symbol, df in data.items():
            if field not in df.columns:
                logger.warning(f"{symbol} missing '{field}', skipped")
                continue
            series[symbol] = pd.to_numeric(df[field], errors="coerce")
        return pd.DataFrame(series).sort_index()

    @staticmethod
    def _normalize_symbols(symbols):
        if isinstance(symbols, str):
            symbols = [symbols]
        return list(dict.fromkeys(str(s).strip().upper() for s in symbols if str(s).strip()))

    @staticmethod
    def _validate_interval(interval):
        if interval not in VALID_INTERVALS:
            raise ValueError(f"Invalid interval {interval!r}: {sorted(VALID_INTERVALS)}")

def _parse_args():
    p = argparse.ArgumentParser(description="Download/cache OHLCV via yfinance.")
    p.add_argument("--symbols", nargs="+", required=True)
    p.add_argument("--start", required=True)
    p.add_argument("--end")
    p.add_argument("--interval", default="1d")
    p.add_argument("--data-dir", default=str(config.DATA_DIR))
    p.add_argument("--force-refresh", action="store_true")
    p.add_argument("--update", action="store_true")
    return p.parse_args()

def main():
    a = _parse_args()
    loader = DataLoader(data_dir=a.data_dir)
    if a.update:
        loader.update(a.symbols, interval=a.interval)
        return
    data = loader.get_data(a.symbols, a.start, a.end, a.interval, force_refresh=a.force_refresh)
    for symbol, df in data.items():
        print(f"{symbol}: {len(df)} rows, {df.index.min()} -> {df.index.max()}")

if __name__ == "__main__":
    main()
