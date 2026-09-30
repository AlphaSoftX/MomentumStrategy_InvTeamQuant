"""
data_loader.py
==============
Downloads, caches and loads OHLCV data via yfinance.

Cache layout: <data_dir>/raw/<interval>/<symbol>.parquet (+ <symbol>.json coverage sidecar).

Prices are stored with auto_adjust=False plus actions:
    Close       split-adjusted only  (turnover = Close * Volume is correct; universe.py rebuilds the
                                      traded price from "Stock Splits" for the price floor)
    Adj Close   split + dividend adjusted (use this for returns / momentum)

The sidecar records the requested [start, through] range, so IPOs after `start`, delistings and
weekends/holidays don't trigger a re-download on every call.

Library:
    loader = DataLoader(data_dir="data")
    prices = loader.get_data(["RELIANCE.NS"], start="2015-01-01", end="2024-12-31")
    closes = loader.get_data(symbols, start="2015-01-01", wide=True)   # Adj Close matrix

Script:
    python data_loader.py --symbols RELIANCE.NS TCS.NS --start 2015-01-01 --interval 1d
    python data_loader.py --symbols RELIANCE.NS TCS.NS --start 2015-01-01 --update
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import yfinance as yf

import config
from utils import _safe_symbol, get_logger

logger = get_logger("data_loader")

VALID_INTERVALS = {
    "1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h",
    "1d", "5d", "1wk", "1mo", "3mo",
}

INTRADAY_MAX_LOOKBACK_DAYS = {
    "1m": 7, "2m": 60, "5m": 60, "15m": 60, "30m": 60, "60m": 730, "90m": 60, "1h": 730,
}

# Slack on both ends when inferring coverage from a cache with no sidecar (legacy files): weekend + holiday.
_TAIL_TOLERANCE = pd.Timedelta(days=4)
_DAY = pd.Timedelta(days=1)


@dataclass
class DataLoader:
    """
    Parameters
    ----------
    data_dir : root directory of the cache.
    max_retries, retry_backoff : retries on download *exceptions* (linear backoff, seconds).
    max_workers : symbols processed concurrently; keep moderate or Yahoo rate-limits (HTTP 429).
    """

    data_dir: Path = dc_field(default_factory=lambda: config.DATA_DIR)
    max_retries: int = 3
    retry_backoff: float = 2.0
    max_workers: int = 8

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        self.raw_dir = self.data_dir / config.RAW_SUBDIR
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

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
        """
        Returns {symbol: OHLCV frame} for [start, end) (end exclusive, like yfinance), or with
        wide=True one frame of `field` (columns = symbols). Symbols with no data are omitted.
        """
        symbols = self._normalize_symbols(symbols)
        self._validate_interval(interval)

        results = self._run_parallel(
            lambda sym: self._get_one(sym, start, end, interval, force_refresh), symbols
        )
        data = {s: df for s, df in zip(symbols, results) if df is not None and not df.empty}
        return self._to_wide(data, field) if wide else data

    def update(self, symbols: Sequence[str], interval: str = "1d") -> None:
        """Appends the missing tail (re-fetching the last cached bar) for already-cached symbols."""
        symbols = self._normalize_symbols(symbols)
        self._validate_interval(interval)
        self._run_parallel(lambda sym: self._update_one(sym, interval), symbols)

    # ------------------------------------------------------------------ #
    # Per-symbol work
    # ------------------------------------------------------------------ #

    def _run_parallel(self, fn: Callable[[str], Optional[pd.DataFrame]], symbols: Sequence[str]) -> list:
        """fn(symbol) for every symbol, in input order; one failing symbol never aborts the rest."""

        def guarded(sym: str):
            try:
                return fn(sym)
            except Exception as exc:  # noqa: BLE001
                logger.error(f"{sym} failed: {exc!r}")
                return None

        if self.max_workers <= 1 or len(symbols) <= 1:
            return [guarded(s) for s in symbols]
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(symbols))) as pool:
            return list(pool.map(guarded, symbols))

    def _get_one(
        self, symbol: str, start: str, end: Optional[str], interval: str, force_refresh: bool
    ) -> Optional[pd.DataFrame]:
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
                else:
                    logger.debug(f"{symbol} loaded from cache ({len(cached)} rows)")
                return self._slice(cached, start, end)
            logger.info(f"{symbol} cache starts after requested start, re-downloading")

        # Full history from `start` to today: one download serves any later `end`.
        df = self._download_with_retry(symbol, start, None, interval)
        if df is None or df.empty:
            logger.warning(f"{symbol} no data returned, skipping")
            return self._slice(cached, start, end) if cached is not None else None

        self._write_cache(df, path, start_ts, self._target(None))
        return self._slice(df, start, end)

    def _update_one(self, symbol: str, interval: str) -> None:
        path = self._cache_path(symbol, interval)
        if not path.exists():
            logger.warning(f"{symbol} no cache, skipping. Call get_data() first to bootstrap history.")
            return
        cached = self._read_cache(path)
        if cached.empty:
            return
        self._extend_tail(symbol, cached, path, None, interval, pd.Timestamp.today().normalize())

    def _extend_tail(
        self, symbol: str, cached: pd.DataFrame, path: Path,
        end: Optional[str], interval: str, target: pd.Timestamp,
    ) -> pd.DataFrame:
        cov_start, cov_through = self._coverage(cached, path)
        last = self._naive(cached.index.max())

        # Start AT the last cached bar (not the day after) so a partial bar cached intraday is overwritten.
        new = self._download_with_retry(symbol, last.strftime("%Y-%m-%d"), end, interval)
        if new is None or new.empty:
            # Long-stale with nothing new = delisted/suspended: record coverage so we stop asking.
            if target - last > pd.Timedelta(days=config.MAX_STALENESS_DAYS):
                self._write_meta(path, cov_start, max(cov_through, target))
            return cached

        combined = pd.concat([cached, new])
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
        self._write_cache(combined, path, cov_start, max(cov_through, target))
        logger.info(f"{symbol} appended {len(combined) - len(cached)} rows")
        return combined

    # ------------------------------------------------------------------ #
    # Download
    # ------------------------------------------------------------------ #

    def _download_with_retry(
        self, symbol: str, start: str, end: Optional[str], interval: str
    ) -> Optional[pd.DataFrame]:
        """DataFrame (possibly empty = no data exists) or None after repeated exceptions."""
        self._check_intraday_lookback(start, interval)
        for attempt in range(1, self.max_retries + 1):
            try:
                logger.debug(f"[{symbol}] download {interval} {start} -> {end} (attempt {attempt})")
                df = yf.download(
                    symbol,
                    start=start,
                    end=end,
                    interval=interval,
                    auto_adjust=False,
                    actions=True,
                    progress=False,
                    multi_level_index=False,
                )
                if df is None:
                    df = pd.DataFrame()
                df.index.name = "Date"
                return df
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"{symbol} download failed (attempt {attempt}): {exc}")
                if attempt < self.max_retries:
                    time.sleep(self.retry_backoff * attempt)
        logger.error(f"{symbol} all {self.max_retries} download attempts failed")
        return None

    @staticmethod
    def _check_intraday_lookback(start: str, interval: str) -> None:
        max_days = INTRADAY_MAX_LOOKBACK_DAYS.get(interval)
        if max_days is None:
            return
        requested = (pd.Timestamp.today() - pd.Timestamp(start)).days
        if requested > max_days:
            logger.warning(
                f"interval={interval} only supports ~{max_days} days via yfinance; "
                f"requested {requested}. Data will be truncated."
            )

    # ------------------------------------------------------------------ #
    # Cache + coverage sidecar
    # ------------------------------------------------------------------ #

    def _cache_path(self, symbol: str, interval: str) -> Path:
        return self.raw_dir / interval / f"{_safe_symbol(symbol)}.parquet"

    @staticmethod
    def _meta_path(path: Path) -> Path:
        return path.with_suffix(".json")

    def _read_cache(self, path: Path) -> pd.DataFrame:
        df = pd.read_parquet(path)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        return df

    def _write_cache(self, df: pd.DataFrame, path: Path, start: pd.Timestamp, through: pd.Timestamp) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        df.to_parquet(tmp)
        tmp.replace(path)  # atomic: a crash never leaves a half-written parquet
        self._write_meta(path, start, through)
        logger.debug(f"saved {len(df)} rows -> {path}")

    def _write_meta(self, path: Path, start: pd.Timestamp, through: pd.Timestamp) -> None:
        payload = {"start": start.strftime("%Y-%m-%d"), "through": through.strftime("%Y-%m-%d")}
        self._meta_path(path).write_text(json.dumps(payload))

    def _coverage(self, df: pd.DataFrame, path: Path) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """(earliest requested start, last date requested) from the sidecar, else inferred from data."""
        try:
            m = json.loads(self._meta_path(path).read_text())
            return pd.Timestamp(m["start"]), pd.Timestamp(m["through"])
        except (OSError, ValueError, KeyError):
            return self._naive(df.index.min()) - _TAIL_TOLERANCE, self._naive(df.index.max()) + _TAIL_TOLERANCE

    @staticmethod
    def _naive(ts) -> pd.Timestamp:
        ts = pd.Timestamp(ts)
        return ts.tz_localize(None) if ts.tzinfo is not None else ts

    @staticmethod
    def _target(end: Optional[str]) -> pd.Timestamp:
        """Last bar date we expect to exist: day before `end` (exclusive), capped at yesterday."""
        yesterday = pd.Timestamp.today().normalize() - _DAY
        return min(pd.Timestamp(end) - _DAY, yesterday) if end else yesterday

    @staticmethod
    def _slice(df: pd.DataFrame, start: str, end: Optional[str]) -> pd.DataFrame:
        """[start, end): matches what a fresh yfinance download returns."""
        idx = pd.DatetimeIndex(df.index)
        tz = idx.tz
        lo = pd.Timestamp(start)
        hi = pd.Timestamp(end) if end else None
        if tz is not None:
            lo = lo.tz_localize(tz)
            hi = hi.tz_localize(tz) if hi is not None else None
        mask = idx >= lo
        if hi is not None:
            mask &= idx < hi
        return df[mask].copy()

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _to_wide(data: Dict[str, pd.DataFrame], field: str) -> pd.DataFrame:
        series = {}
        for symbol, df in data.items():
            if field not in df.columns:
                logger.warning(f"{symbol} missing '{field}', skipped in wide output")
                continue
            series[symbol] = df[field]
        return pd.DataFrame(series).sort_index()

    @staticmethod
    def _normalize_symbols(symbols: Sequence[str]) -> List[str]:
        if isinstance(symbols, str):
            symbols = [symbols]
        return list(dict.fromkeys(s.strip().upper() for s in symbols if s.strip()))

    @staticmethod
    def _validate_interval(interval: str) -> None:
        if interval not in VALID_INTERVALS:
            raise ValueError(f"Invalid interval '{interval}'. Must be one of: {sorted(VALID_INTERVALS)}")


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download and cache market data via yfinance.")
    p.add_argument("--symbols", nargs="+", required=True, help="e.g. RELIANCE.NS TCS.NS")
    p.add_argument("--start", required=True, help="YYYY-MM-DD")
    p.add_argument("--end", default=None, help="YYYY-MM-DD, exclusive (default: today)")
    p.add_argument("--interval", default="1d")
    p.add_argument("--data-dir", default=str(config.DATA_DIR))
    p.add_argument("--force-refresh", action="store_true", help="ignore cache and re-download")
    p.add_argument("--update", action="store_true", help="append the missing tail to cached symbols")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    loader = DataLoader(data_dir=args.data_dir)

    if args.update:
        loader.update(args.symbols, interval=args.interval)
        return

    data = loader.get_data(
        args.symbols, args.start, args.end, args.interval, force_refresh=args.force_refresh
    )
    for symbol, df in data.items():
        print(f"{symbol}: {len(df)} rows, {df.index.min()} -> {df.index.max()}")


if __name__ == "__main__":
    main()