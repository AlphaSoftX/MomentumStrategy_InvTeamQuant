"""
universe.py
===========
Three distinct queries; they answer different questions, so they stay separate.

1. get_universe(index, start, end) -> Set[str]
   RANGE + membership only: every symbol whose membership overlaps [start, end].
   "Which symbols do I need price data for?" No eligibility filtering. Load prices from
   (start - momentum lookback - liquidity lookback), not from `start`.

2. members_at(index, date) -> Set[str]
   POINT + membership only.

3. eligible_as_of(index, date) -> Set[str]
   POINT + tradeability, using only information available at `date`:
   member -> not under ASM/GSM/T2T on `date` -> passes price floor and liquidity over the trailing
   window ending at `date` -> (live dates only) resolves on yfinance today.
   This is what a backtest calls per rebalance date.

Performance: membership/surveillance intervals are flattened once per index into numpy arrays (one
vectorised comparison per query); each symbol's rolling price/liquidity stats are computed once on
first load and then looked up by binary search.

Thresholds and paths live in config.py.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

import config
from index_members import explode_intervals
from utils import _safe_symbol, get_logger

logger = get_logger("universe")

_Arrays = Tuple[np.ndarray, np.ndarray, np.ndarray]
# (bar_times int64 ns, avg traded close, non-NaN bar count, avg turnover, turnover threshold)
_PriceStats = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]


def _ts(date) -> pd.Timestamp:
    ts = pd.Timestamp(date)
    return ts.tz_localize(None) if ts.tzinfo is not None else ts


def _ns(ts: pd.Timestamp) -> int:
    return int(ts.to_datetime64().astype("datetime64[ns]").astype(np.int64))


class Universe:
    def __init__(self, data_dir: Path = config.DATA_DIR):
        self.data_dir = Path(data_dir)
        self._members: Dict[str, _Arrays] = {}
        self._surveillance: Dict[str, Optional[_Arrays]] = {}
        self._price_stats: Dict[str, Optional[_PriceStats]] = {}
        self._yf_available: Dict[str, bool] = {}
        self._warned_no_splits = False

    # ------------------------------------------------------------------ #
    # Queries
    # ------------------------------------------------------------------ #

    def get_universe(self, index_name: str, start, end) -> Set[str]:
        start, end = _ts(start), _ts(end)
        if end < start:
            raise ValueError(f"end ({end}) is before start ({start})")
        symbols, starts, ends = self._load_members(index_name)
        mask = (starts <= end.to_datetime64()) & (ends >= start.to_datetime64())
        result = set(symbols[mask].tolist())
        logger.info(f"[{index_name}] {len(result)} symbols overlap [{start.date()}, {end.date()}]")
        return result

    def members_at(self, index_name: str, date) -> Set[str]:
        d64 = _ts(date).to_datetime64()
        symbols, starts, ends = self._load_members(index_name)
        return set(symbols[(starts <= d64) & (ends >= d64)].tolist())

    def eligible_as_of(self, index_name: str, date) -> Set[str]:
        date = _ts(date)

        symbols = self.members_at(index_name, date)
        logger.info(f"[{index_name}] {len(symbols)} members as of {date.date()}")

        symbols = self._exclude_surveilled(index_name, symbols, date)
        logger.info(f"[{index_name}] {len(symbols)} after surveillance filter")

        symbols = self._apply_price_and_liquidity(symbols, date)
        logger.info(f"[{index_name}] {len(symbols)} after price/liquidity filter")

        is_live = (pd.Timestamp.today().normalize() - date).days <= config.MAX_STALENESS_DAYS
        if config.CHECK_YFINANCE_AVAILABILITY and is_live:
            symbols = self._exclude_yfinance_unavailable(symbols)
            logger.info(f"[{index_name}] {len(symbols)} after yfinance availability check")

        return symbols

    # ------------------------------------------------------------------ #
    # Membership / surveillance
    # ------------------------------------------------------------------ #

    def _load_members(self, index_name: str) -> _Arrays:
        if index_name not in self._members:
            path = self.data_dir / config.INDEX_MEMBERS_SUBDIR / f"{index_name}.parquet"
            if not path.exists():
                raise FileNotFoundError(
                    f"No membership file at {path}. Run index_members.py (or --fake) for '{index_name}' first."
                )
            self._members[index_name] = explode_intervals(
                pd.read_parquet(path, columns=["symbol", "intervals"])
            )
        return self._members[index_name]

    def _load_surveillance(self, index_name: str) -> Optional[_Arrays]:
        if index_name not in self._surveillance:
            path = self.data_dir / config.SURVEILLANCE_SUBDIR / f"{index_name}.parquet"
            if path.exists():
                df = pd.read_parquet(path)
                if "stage" in df.columns:
                    df = df[df["stage"].isin(config.SURVEILLANCE_STAGES)]
                self._surveillance[index_name] = explode_intervals(df)
            else:
                logger.warning(f"no surveillance data at {path}: ASM/GSM/T2T filter SKIPPED for '{index_name}'")
                self._surveillance[index_name] = None
        return self._surveillance[index_name]

    def _exclude_surveilled(self, index_name: str, symbols: Set[str], date: pd.Timestamp) -> Set[str]:
        arrays = self._load_surveillance(index_name)
        if arrays is None or not symbols:
            return symbols
        surv_symbols, starts, ends = arrays
        d64 = date.to_datetime64()
        flagged = set(surv_symbols[(starts <= d64) & (ends >= d64)].tolist()) & symbols
        if flagged:
            logger.info(f"[{index_name}] excluded {len(flagged)} surveilled: {sorted(flagged)[:10]}")
        return symbols - flagged

    # ------------------------------------------------------------------ #
    # Price floor & liquidity
    # ------------------------------------------------------------------ #

    def _price_path(self, symbol: str) -> Path:
        return self.data_dir / config.RAW_SUBDIR / config.PRICE_INTERVAL / f"{_safe_symbol(symbol)}.parquet"

    def _split_factor_after(self, df: pd.DataFrame):
        """
        Product of split ratios strictly after each bar. yfinance's Close is back-adjusted for later
        splits, so Close * factor is the price that actually traded then (what a price floor means).
        """
        if "Stock Splits" not in df.columns:
            if not self._warned_no_splits:
                logger.warning(
                    "cache has no 'Stock Splits' column (old auto_adjust cache?): price floor uses "
                    "back-adjusted prices and is biased. Re-download with force_refresh=True."
                )
                self._warned_no_splits = True
            return 1.0
        ratio = df["Stock Splits"].fillna(0.0).replace(0.0, 1.0)
        return ratio[::-1].cumprod()[::-1].shift(-1).fillna(1.0)

    def _load_price_stats(self, symbol: str) -> Optional[_PriceStats]:
        if symbol in self._price_stats:
            return self._price_stats[symbol]

        stats: Optional[_PriceStats] = None
        path = self._price_path(symbol)
        if path.exists():
            df = pd.read_parquet(path)
            if "Close" in df.columns and not df.empty:
                idx = pd.DatetimeIndex(pd.to_datetime(df.index))
                df.index = idx.tz_localize(None) if idx.tz is not None else idx
                df = df[~df.index.duplicated(keep="last")].sort_index()

                window = f"{config.LIQUIDITY_LOOKBACK_DAYS}D"
                close = df["Close"]
                traded = close * self._split_factor_after(df)

                def roll(s: pd.Series):
                    return s.rolling(window, closed="both")  # inclusive [date - lookback, date]

                if "Volume" in df.columns and df["Volume"].notna().any():
                    turnover = roll(close * df["Volume"]).mean()  # split-invariant: price/k * volume*k
                else:
                    turnover = pd.Series(np.nan, index=df.index)

                stats = (
                    df.index.values.astype("datetime64[ns]").astype(np.int64),
                    roll(traded).mean().to_numpy(dtype=float),
                    roll(close).count().to_numpy(dtype=float),
                    turnover.to_numpy(dtype=float),
                    float(config.MIN_AVG_DAILY_TURNOVER_INR),
                )
        self._price_stats[symbol] = stats
        return stats

    def _apply_price_and_liquidity(self, symbols: Set[str], date: pd.Timestamp) -> Set[str]:
        date_ns = _ns(date)
        max_stale_ns = int(pd.Timedelta(days=config.MAX_STALENESS_DAYS).value)

        missing = [s for s in symbols if s not in self._price_stats]
        if len(missing) > 8:
            with ThreadPoolExecutor(max_workers=8) as pool:
                list(pool.map(self._load_price_stats, missing))

        eligible: Set[str] = set()
        no_data = thin = price = liquidity = 0
        for symbol in symbols:
            stats = self._load_price_stats(symbol)
            if stats is None:
                no_data += 1
                continue
            times, avg_price, n_obs, turnover, threshold = stats
            i = int(np.searchsorted(times, date_ns, side="right")) - 1  # last bar <= date
            if i < 0 or (date_ns - times[i]) > max_stale_ns or n_obs[i] < config.MIN_LOOKBACK_TRADING_DAYS:
                thin += 1
            elif not (avg_price[i] >= config.PRICE_FLOOR_INR):  # False for NaN too
                price += 1
            elif not (turnover[i] >= threshold):
                liquidity += 1
            else:
                eligible.add(symbol)

        logger.info(
            f"price/liquidity: {len(eligible)} passed, {no_data} no cached prices, {thin} thin/stale, "
            f"{price} below price floor, {liquidity} below liquidity"
        )
        return eligible

    # ------------------------------------------------------------------ #
    # Live yfinance availability
    # ------------------------------------------------------------------ #

    def _exclude_yfinance_unavailable(self, symbols: Set[str]) -> Set[str]:
        todo = sorted(s for s in symbols if s not in self._yf_available)
        for i in range(0, len(todo), config.YFINANCE_PROBE_BATCH):
            chunk = todo[i : i + config.YFINANCE_PROBE_BATCH]
            try:
                ok = self._probe_yfinance(chunk)
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"yfinance probe failed ({exc}); not excluding this batch")
                continue
            if not ok:  # empty answer = network/rate-limit as likely as delisting: fail open, retry later
                logger.warning(f"yfinance probe returned nothing for {len(chunk)} symbols; not excluding")
                continue
            self._yf_available.update({s: s in ok for s in chunk})

        unavailable = {s for s in symbols if self._yf_available.get(s) is False}
        if unavailable:
            logger.info(f"excluded {len(unavailable)} not on yfinance: {sorted(unavailable)[:10]}")
        return symbols - unavailable

    @staticmethod
    def _probe_yfinance(symbols) -> Set[str]:
        """One batched call; returns the symbols with any Close data. Isolated so tests can patch it."""
        df = yf.download(
            list(symbols),
            period=config.YFINANCE_AVAILABILITY_PROBE_PERIOD,
            progress=False,
            multi_level_index=True,
            threads=True,
        )
        if df is None or df.empty:
            return set()
        close = df["Close"]
        if isinstance(close, pd.Series):
            close = close.to_frame(list(symbols)[0])
        return set(close.columns[close.notna().any()])