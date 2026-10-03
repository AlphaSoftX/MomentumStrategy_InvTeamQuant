"""
Point-in-time universe/eligibility queries.

Three different questions are deliberately kept separate:
  get_universe(index, start, end) -> all symbols whose membership overlaps a range
  members_at(index, date)         -> members on that date
  eligible_as_of(index, date)     -> members that pass only-then-known filters

Eligibility is the only set used by the production backtest.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError as exc:
    yf = None
    _YF_IMPORT_ERROR = exc

import config
from index_members import explode_intervals
from utils import _safe_symbol, get_logger

logger = get_logger("universe")

_PriceStats = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]

def _ts(date):
    ts = pd.Timestamp(date)
    return ts.tz_localize(None) if ts.tzinfo is not None else ts

def _ns(ts):
    return int(pd.Timestamp(ts).to_datetime64().astype("datetime64[ns]").astype(np.int64))

class Universe:
    def __init__(self, data_dir: Path = config.DATA_DIR):
        self.data_dir = Path(data_dir)
        self._members: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._surveillance: Dict[str, Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
        self._price_stats: Dict[str, Optional[_PriceStats]] = {}
        self._yf_available: Dict[str, bool] = {}

    def get_universe(self, index_name, start, end):
        start, end = _ts(start), _ts(end)
        if end < start:
            raise ValueError(f"end ({end}) is before start ({start})")
        symbols, starts, ends = self._load_members(index_name)
        mask = (starts <= end.to_datetime64()) & (ends >= start.to_datetime64())
        return set(symbols[mask].tolist())

    def members_at(self, index_name, date):
        d64 = _ts(date).to_datetime64()
        symbols, starts, ends = self._load_members(index_name)
        return set(symbols[(starts <= d64) & (ends >= d64)].tolist())

    def eligible_as_of(self, index_name, date):
        date = _ts(date)
        symbols = self.members_at(index_name, date)
        symbols = self._exclude_surveilled(index_name, symbols, date)
        symbols = self._apply_price_and_liquidity(symbols, date)

        is_live = (pd.Timestamp.today().normalize() - date).days <= config.MAX_STALENESS_DAYS
        if config.CHECK_YFINANCE_AVAILABILITY and is_live:
            symbols = self._exclude_yfinance_unavailable(symbols)
        return symbols

    def eligibility_panel(self, index_name, dates, columns=None):
        dates = pd.DatetimeIndex(pd.to_datetime(dates))
        cols = set(columns) if columns is not None else None
        rows = []
        universe_dates = []
        for d in dates:
            eligible = self.eligible_as_of(index_name, d)
            if cols is not None:
                eligible &= cols
            rows.append(eligible)
            universe_dates.append(d)
        if not rows:
            return pd.DataFrame(index=dates, columns=sorted(cols or []), dtype=bool)
        all_syms = sorted(set().union(*rows))
        mask = pd.DataFrame(False, index=dates, columns=all_syms)
        for d, syms in zip(universe_dates, rows):
            if syms:
                mask.loc[d, sorted(syms)] = True
        return mask

    def _load_members(self, index_name):
        if index_name not in self._members:
            path = self.data_dir / config.INDEX_MEMBERS_SUBDIR / f"{index_name}.parquet"
            if not path.exists():
                raise FileNotFoundError(
                    f"No membership file at {path}. Run index_members.py first."
                )
            self._members[index_name] = explode_intervals(pd.read_parquet(path, columns=["symbol", "intervals"]))
        return self._members[index_name]

    def _load_surveillance(self, index_name):
        if index_name not in self._surveillance:
            path = self.data_dir / config.SURVEILLANCE_SUBDIR / f"{index_name}.parquet"
            if path.exists():
                df = pd.read_parquet(path)
                if "stage" in df.columns:
                    df = df[df["stage"].isin(config.SURVEILLANCE_STAGES)]
                self._surveillance[index_name] = explode_intervals(df)
            else:
                logger.warning(f"no surveillance data at {path}; ASM/GSM/T2T filter skipped")
                self._surveillance[index_name] = None
        return self._surveillance[index_name]

    def _exclude_surveilled(self, index_name, symbols, date):
        arrays = self._load_surveillance(index_name)
        if arrays is None or not symbols:
            return symbols
        surv_symbols, starts, ends = arrays
        d64 = date.to_datetime64()
        flagged = set(surv_symbols[(starts <= d64) & (ends >= d64)].tolist()) & symbols
        return symbols - flagged

    def _price_path(self, symbol):
        return self.data_dir / config.RAW_SUBDIR / config.PRICE_INTERVAL / f"{_safe_symbol(symbol)}.parquet"

    def _load_price_stats(self, symbol):
        if symbol in self._price_stats:
            return self._price_stats[symbol]
        path = self._price_path(symbol)
        if not path.exists():
            self._price_stats[symbol] = None
            return None
        df = pd.read_parquet(path)
        if df.empty or "Close" not in df.columns:
            self._price_stats[symbol] = None
            return None
        idx = pd.DatetimeIndex(pd.to_datetime(df.index))
        df.index = idx.tz_localize(None) if idx.tz is not None else idx
        df = df[~df.index.duplicated(keep="last")].sort_index()
        close = pd.to_numeric(df["Close"], errors="coerce")
        window = f"{config.LIQUIDITY_LOOKBACK_DAYS}D"
        roll = lambda s: s.rolling(window, closed="both")
        avg_price = roll(close).mean()
        n_obs = roll(close).count()
        if "Volume" in df.columns and df["Volume"].notna().any():
            turnover = roll(close * pd.to_numeric(df["Volume"], errors="coerce")).mean()
        else:
            turnover = pd.Series(np.nan, index=df.index)
        stats = (
            df.index.values.astype("datetime64[ns]").astype(np.int64),
            avg_price.to_numpy(dtype=float),
            n_obs.to_numpy(dtype=float),
            turnover.to_numpy(dtype=float),
            float(config.MIN_AVG_DAILY_TURNOVER_INR),
        )
        self._price_stats[symbol] = stats
        return stats

    def _apply_price_and_liquidity(self, symbols, date):
        date_ns = _ns(date)
        max_stale_ns = int(pd.Timedelta(days=config.MAX_STALENESS_DAYS).value)
        missing = [s for s in symbols if s not in self._price_stats]
        if len(missing) > 8:
            with ThreadPoolExecutor(max_workers=8) as pool:
                list(pool.map(self._load_price_stats, missing))

        eligible = set()
        for symbol in symbols:
            stats = self._load_price_stats(symbol)
            if stats is None:
                continue
            times, avg_price, n_obs, turnover, threshold = stats
            i = int(np.searchsorted(times, date_ns, side="right")) - 1
            if i < 0 or (date_ns - times[i]) > max_stale_ns:
                continue
            if n_obs[i] < config.MIN_LOOKBACK_TRADING_DAYS:
                continue
            if not (avg_price[i] >= config.PRICE_FLOOR_INR):
                continue
            if not (turnover[i] >= threshold):
                continue
            eligible.add(symbol)
        return eligible

    def _exclude_yfinance_unavailable(self, symbols):
        if yf is None:
            raise ImportError(
                "yfinance is required for live availability probes"
            ) from _YF_IMPORT_ERROR
        todo = sorted(s for s in symbols if s not in self._yf_available)
        for i in range(0, len(todo), config.YFINANCE_PROBE_BATCH):
            chunk = todo[i:i + config.YFINANCE_PROBE_BATCH]
            try:
                ok = self._probe_yfinance(chunk)
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"yfinance availability probe failed ({exc}); fail-open for batch")
                continue
            if not ok:
                logger.warning(f"yfinance availability probe returned no data; fail-open for {len(chunk)} names")
                continue
            self._yf_available.update({s: s in ok for s in chunk})
        return {s for s in symbols if self._yf_available.get(s, True)}

    @staticmethod
    def _probe_yfinance(symbols):
        if not symbols:
            return set()
        df = yf.download(
            list(symbols),
            period=config.YFINANCE_AVAILABILITY_PROBE_PERIOD,
            progress=False,
            threads=False,
            repair=config.YFINANCE_REPAIR,
            auto_adjust=False,
        )
        if df is None or df.empty:
            return set()
        close = df["Close"]
        if isinstance(close, pd.Series):
            close = close.to_frame(list(symbols)[0])
        return set(close.columns[close.notna().any()])
