"""
index_members.py
================
Point-in-time NSE index membership from free sources, saved to
    <data_dir>/index_members/<index>.parquet
    symbol    : yfinance form, e.g. "RELIANCE.NS"
    intervals : [[start, end], ...]   end = NaT while still a member

Why: free feeds only give today's constituents; backtesting with them silently drops every stock that
was removed, delisted or went bust, which is exactly what a momentum book ends up holding.

Sources
    1. NSE IndexInclExcl.xls            exact IN/OUT events, ~1998 - Jul 2020
    2. niftyindices monthly zips        full snapshots, Apr 2013 - Mar 2022
    3. Wayback captures of the live CSV full snapshots, 2022 - today (monthly, approximate)
    4. Live constituent CSV             anchor for "member today"; also appended as the final snapshot
    5. manual/<index>_manual_events.csv symbol,date,action hand patches (e.g. from press-release PDFs)
Snapshots (2-4) are merged into ONE timeline and diffed into approximate events, so gaps between sources
are still diffed. Exact events (1, 5) override approximate ones describing the same transition.

The fetchers are unverified against the live sites (they block bare requests and change formats):
run `--diagnostics` first and expect small fixes.

    python index_members.py --index nifty500 --data-dir data --diagnostics
    python index_members.py --index nifty500 --data-dir data
    python index_members.py --fake        # placeholder file from a seed CSV, NOT bias-free
"""

from __future__ import annotations

import argparse
import io
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import config
from utils import get_logger

logger = get_logger("index_members")

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

NSE_HOME = "https://www.nseindia.com"
INCL_EXCL_XLS_URL = "https://archives.nseindia.com/content/indices/IndexInclExcl.xls"
INCL_EXCL_SHEET_CANDIDATES = ["Nifty 500", "NIFTY 500", "S&P CNX 500", "CNX 500"]

CURRENT_CSV_URLS = {
    "nifty500": [
        "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv",
        "https://niftyindices.com/IndexConstituent/ind_nifty500list.csv",
    ],
}
# Older Wayback captures may sit under the previous host as well.
WAYBACK_TARGETS = {
    "nifty500": CURRENT_CSV_URLS["nifty500"]
    + ["https://archives.nseindia.com/content/indices/ind_nifty500list.csv"],
}

MONTHLY_ZIP_URL_TMPL = (
    "https://niftyindices.com/Indices_-_Market_Capitalisation_and_Weightage/indices_data{month_abbr}{year}.zip"
)
MONTHLY_ZIP_COVERAGE = (pd.Timestamp("2013-04-01"), pd.Timestamp("2022-03-31"))
WAYBACK_CDX_URL = "https://web.archive.org/cdx/search/cdx"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/market-data/live-equity-market",
}

EARLIEST_COVERAGE = pd.Timestamp("2000-01-01")  # floor for members with no discoverable events
SNAPSHOT_SIZE_TOLERANCE = 0.10                  # drop snapshots > 10% away from the median size
SHADOW_WINDOW_DAYS = 45                         # approx event this close to an exact one is a duplicate

_EVENT_COLS = ["symbol", "date", "action", "precision", "source"]


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------

def _http_session(headers: Optional[dict] = None) -> requests.Session:
    session = requests.Session()
    if headers:
        session.headers.update(headers)
    retry = Retry(total=3, backoff_factor=1.0, status_forcelist=(429, 500, 502, 503, 504), allowed_methods=("GET",))
    session.mount("https://", HTTPAdapter(max_retries=retry, pool_maxsize=16))
    return session


def _nse_session(timeout: float = 10.0) -> requests.Session:
    """Session with browser headers, warmed up on the homepage for cookies (NSE 403s bare requests)."""
    session = _http_session(HEADERS)
    try:
        session.get(NSE_HOME, timeout=timeout)
    except requests.RequestException as exc:
        logger.warning(f"NSE warm-up failed ({exc}); continuing anyway")
    return session


def _map_threads(fn: Callable, items, workers: int = 6) -> list:
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(fn, items))


# --------------------------------------------------------------------------
# Parsing helpers
# --------------------------------------------------------------------------

def _find_col(columns, hints: List[str]) -> Optional[str]:
    lower = {str(c).lower(): str(c) for c in columns}
    for hint in hints:
        for lc, orig in lower.items():
            if hint in lc:
                return orig
    return None


def _to_yf_symbol(raw: pd.Series) -> pd.Series:
    """'RELIANCE' -> 'RELIANCE.NS' (existing suffix kept); junk/dummy rows become NaN."""
    s = raw.astype(str).str.strip().str.upper()
    s = s.where(s.str.contains(".", regex=False), s + ".NS")
    return s.mask(s.str.startswith(("NAN.", "DUMMY")) | (s == ".NS"))


def _symbols_from_frame(df: pd.DataFrame) -> Set[str]:
    col = _find_col(df.columns, ["symbol"])
    return set(_to_yf_symbol(df[col]).dropna()) if col else set()


def _empty_events() -> pd.DataFrame:
    return pd.DataFrame(columns=_EVENT_COLS)


def _events_frame(symbols, dates, actions, precision: str, source: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": pd.Series(list(symbols), dtype=object),
            "date": pd.to_datetime(pd.Series(list(dates))),
            "action": pd.Series(list(actions), dtype=object),
            "precision": precision,
            "source": source,
        }
    )


# --------------------------------------------------------------------------
# Source 1: NSE IndexInclExcl.xls
# --------------------------------------------------------------------------

def fetch_incl_excl_events(session: requests.Session, index_key: str = "nifty500") -> pd.DataFrame:
    """Exact IN/OUT events. Returns an empty frame (never raises) on any failure. Column names are matched fuzzily."""
    try:
        resp = session.get(INCL_EXCL_XLS_URL, timeout=20)
        resp.raise_for_status()
        book = pd.ExcelFile(io.BytesIO(resp.content), engine="xlrd")  # legacy .xls needs `pip install xlrd`
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"IndexInclExcl.xls unavailable: {exc}")
        return _empty_events()

    sheet = next((s for s in INCL_EXCL_SHEET_CANDIDATES if s in book.sheet_names), None)
    if sheet is None:
        logger.warning(f"none of {INCL_EXCL_SHEET_CANDIDATES} in workbook; sheets: {book.sheet_names}")
        return _empty_events()

    raw = book.parse(sheet)
    raw.columns = [str(c).strip() for c in raw.columns]
    logger.info(f"IndexInclExcl.xls['{sheet}'] columns: {list(raw.columns)}")

    symbol_col = _find_col(raw.columns, ["symbol"])
    action_col = _find_col(raw.columns, ["action", "inclusion", "exclusion", "type"])
    date_col = _find_col(raw.columns, ["effective date", "date"])
    if not (symbol_col and date_col):
        logger.warning(f"cannot identify symbol/date columns (got {symbol_col}, {date_col}); adjust _find_col hints")
        return _empty_events()

    symbols = _to_yf_symbol(raw[symbol_col])
    dates = pd.to_datetime(raw[date_col], errors="coerce")
    if action_col:
        is_out = raw[action_col].astype(str).str.lower().str.contains("excl|remov|out", regex=True, na=False)
    else:
        is_out = pd.Series(False, index=raw.index)

    valid = symbols.notna() & dates.notna()
    events = _events_frame(
        symbols[valid], dates[valid], np.where(is_out[valid], "OUT", "IN"), "exact", "nse_incl_excl_xls"
    )
    logger.info(f"IndexInclExcl.xls: {len(events)} events")
    return events


# --------------------------------------------------------------------------
# Sources 2-4: snapshots
# --------------------------------------------------------------------------

def _symbols_from_zip(zip_bytes: bytes) -> Set[str]:
    """Inspect zf.namelist() on a real download and adjust the member match if this misses."""
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = zf.namelist()
        candidates = [n for n in names if "500" in n.lower() and n.lower().endswith(".csv")]
        if not candidates:
            logger.warning(f"no Nifty 500 csv in zip; contents: {names}")
            return set()
        with zf.open(candidates[0]) as f:
            return _symbols_from_frame(pd.read_csv(f))


def fetch_monthly_snapshots(
    session: requests.Session,
    start: pd.Timestamp = MONTHLY_ZIP_COVERAGE[0],
    end: pd.Timestamp = MONTHLY_ZIP_COVERAGE[1],
) -> Dict[pd.Timestamp, Set[str]]:
    """
    {month_end: symbols}. Keyed at month END: if the file is an end-of-month list, month-start keys
    would put additions up to a month before they were true (look-ahead). Failures are per-month.
    """

    def one(month_start: pd.Timestamp):
        label = f"{month_start:%b%Y}"
        url = MONTHLY_ZIP_URL_TMPL.format(month_abbr=month_start.strftime("%b"), year=month_start.year)
        try:
            resp = session.get(url, timeout=20)
            resp.raise_for_status()
            symbols = _symbols_from_zip(resp.content)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"monthly snapshot {label} failed: {exc}")
            return None
        if not symbols:
            return None
        logger.info(f"monthly snapshot {label}: {len(symbols)} symbols")
        return month_start + pd.offsets.MonthEnd(0), symbols

    months = pd.date_range(start, end, freq="MS")
    return dict(r for r in _map_threads(one, months) if r)


def fetch_wayback_snapshots(
    index_key: str = "nifty500",
    start: pd.Timestamp = pd.Timestamp("2022-01-01"),
    end: Optional[pd.Timestamp] = None,
) -> Dict[pd.Timestamp, Set[str]]:
    """{capture_date: symbols}, at most one capture per month per URL (constituents change twice a year)."""
    end = end or pd.Timestamp.today()
    session = _http_session()

    captures: Dict[str, str] = {}  # wayback timestamp -> original url
    for target in WAYBACK_TARGETS[index_key]:
        try:
            resp = session.get(
                WAYBACK_CDX_URL,
                params={
                    "url": target, "output": "json", "filter": "statuscode:200",
                    "from": start.strftime("%Y%m%d"), "to": end.strftime("%Y%m%d"),
                    "collapse": "timestamp:6",
                },
                timeout=30,
            )
            resp.raise_for_status()
            rows = resp.json()
        except (requests.RequestException, ValueError) as exc:
            logger.warning(f"Wayback CDX lookup failed for {target}: {exc}")
            continue
        if len(rows) < 2:
            continue
        ts_col = rows[0].index("timestamp")
        for row in rows[1:]:
            captures.setdefault(row[ts_col], target)

    if not captures:
        logger.info("Wayback CDX returned no usable captures")
        return {}

    def one(item: Tuple[str, str]):
        ts, target = item
        try:
            resp = session.get(f"https://web.archive.org/web/{ts}if_/{target}", timeout=30)
            resp.raise_for_status()
            symbols = _symbols_from_frame(pd.read_csv(io.StringIO(resp.text)))
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"wayback capture {ts} failed: {exc}")
            return None
        if not symbols:
            return None
        logger.info(f"wayback snapshot {ts[:8]}: {len(symbols)} symbols")
        return pd.Timestamp(ts[:8]), symbols

    return dict(r for r in _map_threads(one, sorted(captures.items()), workers=4) if r)


def fetch_current_constituents(session: requests.Session, index_key: str = "nifty500") -> Set[str]:
    for url in CURRENT_CSV_URLS[index_key]:
        try:
            resp = session.get(url, timeout=15)
            resp.raise_for_status()
            symbols = _symbols_from_frame(pd.read_csv(io.StringIO(resp.text)))
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"current constituents fetch failed for {url}: {exc}")
            continue
        if symbols:
            logger.info(f"current constituents ({url}): {len(symbols)} symbols")
            return symbols
    logger.error("could not fetch current constituents from any known URL")
    return set()


def _drop_outlier_snapshots(snapshots: Dict[pd.Timestamp, Set[str]]) -> Dict[pd.Timestamp, Set[str]]:
    """A truncated or block-page capture would otherwise emit hundreds of fake OUT then IN events."""
    if len(snapshots) < 3:
        return snapshots
    median = float(np.median([len(s) for s in snapshots.values()]))
    kept = {d: s for d, s in snapshots.items() if abs(len(s) - median) <= SNAPSHOT_SIZE_TOLERANCE * median}
    dropped = sorted(set(snapshots) - set(kept))
    if dropped:
        logger.warning(f"dropped {len(dropped)} outlier-sized snapshots (median {median:.0f}): {[d.date() for d in dropped[:10]]}")
    return kept


def snapshots_to_events(snapshots: Dict[pd.Timestamp, Set[str]], source: str = "snapshot") -> pd.DataFrame:
    """Diffs consecutive snapshots into approximate IN/OUT events dated at the later snapshot."""
    dates = sorted(snapshots)
    syms: List[str] = []
    when: List[pd.Timestamp] = []
    acts: List[str] = []
    for prev, cur in zip(dates[:-1], dates[1:]):
        added, removed = snapshots[cur] - snapshots[prev], snapshots[prev] - snapshots[cur]
        syms += [*added, *removed]
        when += [cur] * (len(added) + len(removed))
        acts += ["IN"] * len(added) + ["OUT"] * len(removed)
    return _events_frame(syms, when, acts, "approx", source)


# --------------------------------------------------------------------------
# Source 5: manual overrides
# --------------------------------------------------------------------------

def load_manual_events(path: Path) -> pd.DataFrame:
    """CSV with columns symbol,date,action (IN/OUT). Empty frame if the file doesn't exist."""
    if not path.exists():
        return _empty_events()
    raw = pd.read_csv(path)
    symbols = _to_yf_symbol(raw["symbol"])
    actions = raw["action"].astype(str).str.strip().str.upper()
    valid = symbols.notna() & actions.isin(["IN", "OUT"])
    events = _events_frame(
        symbols[valid], pd.to_datetime(raw["date"])[valid], actions[valid], "exact", "manual_override"
    )
    logger.info(f"loaded {len(events)} manual events from {path}")
    return events


# --------------------------------------------------------------------------
# Interval reconstruction
# --------------------------------------------------------------------------

def _drop_shadowed_approx(events: pd.DataFrame) -> pd.DataFrame:
    """Drops approx events within SHADOW_WINDOW_DAYS of an exact event for the same symbol and action."""
    if events.empty or "precision" not in events.columns:
        return events
    exact = events[events["precision"] == "exact"]
    approx = events[events["precision"] != "exact"]
    if exact.empty or approx.empty:
        return events
    left = approx.sort_values("date")
    right = exact[["symbol", "action", "date"]].sort_values("date").assign(_hit=1)
    shadowed = pd.merge_asof(
        left, right, on="date", by=["symbol", "action"],
        direction="nearest", tolerance=pd.Timedelta(days=SHADOW_WINDOW_DAYS),
    )["_hit"].notna().to_numpy()
    return pd.concat([exact, left[~shadowed]], ignore_index=True)


def build_intervals(
    all_events: pd.DataFrame,
    current_members: Set[str],
    as_of: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """
    Reduces events to [start, end] intervals per symbol (state machine: starts OUT, IN opens, OUT closes,
    repeated actions ignored), fully vectorised:
      1. drop each symbol's events before its first IN,
      2. collapse runs of identical consecutive actions (keep the first),
      3. what remains alternates IN, OUT, ...: each IN pairs with the next row if it is that symbol's OUT.
    Reconciliation with the live list:
      open interval, not a current member -> closed at `as_of` (missed exclusion event);
      current member whose last interval is closed -> re-opened at `as_of` (missed re-entry; no phantom past membership);
      current member with no events -> open from EARLIEST_COVERAGE (unknown join date, logged).
    """
    as_of = as_of or pd.Timestamp.today().normalize()
    current = set(current_members)

    iv = pd.DataFrame(
        {
            "symbol": pd.Series(dtype=object),
            "start": pd.Series(dtype="datetime64[ns]"),
            "end": pd.Series(dtype="datetime64[ns]"),
        }
    )
    ev = _drop_shadowed_approx(all_events)
    ev = ev[ev["action"].isin(("IN", "OUT"))][["symbol", "date", "action"]].copy()
    if not ev.empty:
        ev["date"] = pd.to_datetime(ev["date"])
        ev = ev.sort_values(["symbol", "date"]).reset_index(drop=True)  # multi-key sort is stable

        seen_in = ev["action"].eq("IN").astype(int).groupby(ev["symbol"]).cummax().astype(bool)
        ev = ev[seen_in].reset_index(drop=True)

        repeat = ev["symbol"].eq(ev["symbol"].shift()) & ev["action"].eq(ev["action"].shift())
        ev = ev[~repeat].reset_index(drop=True)

        closes = ev["symbol"].eq(ev["symbol"].shift(-1)) & ev["action"].shift(-1).eq("OUT")
        end = ev["date"].shift(-1).where(closes)
        ins = ev["action"].eq("IN")
        iv = pd.DataFrame({"symbol": ev.loc[ins, "symbol"], "start": ev.loc[ins, "date"], "end": end[ins]})
        iv["end"] = iv["end"].mask(iv["end"].isna() & ~iv["symbol"].isin(current), as_of)

    iv = iv.sort_values(["symbol", "start"], kind="stable").reset_index(drop=True)

    last = iv.groupby("symbol").tail(1)
    reopen = last.loc[last["symbol"].isin(current) & last["end"].notna(), "symbol"].to_numpy()
    unknown_start = sorted(current - set(iv["symbol"]))
    extra = []
    if len(reopen):
        logger.warning(f"{len(reopen)} current members had a closed last interval (missed re-entry); re-opened at {as_of.date()}: {list(reopen[:10])}")
        extra.append(pd.DataFrame({"symbol": reopen, "start": as_of, "end": pd.NaT}))
    if unknown_start:
        logger.warning(
            f"{len(unknown_start)} current members have no events; start floored to {EARLIEST_COVERAGE.date()}: {unknown_start[:10]}"
        )
        extra.append(pd.DataFrame({"symbol": unknown_start, "start": EARLIEST_COVERAGE, "end": pd.NaT}))
    if extra:
        iv = pd.concat([iv, *extra], ignore_index=True).sort_values(["symbol", "start"], kind="stable")

    if iv.empty:
        return pd.DataFrame({"symbol": [], "intervals": []})

    iv = iv.reset_index(drop=True)
    symbols = iv["symbol"].to_numpy()
    pairs = [[s, e] for s, e in zip(iv["start"].tolist(), iv["end"].tolist())]
    edges = np.concatenate(([0], np.flatnonzero(symbols[1:] != symbols[:-1]) + 1, [len(symbols)]))
    return pd.DataFrame(
        {"symbol": symbols[edges[:-1]], "intervals": [pairs[i:j] for i, j in zip(edges[:-1], edges[1:])]}
    )


# --------------------------------------------------------------------------
# Placeholder membership (unblocks downstream work; NOT survivorship-bias-free)
# --------------------------------------------------------------------------

DEFAULT_SEED_CSV = config.DATA_DIR / config.INDEX_MEMBERS_SUBDIR / "seed" / "nifty500_symbols_seed.csv"


def load_seed_csv(csv_path: Path = DEFAULT_SEED_CSV) -> pd.DataFrame:
    """Official NSE export or any CSV with a Symbol column -> ['symbol', 'isin', 'company_name', 'industry']."""
    raw = pd.read_csv(csv_path)
    symbol_col = _find_col(raw.columns, ["symbol"])
    if not symbol_col:
        raise ValueError(f"No symbol-like column in {csv_path}. Columns: {list(raw.columns)}")

    def col(hints: List[str]) -> Optional[pd.Series]:
        name = _find_col(raw.columns, hints)
        return raw[name].astype(str).str.strip() if name else None

    out = pd.DataFrame(
        {
            "symbol": _to_yf_symbol(raw[symbol_col]),
            "isin": col(["isin"]),
            "company_name": col(["company name", "company", "security name"]),
            "industry": col(["industry", "sector"]),
        }
    )
    return out.dropna(subset=["symbol"]).drop_duplicates("symbol").reset_index(drop=True)


def build_fake_membership(
    symbols: List[str],
    data_dir: Path = config.DATA_DIR,
    index_name: str = "nifty500",
    isin_map: Optional[Dict[str, str]] = None,
    company_map: Optional[Dict[str, str]] = None,
    industry_map: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Every symbol a full-time member from EARLIEST_COVERAGE to today. This is today's list projected
    backwards, i.e. exactly the survivorship bias this module exists to remove. Swap for
    build_index_membership() before trusting any backtest. ISIN is carried through because it is the
    identity that survives ticker renames (e.g. ZOMATO -> ETERNAL).
    """
    symbols = sorted({s.strip().upper() for s in symbols if s.strip()})
    df = pd.DataFrame(
        {"symbol": symbols, "intervals": [[[EARLIEST_COVERAGE, pd.NaT]] for _ in symbols]}
    )
    for name, mapping in (("isin", isin_map), ("company_name", company_map), ("industry", industry_map)):
        if mapping:
            df[name] = [mapping.get(s) for s in symbols]

    save_members(df, data_dir, index_name)
    logger.warning(f"FAKE membership for {len(df)} symbols: survivorship-biased placeholder only")
    return df


# --------------------------------------------------------------------------
# Storage & queries
# --------------------------------------------------------------------------

def save_members(df: pd.DataFrame, data_dir: Path, index_name: str) -> Path:
    out_dir = Path(data_dir) / config.INDEX_MEMBERS_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{index_name}.parquet"
    df.to_parquet(path)
    logger.info(f"saved {len(df)} symbols -> {path}")
    return path


def load_members(data_dir: Path, index_name: str) -> pd.DataFrame:
    return pd.read_parquet(Path(data_dir) / config.INDEX_MEMBERS_SUBDIR / f"{index_name}.parquet")


def explode_intervals(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    symbol/intervals table -> parallel arrays (symbols, starts, ends), one row per interval, so a
    point-in-time or overlap query is a single vectorised comparison. Open ends (NaT) become
    Timestamp.max. Shared with universe.py.
    """
    symbols, starts, ends = [], [], []
    for symbol, intervals in zip(df["symbol"], df["intervals"]):
        for start, end in intervals:
            symbols.append(symbol)
            starts.append(start)
            ends.append(end if pd.notna(end) else pd.Timestamp.max)
    return (
        np.array(symbols, dtype=object),
        np.array(starts, dtype="datetime64[ns]"),
        np.array(ends, dtype="datetime64[ns]"),
    )


def members_as_of(df: pd.DataFrame, date: pd.Timestamp) -> Set[str]:
    symbols, starts, ends = explode_intervals(df)
    d64 = pd.Timestamp(date).to_datetime64()
    return set(symbols[(starts <= d64) & (ends >= d64)].tolist())


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def build_index_membership(
    index_key: str = "nifty500",
    data_dir: Path = config.DATA_DIR,
    diagnostics: bool = False,
) -> pd.DataFrame:
    data_dir = Path(data_dir)
    session = _nse_session()

    current = fetch_current_constituents(session, index_key)
    if not current:
        # Without the anchor every open interval would be closed as a "missed exclusion".
        raise RuntimeError("live constituent list unavailable; refusing to build membership without it")

    snapshots = {
        **fetch_monthly_snapshots(session),
        **fetch_wayback_snapshots(index_key),
    }
    snapshots = _drop_outlier_snapshots(snapshots)
    snapshots[pd.Timestamp.today().normalize()] = current  # closes the gap between the last capture and today

    manual_path = data_dir / config.INDEX_MEMBERS_SUBDIR / "manual" / f"{index_key}_manual_events.csv"
    all_events = pd.concat(
        [
            fetch_incl_excl_events(session, index_key),
            snapshots_to_events(snapshots),
            load_manual_events(manual_path),
        ],
        ignore_index=True,
    )

    if diagnostics:
        _print_coverage_report(all_events, current)

    members = build_intervals(all_events, current)
    save_members(members, data_dir, index_key)
    return members


def _print_coverage_report(all_events: pd.DataFrame, current: Set[str]) -> None:
    print("\n--- coverage report ---")
    print(f"current constituents: {len(current)}")
    if all_events.empty:
        print("NO EVENTS FROM ANY SOURCE: check network/URLs above.")
        return
    for source, g in all_events.groupby("source"):
        print(f"{source:>20}: {len(g):5d} events, {g['date'].min().date()} -> {g['date'].max().date()}")
    print(f"{'distinct symbols':>20}: {all_events['symbol'].nunique()}")
    print("-----------------------\n")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build point-in-time index membership from free NSE sources.")
    p.add_argument("--index", default="nifty500", choices=list(CURRENT_CSV_URLS))
    p.add_argument("--data-dir", default=str(config.DATA_DIR))
    p.add_argument("--diagnostics", action="store_true", help="print a per-source coverage report")
    p.add_argument("--fake", action="store_true", help="placeholder membership from --seed-csv (NOT bias-free)")
    p.add_argument("--seed-csv", default=str(DEFAULT_SEED_CSV), help="CSV with a 'Symbol' column, for --fake")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    data_dir = Path(args.data_dir)

    if args.fake:
        seed = load_seed_csv(Path(args.seed_csv))
        maps = {
            key: dict(zip(seed["symbol"], seed[col])) if seed[col].notna().any() else None
            for key, col in (("isin_map", "isin"), ("company_map", "company_name"), ("industry_map", "industry"))
        }
        build_fake_membership(seed["symbol"].tolist(), data_dir=data_dir, index_name=args.index, **maps)
        return

    build_index_membership(args.index, data_dir, args.diagnostics)


if __name__ == "__main__":
    main()
