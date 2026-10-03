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
import difflib
import io
import random
import re
import threading
import time
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

PDF_MIN_RESOLVED = 475    # a PDF snapshot must resolve >= this many of ~500 names, else partial parsing would fake IN/OUT churn
NAME_FUZZY_CUTOFF = 0.93
SNAPSHOT_GAP_WARN_DAYS = 200

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
# Company-name -> ticker registry
# Several NSE sources (IndexInclExcl.xls, old factsheet PDFs) identify stocks by NAME only.
# Every CSV we parse that has both a name and a symbol feeds this registry.
# --------------------------------------------------------------------------

_NAME_MAP: Dict[str, str] = {}
_NAME_LOCK = threading.Lock()
_NAME_COL_HINTS = ["company name", "security name", "scrip name", "company"]


def _norm_name(name) -> str:
    s = str(name).lower().replace("&", " and ")
    s = re.sub(r"\(.*?\)", " ", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    s = re.sub(r"\b(limited|ltd|pvt|private|corporation|corp|company|co|inc)\b", " ", s)
    return " ".join(s.split())


def _register_names(df: pd.DataFrame) -> None:
    sym_col = _find_col(df.columns, ["symbol"])
    name_col = _find_col(df.columns, _NAME_COL_HINTS)
    if not (sym_col and name_col):
        return
    syms = _to_yf_symbol(df[sym_col])
    with _NAME_LOCK:
        for n, sy in zip(df[name_col], syms):
            if isinstance(sy, str) and pd.notna(n):
                _NAME_MAP.setdefault(_norm_name(n), sy)


def load_name_aliases(path: Path) -> int:
    """Hand-curated name,symbol pairs (data/index_members/manual/name_aliases.csv). They take priority."""
    if not path.exists():
        return 0
    raw = pd.read_csv(path)
    syms = _to_yf_symbol(raw["symbol"])
    with _NAME_LOCK:
        for n, sy in zip(raw["name"], syms):
            if isinstance(sy, str):
                _NAME_MAP[_norm_name(n)] = sy
    return len(raw)


def resolve_name(name, fuzzy: bool = True) -> Optional[str]:
    key = _norm_name(name)
    if not key:
        return None
    with _NAME_LOCK:
        hit = _NAME_MAP.get(key)
        if hit or not fuzzy:
            return hit
        close = difflib.get_close_matches(key, list(_NAME_MAP), n=1, cutoff=NAME_FUZZY_CUTOFF)
        return _NAME_MAP[close[0]] if close else None


def _http_get(url: str, *, params=None, timeout: float = 30, attempts: int = 4, headers=None):
    """GET with manual backoff. requests' Retry does not cover SSL/connection resets, which web.archive.org produces under load."""
    last: Optional[Exception] = None
    for i in range(1, attempts + 1):
        target = url
        if i == attempts and url.startswith("https://web.archive.org"):
            target = "http://" + url[len("https://"):]  # last resort for flaky TLS in front of the archive
        try:
            resp = _http_session(headers or HEADERS).get(target, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            last = exc
            time.sleep(1.5 * i + random.uniform(0, 1))
    raise last  # type: ignore[misc]


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
    _register_names(df)
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

_OUT_RE = re.compile(r"excl|remov|delet|dropp|\bout\b|replac(ed|ing)? by", re.I)
_IN_RE = re.compile(r"incl|add|insert|\bin\b|enter", re.I)


def _classify_action(text: str) -> Optional[str]:
    t = str(text)
    if _OUT_RE.search(t):
        return "OUT"
    if _IN_RE.search(t):
        return "IN"
    return None


def parse_incl_excl_frame(raw: pd.DataFrame, unresolved_sink: Optional[list] = None) -> pd.DataFrame:
    """
    Turns the IndexInclExcl sheet into exact IN/OUT events. The real sheet has columns
    ['Index Name', 'Event Date', 'Scrip Name', 'Description'] - NAMES, not tickers - so names are resolved through
    the registry fed by every CSV we parsed (plus manual aliases). Unresolvable rows are logged, never guessed.
    """
    raw = raw.copy()
    raw.columns = [str(c).strip() for c in raw.columns]
    symbol_col = _find_col(raw.columns, ["symbol"])
    name_col = _find_col(raw.columns, ["scrip name", "company name", "security name", "company", "name"])
    action_col = _find_col(raw.columns, ["description", "action", "inclusion", "exclusion", "type"])
    date_col = _find_col(raw.columns, ["effective date", "event date", "date"])
    if not date_col or not (symbol_col or name_col) or not action_col:
        logger.warning(f"cannot identify columns (symbol={symbol_col}, name={name_col}, action={action_col}, date={date_col}); "
                       f"columns were {list(raw.columns)}")
        return _empty_events()

    dates = pd.to_datetime(raw[date_col], errors="coerce")
    actions = raw[action_col].map(_classify_action)
    logger.info(f"IndexInclExcl action values (sample): {raw[action_col].astype(str).value_counts().head(5).to_dict()}")
    if symbol_col:
        symbols = _to_yf_symbol(raw[symbol_col])
    else:
        symbols = raw[name_col].map(resolve_name)

    ok = symbols.notna() & dates.notna() & actions.notna()
    n_total = int(dates.notna().sum())
    n_unmapped = int((symbols.isna() & dates.notna()).sum())
    n_unclassified = int((actions.isna() & dates.notna()).sum())
    logger.info(f"IndexInclExcl: {int(ok.sum())}/{n_total} rows usable "
                f"({n_unmapped} names not resolvable to a ticker, {n_unclassified} actions not classifiable)")
    if unresolved_sink is not None and name_col:
        bad = raw.loc[symbols.isna() & dates.notna(), [name_col, date_col, action_col]]
        unresolved_sink.extend(bad.astype(str).values.tolist())
    return _events_frame(symbols[ok], dates[ok], actions[ok], "exact", "nse_incl_excl_xls")


def fetch_incl_excl_events(session: requests.Session, index_key: str = "nifty500",
                           unresolved_sink: Optional[list] = None) -> pd.DataFrame:
    """Exact IN/OUT events. Returns an empty frame (never raises) on any failure."""
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
    logger.info(f"IndexInclExcl.xls['{sheet}'] columns: {list(raw.columns)}")
    return parse_incl_excl_frame(raw, unresolved_sink)


# --------------------------------------------------------------------------
# Sources 2-4: snapshots
# --------------------------------------------------------------------------

_NUM_TOKEN = re.compile(r"^[\d.,]+%?$")
_PDF_NAME_RE = re.compile(r"(^|/)(cnx|nifty)_?500_[a-z]{3}\d{4}\.pdf$", re.I)


def symbols_from_text_lines(lines: List[str], known_bare: Set[str]) -> Tuple[Set[str], List[str]]:
    """
    Pure function (unit-testable): factsheet text lines -> (resolved tickers, unresolved row-like lines).
    A data row is recognised by a leading serial number or a trailing numeric (weight). The company name is the text
    between them; it is resolved by name, else by a ticker-looking token that is a known symbol.
    """
    found: Set[str] = set()
    unresolved: List[str] = []
    for line in lines:
        toks = line.split()
        if len(toks) < 2:
            continue
        has_serial = bool(re.match(r"^\d{1,3}[.)]?$", toks[0]))
        has_weight = bool(_NUM_TOKEN.match(toks[-1]))
        if not (has_serial or has_weight):
            continue
        body = toks[1:] if has_serial else toks
        while body and _NUM_TOKEN.match(body[-1]):      # strip trailing weight/price columns, keep numbers inside names ("360 ONE WAM")
            body = body[:-1]
        if not body:
            continue
        name_toks = body
        sym = resolve_name(" ".join(body))              # full text, fuzzy allowed
        if sym is None:                                  # name followed by extra columns (industry, series...): try prefixes, exact only
            for k in range(len(body) - 1, 0, -1):
                sym = resolve_name(" ".join(body[:k]), fuzzy=False)
                if sym:
                    break
        if sym is None:
            for t in body:
                bare = re.sub(r"[^A-Z0-9&-]", "", t)
                if len(bare) >= 2 and bare == t.strip(".,") and bare in known_bare:
                    sym = bare + ".NS"
                    break
        if sym:
            found.add(sym)
        elif name_toks:
            unresolved.append(" ".join(name_toks))
    return found, unresolved


def _pdf_lines(pdf_bytes: bytes) -> List[str]:
    try:
        from pypdf import PdfReader
    except ImportError:
        logger.warning("pypdf not installed: cannot read PDF snapshots (pip install pypdf)")
        return []
    reader = PdfReader(io.BytesIO(pdf_bytes))
    out: List[str] = []
    for page in reader.pages:
        try:  # "layout" keeps table cells on one line; plain mode emits every cell as its own line
            text = page.extract_text(extraction_mode="layout") or ""
        except TypeError:  # older pypdf
            text = page.extract_text() or ""
        out.extend(ln.strip() for ln in text.splitlines() if ln.strip())
    return out


def _known_bare_symbols() -> Set[str]:
    with _NAME_LOCK:
        return {v[:-3] for v in _NAME_MAP.values() if v.endswith(".NS")}


def _symbols_from_zip(zip_bytes: bytes) -> Set[str]:
    """Older zips hold a Nifty 500 CSV; most (2013-05 onward) hold only factsheet PDFs, which are parsed best-effort."""
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = zf.namelist()
        csvs = [n for n in names if "500" in n.lower() and n.lower().endswith(".csv")]
        if csvs:
            with zf.open(csvs[0]) as f:
                return _symbols_from_frame(pd.read_csv(f))
        pdfs = [n for n in names if _PDF_NAME_RE.search(n)]
        if not pdfs:
            logger.warning(f"no Nifty 500 csv/pdf in zip; contents: {names[:8]}...")
            return set()
        with zf.open(pdfs[0]) as f:
            found, unresolved = symbols_from_text_lines(_pdf_lines(f.read()), _known_bare_symbols())
        if len(found) < PDF_MIN_RESOLVED:
            logger.warning(f"{pdfs[0]}: resolved only {len(found)} names ({len(unresolved)} unresolved); "
                           f"snapshot rejected (need >= {PDF_MIN_RESOLVED}) to avoid fake IN/OUT churn")
            return set()
        return found


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
        worker_session = _http_session(HEADERS)
        try:
            resp = worker_session.get(url, timeout=20)
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
    session = _http_session(HEADERS)

    captures: Dict[str, str] = {}  # wayback timestamp -> original url
    for target in WAYBACK_TARGETS[index_key]:
        try:
            resp = _http_get(
                WAYBACK_CDX_URL,
                params={
                    "url": target, "output": "json", "filter": "statuscode:200",
                    "from": start.strftime("%Y%m%d"), "to": end.strftime("%Y%m%d"),
                    "collapse": "timestamp:6",
                },
            )
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
            resp = _http_get(f"https://web.archive.org/web/{ts}if_/{target}")
            symbols = _symbols_from_frame(pd.read_csv(io.StringIO(resp.text)))
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"wayback capture {ts} failed: {exc}")
            return None
        if not symbols:
            return None
        logger.info(f"wayback snapshot {ts[:8]}: {len(symbols)} symbols")
        return pd.Timestamp(ts[:8]), symbols

    return dict(r for r in _map_threads(one, sorted(captures.items()), workers=2) if r)


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
    """
    Diffs consecutive snapshots into approximate IN/OUT events dated at the later snapshot.
    Members of the FIRST snapshot get an IN event at EARLIEST_COVERAGE (precision "floor": start unknown). Without it a
    stock that was in the first snapshot and later removed has only an OUT event, which build_intervals discards, so it
    would vanish from history entirely (pure survivorship bias).
    """
    dates = sorted(snapshots)
    syms: List[str] = []
    when: List[pd.Timestamp] = []
    acts: List[str] = []
    precision: List[str] = []
    if dates:
        first = sorted(snapshots[dates[0]])
        syms += first
        when += [EARLIEST_COVERAGE] * len(first)
        acts += ["IN"] * len(first)
        precision += ["floor"] * len(first)
    for prev, cur in zip(dates[:-1], dates[1:]):
        added, removed = snapshots[cur] - snapshots[prev], snapshots[prev] - snapshots[cur]
        syms += [*added, *removed]
        when += [cur] * (len(added) + len(removed))
        acts += ["IN"] * len(added) + ["OUT"] * len(removed)
        precision += ["approx"] * (len(added) + len(removed))
    out = _events_frame(syms, when, acts, "approx", source)
    out["precision"] = precision
    return out


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
    if not ev.empty and "precision" in ev.columns:
        # "floor" = unknown start. If exact events exist for the symbol, they know better.
        exact_syms = set(ev.loc[(ev["precision"] == "exact"), "symbol"])
        ev = ev[~((ev["precision"] == "floor") & ev["symbol"].isin(exact_syms))]
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
    members_dir = data_dir / config.INDEX_MEMBERS_SUBDIR
    session = _nse_session()

    # Seed names->tickers first (and any hand-made aliases): several sources identify stocks by NAME only.
    if DEFAULT_SEED_CSV.exists():
        try:
            _register_names(pd.read_csv(DEFAULT_SEED_CSV))
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"seed csv not usable for name mapping: {exc}")
    n_alias = load_name_aliases(members_dir / "manual" / "name_aliases.csv")

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

    unresolved: list = []
    manual_path = members_dir / "manual" / f"{index_key}_manual_events.csv"
    all_events = pd.concat(
        [
            fetch_incl_excl_events(session, index_key, unresolved),   # runs AFTER snapshots so the name registry is filled
            snapshots_to_events(snapshots),
            load_manual_events(manual_path),
        ],
        ignore_index=True,
    )

    if unresolved:
        out = members_dir / "unresolved_names.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(unresolved, columns=["name", "date", "description"]).to_csv(out, index=False)
        logger.warning(f"{len(unresolved)} include/exclude rows could not be mapped to a ticker -> {out}. "
                       f"Add 'name,symbol' rows to manual/name_aliases.csv and re-run.")

    if all_events.empty:
        raise RuntimeError(
            "No membership events were recovered. Refusing to create a historical membership file "
            "that could silently become a survivorship-biased backtest."
        )

    if diagnostics:
        _print_coverage_report(all_events, current, snapshots, n_alias, len(unresolved))

    members = build_intervals(all_events, current)
    save_members(members, data_dir, index_key)
    return members


def _snapshot_gaps(snapshots: Dict[pd.Timestamp, Set[str]]) -> List[Tuple[pd.Timestamp, pd.Timestamp, int]]:
    d = sorted(snapshots)
    return [(a, b, (b - a).days) for a, b in zip(d[:-1], d[1:]) if (b - a).days > SNAPSHOT_GAP_WARN_DAYS]


def _print_coverage_report(all_events, current, snapshots=None, n_alias=0, n_unresolved=0) -> None:
    print("\n--- coverage report ---")
    print(f"current constituents: {len(current)}")
    if all_events.empty:
        print("NO EVENTS FROM ANY SOURCE: check network/URLs above.")
        return
    for source, g in all_events.groupby("source"):
        exact = int((g["precision"] == "exact").sum())
        print(f"{source:>20}: {len(g):5d} events ({exact} exact), {g['date'].min().date()} -> {g['date'].max().date()}")
    print(f"{'distinct symbols':>20}: {all_events['symbol'].nunique()}")
    print(f"{'name aliases loaded':>20}: {n_alias}")
    print(f"{'unmapped names':>20}: {n_unresolved}  (see unresolved_names.csv)")
    if snapshots:
        d = sorted(snapshots)
        print(f"{'snapshots':>20}: {len(d)}, {d[0].date()} -> {d[-1].date()}")
        gaps = _snapshot_gaps(snapshots)
        if gaps:
            print("\n  WARNING: large gaps between snapshots. Every membership change inside a gap is dated at the END of")
            print("  the gap (look-ahead for additions) unless an exact event covers it:")
            for a, b, days in gaps:
                print(f"    {a.date()} -> {b.date()}  ({days // 30} months)")
    print("-----------------------\n")


def inspect_month(month_abbr: str, year: int) -> None:
    """Debug helper: download ONE monthly zip, show how its Nifty 500 PDF is read, and what fails to resolve."""
    url = MONTHLY_ZIP_URL_TMPL.format(month_abbr=month_abbr, year=year)
    resp = _http_get(url, headers=HEADERS)
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        pdfs = [n for n in zf.namelist() if _PDF_NAME_RE.search(n)]
        print(f"zip entries matching Nifty 500: {pdfs}")
        if not pdfs:
            print(f"all entries: {zf.namelist()}")
            return
        lines = _pdf_lines(zf.read(pdfs[0]))
    print(f"\n{len(lines)} text lines. First 40:")
    for ln in lines[:40]:
        print("   |", ln)
    found, unresolved = symbols_from_text_lines(lines, _known_bare_symbols())
    print(f"\nresolved {len(found)} symbols; {len(unresolved)} row-like lines unresolved. First 25 unresolved:")
    for u in unresolved[:25]:
        print("   ?", u)


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
    p.add_argument("--inspect-month", metavar="MonYYYY", help="debug one monthly zip, e.g. Aug2013")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    data_dir = Path(args.data_dir)

    if args.inspect_month:
        if DEFAULT_SEED_CSV.exists():
            _register_names(pd.read_csv(DEFAULT_SEED_CSV))
        load_name_aliases(data_dir / config.INDEX_MEMBERS_SUBDIR / "manual" / "name_aliases.csv")
        inspect_month(args.inspect_month[:3], int(args.inspect_month[3:]))
        return

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
