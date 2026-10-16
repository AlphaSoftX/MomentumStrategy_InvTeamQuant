# Point-in-time Nifty 500 membership: limitations of free data and of the current code

Scope: `index_members.py` (with the fixes applied so far), `config.py`, `utils.py`, and the console output and `nifty500.parquet` from your run.
Status key used below: **Observed** = seen in your log or parquet. **Reasoned** = follows from the code or the data format. **Untested** = plausible but not verified.

---

## 1. Bottom line

The file `nifty500.parquet` is **not point-in-time clean** for roughly **2013-10 → 2022-05**, and it is weak before 2013 as well.

| Date | Members in file | True size |
|---|---|---|
| 2008-01-01 | 383 | ~500 |
| 2012-01-01 | 468 | ~500 |
| 2018-01-01 | 445 | ~500 |
| 2022-05-03 | 461 | ~500 |
| 2022-05-05 | 506 | ~500 |

(Those counts were taken before the last two fixes, so reruns will differ. The 2022-05-04 jump to 570 and the 2026 count of 528 came from an inclusive-end bug that is now fixed.)

Right before the 2022 gap closes, about 109 true members were missing and about 64 stale ones were present (**Observed**). The main cause is data coverage. The remaining part is code that does not extract everything the available data could give. Do not use this file to judge holdout results from 2021 onward until the gap is closed.

---

## 2. Limitations of the free data sources

### 2.1 What exists, and what each source cannot do

| Source | What it gives | Main limitation |
|---|---|---|
| `IndexInclExcl.xls` | Exact IN/OUT events, 1998 → 2020-09-14 | Company **names only, no tickers**. Ends in 2020. Row wording and date formats are inconsistent. |
| niftyindices monthly zips | Full lists, Apr 2013 → Mar 2022 | Only Apr-Sep 2013 contain a constituent CSV (**Observed**). From Oct 2013 the zips hold PDFs with names and weights, **no tickers**. |
| Wayback captures of the live CSV | Full lists, 2022 → today | Only captures from 2022 were requested. Capture date is not the date the list changed. |
| Live constituent CSV | Today's 500 | Only the present. |
| Manual events CSV | Whatever you add | Needs hand work. You have none yet. |

### 2.2 Structural problems (apply to any free source)

1. **Name-to-ticker gap.** The two sources that cover 2013-2020 identify stocks by name. Mapping a 2015 company name to a ticker is guesswork unless you have a historical name table.
2. **Delisted, merged, and renamed companies.** Many removed stocks no longer have a ticker. yfinance generally does not carry delisted stocks, so even a perfect membership file cannot remove survivorship bias on the price side. (**Reasoned**; check your own `CHECK_YFINANCE_AVAILABILITY` results.)
3. **A multi-year hole with no source.** There are no exact events after 2020-09-14 and no snapshots between 2013-09-30 and 2022-05-04 (**Observed**). No free source we found covers that window.
4. **No effective-date precision for snapshots.** A snapshot only says "present on or before this date". Changes inside a gap can only be dated at the end of the gap.
5. **Format drift.** The xls mixes true datetimes and `dd-mm-yyyy` strings (**Observed**), and PDFs change layout over the years. Parsers break silently when this happens.
6. **Site access is fragile.** NSE and niftyindices block bare requests and change URLs. The fetchers worked in your run, but nothing guarantees they will next month.
7. **Unexplained size anomalies in the sources.**
   - The 2013 monthly CSVs shrink from 500 to 475 symbols over six months. The real index is always 500, so this is likely parse loss, but the cause was **not verified**.
   - Several Wayback lists have 501 symbols. The cause is also unverified.

### 2.3 What free data can realistically give you

- A good Nifty 500 history from roughly **2022 onward** (Wayback snapshots).
- A usable but incomplete history from **1998-2020** (exact events, limited by name resolution).
- A rough picture for **2013-2022** that depends on how many names you can resolve.
- Nothing authoritative for **2020-09 → 2022-05** without press releases or a paid dataset.

---

## 3. Limitations in the current (fixed) code

These are things the **available data could support** but the code does not yet do.

### 3.1 Data it could use but does not

| # | Limitation | Evidence | Impact |
|---|---|---|---|
| 1 | **Name registry only knows today's Nifty 500 names** (plus an optional seed CSV). A name that left the index but is still listed cannot resolve. | 1,245 unresolved xls rows; PDF unresolved counts grow from 64 to ~160 over time (**Observed**). | Biggest single loss of events and of PDF snapshots. Loading NSE's full equity list into `data/index_members/names/` is supported now but **untested** against the real file. |
| 2 | **Wayback search starts at 2022-01-01 (hard-coded).** | Log shows captures only from 2022 (**Observed**). | Earlier captures of the live CSV, possibly under other URLs, may exist for 2016-2021 and could fill the gap. **Untested.** |
| 3 | **Whole PDFs are rejected if fewer than 475 names resolve.** Unresolved rows are thrown away. | Oct 2013 PDF rejected at 474 (**Observed**). | About 100 monthly PDFs unused. Because rows are discarded, a month with 25 dead names is lost entirely. |
| 4 | **Unresolved xls rows are discarded instead of kept under a name-based ID.** | 440 rows from 2008 on unresolved (**Observed**). | Their IN/OUT dates are lost. Dead companies would be unpriceable anyway, but still-listed ones are not. |
| 5 | **No ISIN or rename tracking.** Symbols are the identity. | Reasoned. | A renamed ticker (old symbol in an old snapshot, new symbol in the live list) produces a fake OUT and fake IN. |
| 6 | **Name matching is simple.** First-wins registry, fuzzy cutoff 0.93, no handling of suffixes like `-old`, no history of company renames. | Names like `Rane Brake Linings Ltd. -old` in the unresolved file (**Observed**). | Some resolvable names stay unresolved, and a few could map to the wrong ticker. |
| 7 | **The 2020-09 → 2022-05 window has no manual events.** | Reasoned. | The code supports `manual/nifty500_manual_events.csv`, but nothing is provided. |

### 3.2 Reconstruction logic

| # | Limitation | Impact |
|---|---|---|
| 8 | **Approximate events are dated at the later snapshot.** | A stock added mid-gap enters late. A stock removed mid-gap stays too long. Stale members and missing members both result. |
| 9 | **Fixed 45-day shadow window** between approximate and exact events. | Too narrow for large gaps, so duplicate or conflicting events can survive. They mostly collapse, but this has not been audited. |
| 10 | **Transitions fully inside a gap are invisible.** If a stock enters and leaves between two snapshots and has no resolvable exact event, it never appears. | Short-lived members (often the interesting ones for momentum) are missing. |
| 11 | **Action labels come from regex on a description field.** | Works on the two values seen (`Inclusion into Index`, `Exclusion from Index`). Any new wording would be misclassified or dropped. |
| 12 | **Date parsing relies on `dayfirst=True` plus `format="mixed"`.** | Correct for the strings seen, but an ambiguous string in another convention would be misread silently. |
| 13 | **Outlier filter assumes most snapshots are roughly the true size** (±10% of the median). | If most snapshots are bad, good ones can be dropped. |
| 14 | **A symbol that appears in no snapshot and has no resolvable event is invisible.** | Cannot be fixed in code. |

### 3.3 Output and engineering

| # | Limitation | Impact |
|---|---|---|
| 15 | **The parquet carries no confidence.** There is no per-interval precision, source, or "floor" flag. | Downstream code cannot exclude uncertain periods, weight them, or warn. |
| 16 | **Fake and real builds are indistinguishable** in the saved file. | A `--fake` file can be mistaken for a real one. |
| 17 | **Snapshot fetching runs in threads that share the name registry.** | Which PDFs resolve can depend on which CSV months finished first, so results may differ between runs. |
| 18 | **Interval convention must match downstream.** Intervals are now `[start, end)`. `universe.py` also uses `explode_intervals`. | If it still compares `end >= date`, counts will be inflated at transitions. |
| 19 | **No automatic quality gate beyond "some history exists".** The size check is printed only with `--diagnostics`. | A build with 380 members in 2008 still saves. |
| 20 | **Large parts of the fetch layer have not been regression-tested.** Only synthetic cases for the interval logic were tested. | New site changes will not be caught until a run fails. |

---

## 4. Which items are fixed and which are not

**Fixed in the current `index_members.py`:**

- Crash from empty source frames (object dtype in `merge_asof`).
- Silent loss of ~30% of xls rows from mixed date formats, plus the day/month swap.
- Stocks with an exact OUT but no earlier IN vanishing from history.
- The "no history" guard that could never fire.
- Current members left closed after an exact OUT that both snapshots contradict (the six re-opened names). This uses snapshot presence as evidence. Verified on a synthetic case, **not rerun on your real data**.
- Inclusive interval end inflating counts (570 and 528).
- Support for extra name files in `data/index_members/names/`.
- A year-by-year size check printed with `--diagnostics`.

**Not fixed (needs data, effort, or decisions):** items 2-7 and 15-17 in section 3.

---

## 5. Suggested next steps, by expected benefit

1. **Add NSE's full equity list** to `data/index_members/names/` and rerun with `--diagnostics`. Watch the size table and the number of rejected PDFs.
2. **Extend the Wayback start date** to 2016 or earlier and test other archived URLs for the live CSV.
3. **Keep unresolved PDF rows as placeholder names** so snapshots are complete and diffs are correct even for dead companies. Filter out header and footer noise first. This needs a look at one raw PDF's text.
4. **Hand-patch 2020-09 → 2022-05** from NSE reconstitution press releases through the manual events file. This is about two reconstitutions a year.
5. **Store `precision` and `source` per interval** in the parquet, and add a per-date confidence or "members_as_of count" column, so the backtest can drop low-confidence periods.
6. **Add a hard gate:** fail the build, or flag the file, if `members_as_of` is not within about 5% of 500 for dates in your test window.
7. **Align `universe.py`** with the `[start, end)` convention and test it against `members_as_of`.
8. **Consider a paid point-in-time constituents dataset** if the 2021+ holdout must be defensible. Free data cannot reach that standard for the 2020-2022 window.

---

## 6. Validation checklist for any new build

- `members_as_of` is ~500 (±5%) for every year-start from 2008 on.
- No jump above +/-30 members between consecutive days outside a known reconstitution date.
- Every current constituent has an open interval whose start is plausible (not today unless it is a recent addition).
- Spot-check 10 known historical additions and 10 removals against press releases.
- The share of exact versus approximate versus floor-dated events is reported for each year.
- Unresolved names, by year, are reviewed for still-listed companies.
