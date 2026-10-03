# Design decisions and audit notes

## 1. Authoritative data-loader decision

The ZIP contained its own `data_loader.py` that generated synthetic prices. The separately uploaded `data_loader.py`
is materially different: it downloads OHLCV through yfinance, stores an atomic parquet cache, records coverage in a
sidecar JSON, and returns adjusted close when requested. The integrated system uses the latter.

The production cache stores:
  - `Close`: used for trade-price/liquidity checks.
  - `Adj Close`: used for returns, momentum, and portfolio P&L.
  - `Stock Splits` / `Dividends`: retained so corporate-action-aware auditing is possible.

`auto_adjust=False`, `actions=True`, and `multi_level_index=False` are explicit. Current yfinance documentation
describes `auto_adjust` as controlling automatic OHLC adjustment and `actions=True` as including dividend/split
actions. `repair=True` is enabled because yfinance documents repairs for certain split/adjustment inconsistencies.

## 2. Point-in-time membership and eligibility

The uploaded `index_members.py` reconstructs membership intervals rather than projecting today's constituents backward.
The integrated `Universe` then applies, in order:
  1. membership on the exact signal date
  2. surveillance exclusion on that date when surveillance data exists
  3. trailing price/liquidity checks ending on that date
  4. yfinance availability only for dates close to today

Today's yfinance availability is never used to decide historical eligibility, because that would introduce
survivorship/look-ahead bias.

## 3. Historical data-source limitation

The supplied membership script itself warns that its fetchers need live diagnostics. I did not pretend to verify
the live NSE/Wayback responses inside this environment because network access for package/data downloads is not
available here. The code therefore keeps the diagnostic gate and refuses to write a completely empty membership
history.

NSE Indices currently describes NIFTY 500 as a 500-company broad-market index and offers an Index Constituent
download on the index page. It also advertises historical constituent data as part of its data products.
For research-grade backtests, any approximate free-source reconstruction should be checked against an authoritative
historical constituent dataset.

## 4. Signal timing

For signal date t, momentum features use:

    log(P[t-1]) - log(P[t-L-1])

so the current month return is excluded and the immediately preceding month is not used as a directional
momentum observation.

Features are constructed before model fitting and do not use future prices.

## 5. Eligibility before cross-sectional normalization

Cross-sectional z-scores are computed after point-in-time eligibility filtering. Otherwise, a stock that could
not have been traded on date t could still influence the cross-sectional mean/std used to score the stocks that
were tradable.

A fixed 5-sigma clip is applied after normalization to reduce the influence of pathological feature outliers
without adding a learned parameter.

## 6. Label construction

The training label is the forward k-month log return:

    log(P[t+k]) - log(P[t])

The label is never inserted into the feature panel. Samples with missing endpoints naturally drop out of the
training set, which prevents the model from inventing a realized return for a stock that lacks a valid endpoint.

## 7. Walk-forward training and purging

The outer prediction at date t may use only training rows whose label window is completely finished at least
`embargo` months before t.

Alpha selection is nested inside the training window. Importantly, the inner validation split is itself purged:
the inner training labels must also finish before the validation block after the same k+embargo buffer.

This fixes a subtle weakness in the original ZIP implementation, where the inner time split did not purge
overlapping forward labels.

## 8. Why Ridge

The model is deliberately linear and regularized rather than tree/deep-learning based. Monthly cross-sectional
data contains a lot of noise, and a small feature set plus Ridge has substantially fewer effective degrees of
freedom.

The Ridge `alpha_grid` is expressed as multipliers of the training-row count. This fixes another issue in the
original code: absolute alpha values like 0.1 or 50 become weak regularization as the sample count grows.

## 9. Correct rank IC

The original model comments said alpha was selected using rank correlation, but the implementation used
`np.corrcoef`, which is Pearson correlation. The integrated version uses Spearman correlation through pandas'
`corr(..., method="spearman")`, matching the stated objective.

## 10. Portfolio return handling

Monthly returns use `pct_change(fill_method=None)`. This is explicit and important: implicit forward filling of
missing prices can create artificial returns or silently erase a delisting/suspension event.

The portfolio engine does not silently drop a held stock with a missing next-period return. The default policy is
`liquidate`, which treats the missing holding value as a conservative -100% return and records the event in
`missing_returns`. A strict `raise` mode is also available for data-audit runs.

This is intentionally conservative because free Yahoo data does not provide a reliable universal delisting
proceeds series.

## 11. Turnover and costs

The original ZIP turnover formula used the number of holdings in the symmetric difference. That double-counts a
stock swap because the exiting name and entering name are both counted.

The integrated engine instead uses one-way portfolio turnover:

    0.5 * sum(|target_weight - previous_weight|)

with cash included as an asset. This is the standard portfolio-weight turnover convention and behaves correctly
for both membership changes and exposure changes.

Transaction cost = turnover * (tx_cost_bps + slippage_bps) / 10,000.

## 12. Inverse-volatility weighting

Inverse-volatility weights are computed only from returns through the rebalance date. A missing/zero volatility
estimate falls back to the mean inverse-volatility value; if no finite volatility exists, the basket falls back to
equal weight.

This makes the portfolio engine robust instead of turning a data gap into NaN weights.

## 13. Drawdown overlay

The ZIP had a drawdown overlay enabled by default. I changed the production default to OFF.

Reason: a nonlinear drawdown rule introduces additional model degrees of freedom (trigger and recovery behavior)
and can improve historical charts while increasing overfitting. It remains available for controlled research through
configuration and should not be tuned on the final holdout.

## 14. Holdout design

The system reports two different things:
  - an adaptive walk-forward curve, which is useful for realistic research/live-style behavior but continues to
    learn from older realized data as time advances;
  - a frozen holdout, in which alpha is selected only from diagnostics before the holdout, and the final model is
    fitted only on pre-holdout training data.

The second is the cleaner estimate when the goal is an untouched final test.

## 15. Phase 1 k/n sweep

The simple momentum model is retained only as a research tool for selecting a robust neighborhood of k and n.
The sweep precomputes score panels once per j and caches portfolio curves for each parameter combination instead of
rebuilding the full backtest for every walk-forward fold.

After development selection, k and n are fixed in `config.py`; the final model then uses its own walk-forward
Ridge score generator.

## 16. Universe comparison layer

The original `universes.py` created synthetic size-based NIFTY-style memberships. That remains inappropriate for
production. The integrated comparison helper now expects actual point-in-time membership masks/files and uses them
only as a trading filter.

Do not use the synthetic universe generator for a final performance claim.

## 17. Threading

The data downloader already parallelizes symbols, so each individual yfinance request in the integrated version
is single-threaded. This avoids nested thread pools, is easier to reason about with rate limits, and still achieves
concurrency at the symbol level.

Membership archive downloads use separate HTTP sessions inside worker tasks rather than sharing one requests
session across threads.

## 18. Efficiency

The main expensive objects are:
  - downloaded daily history: cached to disk
  - rolling price/liquidity stats: computed once per symbol inside `Universe`
  - feature construction: vectorized pandas
  - Ridge: only a handful of features
  - sweep: score/backtest reuse

No model should be retrained inside a loop over individual securities.

## 19. Remaining items that are deliberately not hidden

1. The environment used for this audit did not have yfinance/pyarrow/xlrd installed, and package installation was
   blocked by unavailable network access. Therefore I could not execute a live Yahoo/NSE download here.
2. The free historical membership sources in the supplied file need the requested `--diagnostics` run in a networked
   environment.
3. Surveillance history is still optional because no builder for that dataset was supplied.
4. Exact delisting proceeds are not available from the supplied free data, so missing-held-return handling is
   explicitly conservative rather than silently optimistic.

## 20. Audit fixes (second pass)

1. **Silent "no data for any stock" (data_loader.py).** yfinance does not raise when Yahoo rate-limits/blocks or the
   installed version is stale; it returns an empty frame and records the reason in `yfinance.shared._ERRORS`. The loader
   treated that as "no data, skip" and never retried. Now: an empty frame with a non-"delisted" error is a transient failure
   (retried with jittered backoff, returns `None`, never cached); a genuine "delisted/no price data" empty frame is accepted;
   `get_data` raises a clear `RuntimeError` when nothing at all loads. Default workers 8 -> 4.
2. **Version-incompatible kwargs** (`multi_level_index` on old yfinance) used to raise `TypeError` that was swallowed and
   retried as "no data". Kwargs are now filtered by the installed `yf.download` signature; `TypeError` is never masked.
3. **Cache poisoning.** A failed tail extension used to advance the coverage sidecar to "yesterday", so the missing
   tail was never fetched again. Coverage now advances only on a genuine empty response.
4. **Turnover ignored weight drift** (engine compared new target weights to the previous *target*, not to the drifted
   holdings), understating turnover and costs; non-rebalance months were also silently re-equalised for free. The engine now
   drifts holdings buy-and-hold and charges turnover on full weights including cash. The old `_turnover` also applied
   exposure twice when the drawdown overlay was on; `universe_strategy.py` additionally used the post-update exposure.
5. **Sortino** used the std of only the negative returns; it now uses standard downside deviation.
6. `mask_panel` vectorised (was a per-row `.loc` loop). `requirements.txt`: `yfinance>=0.2.54`, `scipy` (needed by `repair=True`).

Not changed, but worth deciding: `TX_COST_BPS + SLIPPAGE_BPS` is charged on one-way turnover. If 15 bps is meant per side,
true cost is ~2x. Sharpe/Sortino use a 0% risk-free rate. Training labels drop stocks that vanish (survivorship in the
fit) while the portfolio charges them -100%; the two are inconsistent but the portfolio side is the conservative one.

## 21. Membership builder audit (from a real `--diagnostics` run)

Observed on a live run: only the Apr-2013 monthly zip contained a CSV (all later zips hold factsheet PDFs only);
IndexInclExcl.xls has columns `Index Name, Event Date, Scrip Name, Description` (company NAMES, no ticker) so it
produced zero events; 4 Wayback fetches died with SSL errors. Net effect: membership history was one snapshot in 2013,
then nothing until 2022.

1. **First-snapshot survivorship bug.** Members of the first snapshot got no IN event, so one that was later removed had only
   an OUT event, which `build_intervals` discards: it vanished from history. First-snapshot members now get an IN at
   `EARLIEST_COVERAGE` (precision `floor`), and a `floor` IN never overrides exact events for the same symbol.
2. **Name->ticker registry.** Every parsed CSV with name+symbol feeds it; `manual/name_aliases.csv` (`name,symbol`) overrides.
   The xls is parsed by name (`Description` is the action column); unmappable rows go to `unresolved_names.csv`, never guessed.
3. **PDF snapshots** are parsed best-effort (pypdf layout mode). A PDF snapshot must resolve >= 475 names or it is rejected,
   because a partial parse would emit hundreds of fake IN/OUT events. Debug a month with
   `python index_members.py --inspect-month Aug2013`.
4. **Wayback**: manual backoff retries (SSL resets are not covered by urllib3 Retry), 2 workers, http fallback on last try.
5. The coverage report now lists snapshot gaps > 200 days: any change inside a gap is dated at the gap's end.

Known limit: companies that left the index and were later delisted/renamed appear in no current CSV, so their names cannot
be mapped to tickers automatically. Use `unresolved_names.csv` + `name_aliases.csv` for the ones that matter.
