# Production Momentum Research System

This is the integrated version of the uploaded momentum project plus the point-in-time data pipeline.

## What changed

The production flow is now:

    index membership
        -> daily OHLCV cache
        -> point-in-time eligibility
        -> monthly adjusted-close panel
        -> lagged features
        -> cross-sectional normalization
        -> purged/embargoed walk-forward Ridge
        -> monthly portfolio
        -> transaction costs
        -> risk diagnostics

The synthetic data loader from the original ZIP is no longer used by the production entry point. The uploaded
yfinance cache/loader is authoritative.

## Run order

1. Create an environment and install dependencies:

    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

2. Build historical NIFTY 500 membership:

    python index_members.py --index nifty500 --data-dir data --diagnostics
    python index_members.py --index nifty500 --data-dir data

   The first command is deliberately a diagnostic check. The builder refuses to create a file when it cannot
   recover any membership events rather than silently producing a survivorship-biased history.

3. Run the production research pipeline:

    python main.py --index nifty500 --data-dir data --history-start 2008-01-01 --history-end 2023-12-31 --holdout-start 2021-01-01

   Data is downloaded once and cached under `data/raw/1d/`. Later runs reuse the cache and fetch only missing tails.

4. Optional Phase 1 research sweep:

    python -c "import data_loader; from sweep_fix_k_n import sweep, generate_feedback; import config"

   In practice, call `sweep()` from a notebook/script after building the same point-in-time eligibility mask.
   Do not use the final holdout to select k/n.

## Output

`main.py` prints:
  - full adaptive walk-forward metrics
  - a separate frozen holdout evaluation
  - model alpha diagnostics
  - turnover and missing-return diagnostics

## Important data caveat

The supplied historical membership builder is intentionally conservative about data gaps, but its free-source
history remains approximate for periods where only snapshot archives are available. The final backtest should be
treated as research until the membership history, corporate actions, surveillance history, and delisting outcomes
have been independently audited.
