"""Central constants. No logic; every module imports paths and thresholds from here."""

from pathlib import Path

# --------------------------------------------------------------------------
# Paths (all relative to a data_dir)
# --------------------------------------------------------------------------

DATA_DIR = Path("data")
RAW_SUBDIR = "raw"                       # <data_dir>/raw/<interval>/<symbol>.parquet
INDEX_MEMBERS_SUBDIR = "index_members"   # <data_dir>/index_members/<index>.parquet
SURVEILLANCE_SUBDIR = "surveillance"     # <data_dir>/surveillance/<index>.parquet; no script builds it yet,
                                         # the filter is skipped (with a warning) when the file is missing

# --------------------------------------------------------------------------
# Eligibility: surveillance
# --------------------------------------------------------------------------

# Point-in-time: a symbol is excluded only while flagged on the query date.
# Using flags from later in the holding period would be look-ahead.
SURVEILLANCE_STAGES = ["ASM", "GSM", "T2T"]

# --------------------------------------------------------------------------
# Eligibility: price floor & liquidity
# --------------------------------------------------------------------------

PRICE_INTERVAL = "1d"

PRICE_FLOOR_INR = 20.0                       # min trailing-average *traded* (split-unadjusted) close
MIN_AVG_DAILY_TURNOVER_INR = 1_00_00_000.0   # 1 crore, trailing-average Close * Volume

# Calendar-day window ending at the query date, never after it (no look-ahead).
LIQUIDITY_LOOKBACK_DAYS = 60

# Minimum non-NaN daily bars inside that window (~41 trading days fit in 60 calendar days).
MIN_LOOKBACK_TRADING_DAYS = 30

# Max calendar days between the query date and the last cached bar (weekends + holidays);
# beyond this the symbol is treated as suspended/delisted at that date.
MAX_STALENESS_DAYS = 7

# --------------------------------------------------------------------------
# Live yfinance availability (live trading only)
# --------------------------------------------------------------------------

# Checked only for query dates within MAX_STALENESS_DAYS of today. For historical dates it would
# use today's information to drop names that were tradeable then (look-ahead / survivorship).
CHECK_YFINANCE_AVAILABILITY = True
YFINANCE_AVAILABILITY_PROBE_PERIOD = "5d"
YFINANCE_PROBE_BATCH = 100

# --------------------------------------------------------------------------
# Regime matching (regime.py)
# --------------------------------------------------------------------------

REGIME_SUBDIR = "regime"                 # <data_dir>/regime/{features.parquet, sweep.csv}
REGIME_INDEX = "nifty500"

HISTORY_START = "2000-01-01"
HISTORY_END = "2023-12-31"               # candidate windows must end on or before this
RECENT_START = "2024-01-01"              # recent window = [RECENT_START, last cached bar]

SWEEP_START = "2000-01-01"               # candidate window start dates x
SWEEP_END = "2020-12-31"
SWEEP_STEP_DAYS = 1                      # trading days between candidate starts
BLOCK_YEARS = 5                          # test 1 block length

PRICE_WARMUP_DAYS = 500                  # price history loaded before HISTORY_START (features + liquidity)
MIN_UNIVERSE_SIZE = 30                   # features are NaN on days with fewer eligible stocks
MAX_ABS_DAILY_RETURN = 0.5               # larger daily moves treated as bad data
TRADING_DAYS_PER_YEAR = 252

TREND_WINDOW = 126
VOL_WINDOW = 21
DRAWDOWN_WINDOW = 252
BREADTH_MA = 200
CORR_WINDOW = 63
MOM_LOOKBACK = 126                       # 6-1 momentum: return from t-126 to t-21
MOM_SKIP = 21
IC_HORIZON = 21                          # realised IC: signal at t-h vs return over (t-h, t]
TOP_FRACTION = 0.10                      # top-decile churn
CHURN_LAG = 21
SMOOTH_WINDOW = 21
SMOOTHED_FEATURES = ["dispersion", "mom_ic", "top_churn"]

# Matching weights (0 drops a feature from the score)
FEATURE_WEIGHTS = {
    "trend": 1.0,
    "vol": 1.0,
    "drawdown": 1.0,
    "breadth": 1.0,
    "dispersion": 1.0,
    "correlation": 1.0,
    "mom_ic": 1.0,
    "top_churn": 1.0,
}

N_QUANTILES = 100                        # Wasserstein-1 via quantile grid
TRAJECTORY_POINTS = 60                   # windows resampled to this many points for shape distance
TRAJECTORY_WEIGHT = 0.3                  # score = (1-w) * distribution dist + w * trajectory dist
MIN_VALID_FRACTION = 0.9                 # per-feature non-NaN share required inside a window

TOP_MATCHES = 5                          # non-overlapping best matches reported
NULL_SAMPLES = 200                       # placebo targets for the null
RANDOM_SEED = 0
