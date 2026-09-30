"""Central configuration for the production momentum research system."""
from pathlib import Path

DATA_DIR = Path("data")
RAW_SUBDIR = "raw"
INDEX_MEMBERS_SUBDIR = "index_members"
SURVEILLANCE_SUBDIR = "surveillance"
RESULTS_SUBDIR = "results"

PRICE_INTERVAL = "1d"
HISTORY_START = "2008-01-01"
HISTORY_END = "2023-12-31"
FINAL_HOLDOUT_START = "2021-01-01"

# Eligibility is evaluated only with information available on the signal date.
SURVEILLANCE_STAGES = ["ASM", "GSM", "T2T"]
PRICE_FLOOR_INR = 20.0
MIN_AVG_DAILY_TURNOVER_INR = 1_00_00_000.0
LIQUIDITY_LOOKBACK_DAYS = 60
MIN_LOOKBACK_TRADING_DAYS = 30
MAX_STALENESS_DAYS = 7

CHECK_YFINANCE_AVAILABILITY = True
YFINANCE_AVAILABILITY_PROBE_PERIOD = "5d"
YFINANCE_PROBE_BATCH = 100
YFINANCE_REPAIR = True

# Signal/model parameters. These are fixed for the final run; do not tune on holdout.
PHASE_2_K = 1
PHASE_2_N = 0.30
LOOKBACKS = (3, 6, 9, 12)
SKIP_RECENT = 1
VOL_WINDOW = 6
HIGH_WINDOW = 12

# Walk-forward protections.
EMBARGO_MONTHS = 1
MIN_TRAIN_MONTHS = 36
INNER_VALIDATION_FRACTION = 0.20
ALPHA_GRID = (0.001, 0.003, 0.01, 0.03, 0.10)  # multiplied by n_train_rows
MIN_TRAIN_ROWS = 200
ROBUST_Z_LIMIT = 5.0

# Strategy/risk. The DD overlay is OFF by default because it adds a tunable
# nonlinear layer; it can be enabled explicitly for research.
TX_COST_BPS = 10.0
SLIPPAGE_BPS = 5.0
WEIGHTING = "inv_vol"
VOL_WEIGHT_WINDOW = 6
REQUIRE_POSITIVE_SCORE = True
ENABLE_DD_OVERLAY = False
DD_THRESHOLD = 0.15
DD_DELEVERAGE = 0.50
MISSING_RETURN_POLICY = "liquidate"  # "liquidate" | "raise" | "zero"

# Research-only parameter sweep.
SWEEP_J_GRID = (3, 6, 9, 12)
SWEEP_K_GRID = (1, 3, 6)
SWEEP_N_GRID = (0.10, 0.20, 0.30)
SWEEP_TRAIN_MONTHS = 60
SWEEP_TEST_MONTHS = 12

TRADING_DAYS_PER_YEAR = 252
