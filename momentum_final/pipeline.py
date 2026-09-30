"""Production orchestration: data -> PIT eligibility -> features -> walk-forward model -> backtest."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

import config
from data_loader import DataLoader
from universe import Universe
import features
import model_ml
import strategy
import metrics

def load_production_inputs(index_name="nifty500", data_dir=None, history_start=None, history_end=None):
    data_dir = Path(data_dir or config.DATA_DIR)
    history_start = history_start or config.HISTORY_START
    history_end = history_end or config.HISTORY_END
    warm_start = pd.Timestamp(history_start) - pd.Timedelta(days=500)
    universe = Universe(data_dir=data_dir)
    symbols = sorted(universe.get_universe(index_name, warm_start, pd.Timestamp(history_end)))
    if not symbols:
        raise RuntimeError(f"No symbols found for {index_name} in membership file")

    loader = DataLoader(data_dir=data_dir)
    prices = loader.get_data(
        symbols,
        start=warm_start.strftime("%Y-%m-%d"),
        end=(pd.Timestamp(history_end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        interval=config.PRICE_INTERVAL,
        wide=True,
        field="Adj Close",
    )
    prices = prices.loc[~prices.index.duplicated(keep="last")].sort_index()
    if prices.empty:
        raise RuntimeError("No adjusted-close price data was loaded")
    return prices, universe

def build_pit_eligibility(prices, universe_obj, index_name):
    monthly = prices.resample("ME").last()
    mask = universe_obj.eligibility_panel(index_name, monthly.index, columns=prices.columns)
    return mask.reindex(index=monthly.index, columns=prices.columns, fill_value=False)

def mask_panel(panel, eligibility):
    keep = []
    for date, ticker in panel.index:
        keep.append(
            date in eligibility.index
            and ticker in eligibility.columns
            and bool(eligibility.loc[date, ticker])
        )
    return panel.loc[np.asarray(keep, dtype=bool)]

def _run_portfolio(monthly, scores, eligibility, k):
    equity, turnover = strategy.run_strategy(
        monthly,
        scores,
        config.PHASE_2_N,
        k,
        tx_cost_bps=config.TX_COST_BPS,
        slippage_bps=config.SLIPPAGE_BPS,
        dd_threshold=config.DD_THRESHOLD,
        dd_deleverage=config.DD_DELEVERAGE,
        require_positive_score=config.REQUIRE_POSITIVE_SCORE,
        weighting=config.WEIGHTING,
        vol_window=config.VOL_WEIGHT_WINDOW,
        enable_dd_overlay=config.ENABLE_DD_OVERLAY,
        missing_return_policy=config.MISSING_RETURN_POLICY,
        eligible=eligibility,
    )
    return equity, turnover

def run_model(prices, eligibility, k=None, final_holdout_start=None):
    k = int(k or config.PHASE_2_K)
    monthly = prices.resample("ME").last()
    logp = np.log(monthly.where(monthly > 0))

    panel = features.build_feature_panel(
        logp,
        lookbacks=config.LOOKBACKS,
        skip_recent=config.SKIP_RECENT,
        vol_window=config.VOL_WINDOW,
        high_window=config.HIGH_WINDOW,
    )
    panel = mask_panel(panel, eligibility)
    if panel.empty:
        raise RuntimeError("No point-in-time eligible feature rows remain")

    panel_z = features.zscore_cross_section(panel, clip=config.ROBUST_Z_LIMIT)
    labels = features.build_labels(logp, k)

    scores, diagnostics = model_ml.train_predict_walkforward(
        panel_z,
        labels,
        list(panel_z.columns),
        k=k,
        embargo=config.EMBARGO_MONTHS,
        min_train_months=config.MIN_TRAIN_MONTHS,
        alpha_grid=config.ALPHA_GRID,
        min_train_rows=config.MIN_TRAIN_ROWS,
        inner_validation_fraction=config.INNER_VALIDATION_FRACTION,
    )
    equity, turnover = _run_portfolio(monthly, scores, eligibility, k)

    result = {
        "monthly_prices": monthly,
        "eligibility": eligibility,
        "features": panel_z,
        "labels": labels,
        "scores": scores,
        "ml_diagnostics": diagnostics,
        "equity_curve": equity,
        "turnover": turnover,
        "metrics": metrics.compute_metrics(equity),
    }

    if final_holdout_start:
        hold_start = pd.Timestamp(final_holdout_start)
        result["holdout_metrics"] = metrics.metrics_for_window(equity, start=hold_start)
        dev_diag = diagnostics.loc[diagnostics["date"] < hold_start] if not diagnostics.empty else diagnostics
        fixed_mult = model_ml.choose_fixed_alpha_from_diagnostics(dev_diag)

        frozen_scores = model_ml.fit_frozen_predict(
            panel_z,
            labels,
            list(panel_z.columns),
            prediction_dates=monthly.index[monthly.index >= hold_start],
            train_end=hold_start - pd.offsets.MonthEnd(1),
            k=k,
            embargo=config.EMBARGO_MONTHS,
            alpha_multiplier=fixed_mult,
            min_train_rows=config.MIN_TRAIN_ROWS,
        )
        frozen_scores = frozen_scores.where(
            eligibility.reindex(
                index=frozen_scores.index,
                columns=frozen_scores.columns,
                fill_value=False,
            )
        )
        frozen_eq, frozen_turnover = _run_portfolio(monthly, frozen_scores, eligibility, k)
        frozen_eq = frozen_eq.loc[frozen_eq.index >= frozen_scores.index.min()]

        result["frozen_holdout_scores"] = frozen_scores
        result["frozen_holdout_equity"] = frozen_eq
        result["frozen_holdout_turnover"] = frozen_turnover.loc[
            frozen_turnover.index >= frozen_scores.index.min()
        ]
        result["frozen_alpha_multiplier"] = fixed_mult
        result["frozen_holdout_metrics"] = metrics.compute_metrics(frozen_eq)
    return result

def run(index_name="nifty500", data_dir=None, history_start=None, history_end=None, holdout_start=None):
    prices, universe = load_production_inputs(
        index_name, data_dir, history_start, history_end
    )
    eligibility = build_pit_eligibility(prices, universe, index_name)
    return run_model(
        prices,
        eligibility,
        k=config.PHASE_2_K,
        final_holdout_start=holdout_start or config.FINAL_HOLDOUT_START,
    )
