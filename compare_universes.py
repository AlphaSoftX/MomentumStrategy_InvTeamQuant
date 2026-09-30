"""Compare multiple point-in-time eligibility universes without synthetic memberships."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

import config
from data_loader import DataLoader
from universe import Universe
import features
import model_ml
import metrics
from universe_strategy import run_fixed_universe_core

def load_prices_for_indices(indices, data_dir, history_start, history_end):
    u = Universe(data_dir=data_dir)
    warm = pd.Timestamp(history_start) - pd.Timedelta(days=500)
    symbols = set()
    for index in indices:
        symbols |= u.get_universe(index, warm, pd.Timestamp(history_end))
    loader = DataLoader(data_dir=data_dir)
    return loader.get_data(
        sorted(symbols), warm.strftime("%Y-%m-%d"),
        (pd.Timestamp(history_end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        interval=config.PRICE_INTERVAL, wide=True, field="Adj Close"
    ), u

def build_scores_and_memberships(indices, prices, u, k):
    monthly = prices.resample("ME").last()
    masks = {
        name: u.eligibility_panel(name, monthly.index, columns=prices.columns).reindex(
            index=monthly.index, columns=prices.columns, fill_value=False
        )
        for name in indices
    }
    # Train the common score model on the widest supplied universe.
    widest = max(indices, key=lambda x: masks[x].sum(axis=1).mean())
    panel = features.build_feature_panel(np.log(monthly.where(monthly > 0)))
    keys = pd.MultiIndex.from_arrays(
        [panel.index.get_level_values("date"), panel.index.get_level_values("ticker")],
        names=["date", "ticker"],
    )
    panel = panel.loc[masks[widest].stack(future_stack=True).reindex(keys).fillna(False).to_numpy(dtype=bool)]
    z = features.zscore_cross_section(panel, clip=config.ROBUST_Z_LIMIT)
    labels = features.build_labels(np.log(monthly.where(monthly > 0)), k)
    scores, _ = model_ml.train_predict_walkforward(
        z, labels, z.columns.tolist(), k=k,
        embargo=config.EMBARGO_MONTHS, min_train_months=config.MIN_TRAIN_MONTHS,
        alpha_grid=config.ALPHA_GRID, min_train_rows=config.MIN_TRAIN_ROWS,
        inner_validation_fraction=config.INNER_VALIDATION_FRACTION,
    )
    membership = {name: masks[name] for name in indices}
    return monthly, scores, membership

def compare(indices, data_dir=config.DATA_DIR, history_start=None, history_end=None, core_n=50, k=None):
    history_start = history_start or config.HISTORY_START
    history_end = history_end or config.HISTORY_END
    k = int(k or config.PHASE_2_K)
    prices, u = load_prices_for_indices(indices, Path(data_dir), history_start, history_end)
    monthly, scores, masks = build_scores_and_memberships(indices, prices, u, k)
    rows = {}
    for name in indices:
        eq, turn = run_fixed_universe_core(
            monthly, scores, masks, name, core_n, k,
            require_positive_score=config.REQUIRE_POSITIVE_SCORE,
            tx_cost_bps=config.TX_COST_BPS, slippage_bps=config.SLIPPAGE_BPS,
            weighting=config.WEIGHTING, vol_window=config.VOL_WEIGHT_WINDOW,
            enable_dd_overlay=config.ENABLE_DD_OVERLAY,
            missing_return_policy=config.MISSING_RETURN_POLICY,
        )
        rows[name] = {**metrics.compute_metrics(eq), "avg_holdings": turn["n_holdings"].mean()}
    return pd.DataFrame(rows).T

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--indices", nargs="+", default=["nifty500"])
    p.add_argument("--data-dir", default=str(config.DATA_DIR))
    p.add_argument("--history-start", default=config.HISTORY_START)
    p.add_argument("--history-end", default=config.HISTORY_END)
    p.add_argument("--core-n", type=int, default=50)
    args = p.parse_args()
    print(compare(args.indices, args.data_dir, args.history_start, args.history_end, args.core_n).to_string())
