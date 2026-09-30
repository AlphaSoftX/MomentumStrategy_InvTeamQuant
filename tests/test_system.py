import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import features
import model_ml
import strategy
import metrics
import index_members
import pipeline

def make_prices(n_months=150, n_stocks=40, seed=7):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2008-01-31", periods=n_months, freq="ME")
    x = rng.normal(0.006, 0.04, (n_months, n_stocks))
    # Add mild persistence so the toy series exercises the momentum path.
    for i in range(1, n_months):
        x[i] += 0.12 * x[i - 1]
    prices = 100 * np.exp(np.cumsum(x, axis=0))
    return pd.DataFrame(prices, index=idx, columns=[f"S{i:03d}" for i in range(n_stocks)])

def test_feature_timing_and_panel():
    p = make_prices(40, 8)
    panel = features.build_feature_panel(np.log(p), lookbacks=(3, 6, 9, 12))
    assert panel.index.names == ["date", "ticker"]
    first = panel.index.get_level_values("date").min()
    assert first >= pd.Timestamp("2009-02-28")

    # For 3m momentum at t, the formula is logP(t-1)-logP(t-4).
    t = first
    ticker = panel.xs(t, level="date").index[0]
    expected = np.log(p.loc[t - pd.offsets.MonthEnd(1), ticker]) - np.log(
        p.loc[t - pd.offsets.MonthEnd(4), ticker]
    )
    assert np.isclose(panel.loc[(t, ticker), "mom_3m"], expected)

def test_model_and_strategy_end_to_end():
    prices = make_prices(150, 40)
    eligibility = pd.DataFrame(True, index=prices.index, columns=prices.columns)
    logp = np.log(prices)
    panel = features.build_feature_panel(logp)
    # Simulate a stock becoming ineligible; it must disappear from both
    # the model cross-section and the trading selection.
    eligibility.loc[eligibility.index[-6:], eligibility.columns[-1]] = False
    keys = pd.MultiIndex.from_arrays(
        [panel.index.get_level_values("date"), panel.index.get_level_values("ticker")],
        names=["date", "ticker"],
    )
    eligible_rows = eligibility.stack(future_stack=True).reindex(keys).fillna(False)
    panel = panel.loc[eligible_rows.to_numpy(dtype=bool)]
    z = features.zscore_cross_section(panel)
    labels = features.build_labels(logp, 1)
    scores, diagnostics = model_ml.train_predict_walkforward(
        z, labels, z.columns.tolist(), k=1, embargo=1,
        min_train_months=36, alpha_grid=(0.001, 0.01, 0.03),
        min_train_rows=100,
    )
    assert not scores.empty
    assert not diagnostics.empty
    eq, turn = strategy.run_strategy(
        prices, scores, 0.25, 1, eligible=eligibility,
        weighting="inv_vol", missing_return_policy="liquidate"
    )
    assert len(eq) >= 2
    assert np.isfinite(eq.to_numpy()).all()
    assert (turn["turnover"] >= 0).all()
    assert (turn["turnover"] <= 1.0 + 1e-12).all()

def test_missing_return_not_ignored():
    idx = pd.date_range("2020-01-31", periods=4, freq="ME")
    prices = pd.DataFrame(
        {"A": [100.0, 110.0, np.nan, np.nan],
         "B": [100.0, 100.0, 100.0, 100.0]},
        index=idx,
    )
    scores = pd.DataFrame({"A": [1.0, 1.0], "B": [0.0, 0.0]}, index=[idx[0], idx[1]])
    eq, turn = strategy.run_strategy(
        prices, scores, 0.5, 1, weighting="equal",
        require_positive_score=True, missing_return_policy="liquidate"
    )
    assert turn.iloc[1]["missing_returns"] == 1

def test_spearman_ic_is_rank_based():
    pred = np.array([0.1, 0.2, 0.4, 0.3])
    actual = np.array([10, 20, 40, 30])
    assert np.isclose(model_ml._rank_ic(pred, actual), 1.0)

def test_membership_state_machine():
    events = pd.DataFrame({
        "symbol": ["A.NS", "A.NS", "A.NS", "B.NS"],
        "date": pd.to_datetime(["2020-01-31", "2020-06-30", "2020-03-31", "2020-01-31"]),
        "action": ["IN", "OUT", "IN", "IN"],
        "precision": ["exact"] * 4,
        "source": ["test"] * 4,
    })
    out = index_members.build_intervals(events, {"A.NS", "B.NS"}, as_of=pd.Timestamp("2021-01-31"))
    a = out.loc[out["symbol"] == "A.NS", "intervals"].iloc[0]
    assert len(a) >= 1
    assert pd.Timestamp(a[0][0]) == pd.Timestamp("2020-01-31")

def test_frozen_holdout_uses_pre_holdout_only():
    prices = make_prices(150, 30)
    logp = np.log(prices)
    panel = features.zscore_cross_section(features.build_feature_panel(logp))
    labels = features.build_labels(logp, 1)
    holdout = pd.Timestamp("2017-01-31")
    diag = model_ml.train_predict_walkforward(
        panel, labels, panel.columns.tolist(), k=1, embargo=1,
        min_train_months=36, min_train_rows=100,
        alpha_grid=(0.001, 0.01),
    )[1]
    mult = model_ml.choose_fixed_alpha_from_diagnostics(diag.loc[diag["date"] < holdout])
    frozen = model_ml.fit_frozen_predict(
        panel, labels, panel.columns.tolist(),
        prediction_dates=prices.index[prices.index >= holdout],
        train_end=holdout - pd.offsets.MonthEnd(1),
        k=1, embargo=1, alpha_multiplier=mult, min_train_rows=100,
    )
    assert not frozen.empty
    assert frozen.index.min() >= holdout
