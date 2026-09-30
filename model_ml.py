"""Walk-forward Ridge model with nested purged time validation."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

def make_model(alpha):
    return Ridge(alpha=float(alpha), fit_intercept=True)

def _rank_ic(pred, actual):
    x = pd.Series(np.asarray(pred, dtype=float))
    y = pd.Series(np.asarray(actual, dtype=float))
    return x.corr(y, method="spearman")

def _fit_ridge(X, y, alpha_multiplier):
    alpha = float(alpha_multiplier) * max(len(X), 1)
    model = make_model(alpha)
    model.fit(X, y)
    return model, alpha

def train_predict_walkforward(
    feat_z,
    labels,
    feature_cols,
    k,
    embargo=1,
    min_train_months=36,
    alpha_grid=(0.001, 0.003, 0.01, 0.03, 0.10),
    min_train_rows=200,
    inner_validation_fraction=0.20,
):
    if k < 1:
        raise ValueError("k must be >= 1")
    if not 0 < inner_validation_fraction < 0.5:
        raise ValueError("inner_validation_fraction must be in (0, 0.5)")
    if len(feature_cols) == 0:
        raise ValueError("feature_cols is empty")
    dates = list(pd.DatetimeIndex(sorted(feat_z.index.get_level_values("date").unique())))
    if not dates:
        raise ValueError("feature panel contains no dates")

    date_index = feat_z.index.get_level_values("date")
    scores = {}
    diagnostics = []

    for i, t in enumerate(dates):
        train_end_idx = i - k - embargo
        if train_end_idx < min_train_months:
            continue

        train_dates = dates[:train_end_idx + 1]
        train_mask = date_index.isin(train_dates)
        X_all = feat_z.loc[train_mask, feature_cols].astype(float)
        y_all = labels.reindex(X_all.index).astype(float)
        valid = y_all.notna()
        X_all, y_all = X_all.loc[valid], y_all.loc[valid]
        if len(X_all) < min_train_rows:
            continue

        # Inner validation is itself purged/embargoed so validation labels
        # cannot overlap the inner training sample.
        split = max(1, int(len(train_dates) * (1 - inner_validation_fraction)))
        val_dates = train_dates[split:]
        inner_train_end = split - k - embargo - 1
        if inner_train_end >= 0 and val_dates:
            inner_train_dates = train_dates[:inner_train_end + 1]
            tr_mask = X_all.index.get_level_values("date").isin(inner_train_dates)
            va_mask = X_all.index.get_level_values("date").isin(val_dates)
            Xi_tr, yi_tr = X_all.loc[tr_mask], y_all.loc[tr_mask]
            Xi_val, yi_val = X_all.loc[va_mask], y_all.loc[va_mask]
        else:
            Xi_tr = yi_tr = Xi_val = yi_val = None

        best_mult = float(alpha_grid[0])
        best_ic = np.nan
        if Xi_tr is not None and len(Xi_tr) >= min_train_rows // 2 and len(Xi_val) >= 20:
            for mult in alpha_grid:
                model, _ = _fit_ridge(Xi_tr, yi_tr, mult)
                pred = model.predict(Xi_val)
                ic = _rank_ic(pred, yi_val)
                if np.isfinite(ic) and (not np.isfinite(best_ic) or ic > best_ic):
                    best_ic = float(ic)
                    best_mult = float(mult)

        model, alpha = _fit_ridge(X_all, y_all, best_mult)
        X_today = feat_z.xs(t, level="date")[feature_cols].astype(float)
        pred_today = pd.Series(model.predict(X_today), index=X_today.index, dtype=float)
        for ticker, value in pred_today.items():
            scores[(t, ticker)] = value

        diagnostics.append({
            "date": t,
            "alpha_multiplier": best_mult,
            "alpha": alpha,
            "val_rank_ic": best_ic,
            "n_train": len(X_all),
            "n_inner_train": 0 if Xi_tr is None else len(Xi_tr),
            "n_inner_val": 0 if Xi_val is None else len(Xi_val),
            "max_train_date": train_dates[-1],
        })

    if not scores:
        raise ValueError("No predictions produced; increase history or reduce min_train_months.")
    series = pd.Series(scores, dtype=float)
    series.index = pd.MultiIndex.from_tuples(series.index, names=["date", "ticker"])
    score_df = series.unstack("ticker").sort_index()
    return score_df, pd.DataFrame(diagnostics)
def choose_fixed_alpha_from_diagnostics(diagnostics, default=0.01):
    """Pick the most frequently selected alpha multiplier from development walk-forward folds."""
    if diagnostics is None or diagnostics.empty or "alpha_multiplier" not in diagnostics.columns:
        return float(default)
    counts = diagnostics["alpha_multiplier"].value_counts()
    return float(counts.index[0]) if len(counts) else float(default)


def fit_frozen_predict(
    feat_z,
    labels,
    feature_cols,
    prediction_dates,
    train_end,
    k,
    embargo=1,
    alpha_multiplier=0.01,
    min_train_rows=200,
):
    """Train once using only data whose k-month label is fully known before train_end, then freeze."""
    train_end = pd.Timestamp(train_end)
    pred_dates = [pd.Timestamp(d) for d in prediction_dates if pd.Timestamp(d) > train_end]
    if not pred_dates:
        return pd.DataFrame()
    dates = pd.DatetimeIndex(sorted(feat_z.index.get_level_values("date").unique()))
    eligible_train_dates = [d for d in dates if d + pd.DateOffset(months=int(k + embargo)) <= train_end]
    if not eligible_train_dates:
        raise ValueError("No frozen-training dates satisfy k+embargo before train_end")
    mask = feat_z.index.get_level_values("date").isin(eligible_train_dates)
    X = feat_z.loc[mask, feature_cols].astype(float)
    y = labels.reindex(X.index).astype(float)
    valid = y.notna()
    X, y = X.loc[valid], y.loc[valid]
    if len(X) < min_train_rows:
        raise ValueError("Not enough frozen-training rows")
    model, _ = _fit_ridge(X, y, alpha_multiplier)
    out = {}
    for d in pred_dates:
        if d not in feat_z.index.get_level_values("date"):
            continue
        Xd = feat_z.xs(d, level="date")[feature_cols].astype(float)
        pred = pd.Series(model.predict(Xd), index=Xd.index, dtype=float)
        for ticker, value in pred.items():
            out[(d, ticker)] = value
    if not out:
        return pd.DataFrame()
    s = pd.Series(out, dtype=float)
    s.index = pd.MultiIndex.from_tuples(s.index, names=["date", "ticker"])
    return s.unstack("ticker").sort_index()

