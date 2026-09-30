"""Leakage-safe monthly cross-sectional features and forward labels."""
from __future__ import annotations
import numpy as np
import pandas as pd

def _validate_monthly_prices(monthly_log_prices):
    if not isinstance(monthly_log_prices.index, pd.DatetimeIndex):
        raise TypeError("monthly_log_prices must have a DatetimeIndex")
    if monthly_log_prices.empty:
        raise ValueError("monthly_log_prices is empty")

def build_feature_panel(
    monthly_log_prices,
    lookbacks=(3, 6, 9, 12),
    skip_recent=1,
    vol_window=6,
    high_window=12,
):
    _validate_monthly_prices(monthly_log_prices)
    if any(int(L) <= 0 for L in lookbacks):
        raise ValueError("lookbacks must be positive")
    feats = {}
    for L in lookbacks:
        feats[f"mom_{L}m"] = (
            monthly_log_prices.shift(skip_recent)
            - monthly_log_prices.shift(L + skip_recent)
        )
    monthly_rets = monthly_log_prices.diff()
    feats[f"vol_{vol_window}m"] = (
        monthly_rets.shift(skip_recent)
        .rolling(vol_window, min_periods=vol_window)
        .std()
    )
    lagged = monthly_log_prices.shift(skip_recent)
    roll_max = lagged.rolling(high_window, min_periods=high_window).max()
    feats["dist_from_high"] = lagged - roll_max

    # One row per (date, ticker); require all features to exist.
    panel = pd.DataFrame({name: frame.stack(future_stack=True) for name, frame in feats.items()})
    panel.index.names = ["date", "ticker"]
    return panel.dropna()

def build_labels(monthly_log_prices, k):
    if int(k) < 1:
        raise ValueError("k must be >= 1")
    fwd = monthly_log_prices.shift(-int(k)) - monthly_log_prices
    return fwd.stack(future_stack=True).rename("fwd_ret")

def zscore_cross_section(feature_panel, clip=5.0):
    if not isinstance(feature_panel.index, pd.MultiIndex) or list(feature_panel.index.names[:2]) != ["date", "ticker"]:
        raise ValueError("feature_panel must use MultiIndex (date, ticker)")
    numeric = feature_panel.astype(float)
    mean = numeric.groupby(level="date").transform("mean")
    std = numeric.groupby(level="date").transform("std").replace(0, np.nan)
    out = (numeric - mean) / std
    out = out.replace([np.inf, -np.inf], np.nan)
    if clip is not None:
        out = out.clip(-abs(float(clip)), abs(float(clip)))
    return out.dropna().sort_index()
