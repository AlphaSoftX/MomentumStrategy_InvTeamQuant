"""Core-N selection: fixed basket size within a supplied eligible universe."""
import numpy as np
import pandas as pd

def build_core_basket(score_row, universe_tickers, core_n, require_positive_score=True):
    if int(core_n) < 1:
        raise ValueError("core_n must be >= 1")
    candidates = score_row.reindex(pd.Index(universe_tickers)).dropna()
    if require_positive_score:
        candidates = candidates[candidates > 0]
    return candidates.sort_values(ascending=False, kind="stable").head(int(core_n)).index

def universe_quality_snapshot(score_row, universe_tickers, core_n):
    basket = build_core_basket(score_row, universe_tickers, core_n, require_positive_score=False)
    scores = score_row.reindex(basket)
    return {
        "n_selected": len(basket),
        "mean_score": float(scores.mean()) if len(scores) else np.nan,
        "min_score": float(scores.min()) if len(scores) else np.nan,
    }
