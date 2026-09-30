"""Simple j-month momentum score used only in the research sweep."""
import numpy as np
import pandas as pd

def compute_momentum_scores(prices, j, skip_recent=1):
    if int(j) <= 0:
        raise ValueError("j must be >= 1")
    monthly = prices.resample("ME").last()
    log_prices = np.log(monthly.where(monthly > 0))
    score = log_prices.shift(skip_recent) - log_prices.shift(int(j) + skip_recent)
    return score.dropna(how="all")
