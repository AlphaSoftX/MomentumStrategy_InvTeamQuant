# RESEARCH/DEMO ONLY — NOT USED BY THE PRODUCTION PIPELINE.
# Its synthetic membership is not suitable for unbiased historical performance claims.

"""
universes.py
------------
Defines multiple candidate "index" universes (e.g. NIFTY 100 / 200 / 500)
as TIME-VARYING sets of constituent tickers, so the multi-universe ranking
layer in core_index.py / universe_strategy.py can be tested without
introducing look-ahead or survivorship bias.

HOW MEMBERSHIP IS BUILT HERE (demo): a slowly time-varying synthetic
"size" proxy per stock is used to reconstitute each universe every
`reconstitution_every` months — using only size data available as of that
reconstitution date. This mirrors how real indices work (periodic
reconstitution using only-then-available data) without needing real market
cap history.

REPLACE FOR PRODUCTION: swap `build_synthetic_size_series` +
`build_universe_membership` for actual historical NIFTY 100/200/500
constituent membership per rebalance date (e.g. from NSE index factsheets
or a vendor feed). Using TODAY's constituents applied backwards to history
would introduce survivorship bias — whatever real data source you use, it
must tell you who was IN each index AS OF each past date, not who is in it
now.
"""

import numpy as np
import pandas as pd

# name -> constituent count. Add/remove candidate universes here.
UNIVERSE_SIZES = {
    "NIFTY_100": 100,
    "NIFTY_200": 200,
    "NIFTY_500": 500,
}


def build_synthetic_size_series(prices, seed=99):
    """
    Synthetic, slowly mean-reverting "size" proxy per stock per month —
    stand-in for market capitalization, used only to assign universe
    membership in this demo. NOT real market cap data.
    """
    monthly_dates = prices.resample("ME").last().index
    tickers = prices.columns
    rng = np.random.default_rng(seed)

    n_t, n_s = len(monthly_dates), len(tickers)
    base_size = rng.normal(0, 1, n_s)
    size = np.zeros((n_t, n_s))
    size[0] = base_size
    phi = 0.98
    noise = rng.normal(0, 0.08, (n_t, n_s))
    for t in range(1, n_t):
        size[t] = phi * size[t - 1] + (1 - phi) * base_size + noise[t]

    return pd.DataFrame(size, index=monthly_dates, columns=tickers)


def build_universe_membership(size_panel, universe_sizes=None, reconstitution_every=6):
    """
    Reconstitutes each named universe every `reconstitution_every` months,
    using only size data as of the reconstitution date — no lookahead. The
    top-K stocks by size become that universe's members until the next
    reconstitution date.

    Parameters
    ----------
    size_panel : DataFrame (date x ticker), e.g. from build_synthetic_size_series
                 or your real historical market-cap panel.
    universe_sizes : dict[name] -> constituent count. Defaults to UNIVERSE_SIZES.
    reconstitution_every : int, months between reconstitutions.

    Returns
    -------
    dict[universe_name] -> DataFrame (date x ticker) of booleans.
    """
    universe_sizes = universe_sizes or UNIVERSE_SIZES
    dates = size_panel.index
    reco_positions = list(range(0, len(dates), reconstitution_every))

    membership = {name: pd.DataFrame(False, index=dates, columns=size_panel.columns)
                  for name in universe_sizes}

    for i, pos in enumerate(reco_positions):
        start_date = dates[pos]
        end_pos = reco_positions[i + 1] if i + 1 < len(reco_positions) else len(dates)
        end_date = dates[end_pos - 1]
        for name, k in universe_sizes.items():
            k_eff = min(k, size_panel.shape[1])
            top_k = size_panel.loc[start_date].sort_values(ascending=False).head(k_eff).index
            membership[name].loc[start_date:end_date, top_k] = True

    return membership


def get_universe_tickers(membership, universe_name, date):
    """Convenience accessor: constituent tickers of `universe_name` as of `date`."""
    row = membership[universe_name].loc[date]
    return row[row].index


if __name__ == "__main__":
    import data_loader
    prices = data_loader.load_price_data(n_stocks=500, seed=11)
    size_panel = build_synthetic_size_series(prices)
    membership = build_universe_membership(size_panel)
    for name in UNIVERSE_SIZES:
        counts = membership[name].sum(axis=1)
        print(name, "avg members:", counts.mean())
