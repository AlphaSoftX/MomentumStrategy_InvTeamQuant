"""Optional fixed/dynamic multi-universe research layer using the common strategy engine."""
import numpy as np
import pandas as pd
from core_index import build_core_basket
from strategy import _make_weights, _turnover_full, _period_outcome, _full_weights, _drift

def _basket_engine(
    monthly_prices, basket_fn, rebalance_dates,
    tx_cost_bps=10.0, slippage_bps=5.0, dd_threshold=0.15,
    dd_deleverage=0.5, weighting="inv_vol", vol_window=6,
    enable_dd_overlay=False, missing_return_policy="liquidate",
):
    monthly_prices = monthly_prices.sort_index()
    rets = monthly_prices.pct_change(fill_method=None)
    reb_dates = [d for d in rebalance_dates if d in monthly_prices.index]
    if len(reb_dates) < 2:
        raise ValueError("Not enough rebalance dates")
    start = monthly_prices.index.get_loc(reb_dates[0])
    reb_set = set(reb_dates)

    weights = pd.Series(dtype=float)
    exposure = 1.0
    peak = 1.0
    equity = [1.0]
    dates_out = [monthly_prices.index[start]]
    prev_full = pd.Series({"__CASH__": 1.0})
    recs = []

    for i in range(start, len(monthly_prices.index) - 1):
        d, nxt = monthly_prices.index[i], monthly_prices.index[i + 1]
        if d in reb_set:
            weights = _make_weights(rets, d, basket_fn(d), weighting, vol_window)
        new_full = _full_weights(weights, exposure)
        turnover = _turnover_full(prev_full, new_full)
        cost = turnover * (tx_cost_bps + slippage_bps) / 10000.0
        equity[-1] *= max(0.0, 1.0 - cost)

        gross, missing, filled = _period_outcome(rets.loc[nxt], weights, missing_return_policy)
        new_eq = equity[-1] * (1.0 + exposure * gross)
        prev_full, drifted = _drift(weights, exposure, filled)  # uses the exposure actually held
        peak = max(peak, new_eq)
        dd = new_eq / peak - 1.0
        if enable_dd_overlay:
            if dd <= -abs(dd_threshold):
                exposure = float(dd_deleverage)
            elif new_eq >= peak:
                exposure = 1.0

        recs.append({"date": d, "turnover": turnover, "n_holdings": len(weights), "missing_returns": missing, "drawdown": dd})
        weights = drifted
        equity.append(new_eq)
        dates_out.append(nxt)

    return pd.Series(equity, index=dates_out, name="equity"), pd.DataFrame(recs).set_index("date")

def run_fixed_universe_core(
    monthly_prices, scores, membership, universe_name, core_n, k_months,
    require_positive_score=True, **engine_kwargs
):
    reb_dates = list(scores.index[::int(k_months)])
    def basket_fn(date):
        row = membership.loc[date, universe_name] if isinstance(membership, pd.DataFrame) else membership[universe_name].loc[date]
        tickers = row[row].index
        return build_core_basket(scores.loc[date], tickers, core_n, require_positive_score)
    return _basket_engine(monthly_prices, basket_fn, reb_dates, **engine_kwargs)

def run_dynamic_universe_switch(
    monthly_prices, scores, membership, universe_names, core_n, k_months,
    selection_lookback_months=12, require_positive_score=True, **engine_kwargs
):
    shadow = {}
    for name in universe_names:
        shadow[name], _ = run_fixed_universe_core(
            monthly_prices, scores, membership, name, core_n, k_months,
            require_positive_score, **engine_kwargs
        )
    shadow_df = pd.DataFrame(shadow)

    reb_dates = list(scores.index[::int(k_months)])
    choice = []
    def score(name, date):
        w = shadow_df[name].loc[:date].tail(selection_lookback_months + 1).pct_change(fill_method=None).dropna()
        if len(w) < 3 or w.std(ddof=1) == 0:
            return -np.inf
        return float(w.mean() / w.std(ddof=1) * np.sqrt(12.0))

    def basket_fn(date):
        ranked = [(score(name, date), name) for name in universe_names]
        best_score, best_name = max(ranked, key=lambda x: (x[0], x[1]))
        choice.append({"date": date, "chosen_universe": best_name, "trailing_sharpe": best_score})
        row = membership[best_name].loc[date] if isinstance(membership, dict) else membership.loc[date, best_name]
        return build_core_basket(scores.loc[date], row[row].index, core_n, require_positive_score)

    live, turnover = _basket_engine(monthly_prices, basket_fn, reb_dates, **engine_kwargs)
    return live, turnover, pd.DataFrame(choice), shadow_df
