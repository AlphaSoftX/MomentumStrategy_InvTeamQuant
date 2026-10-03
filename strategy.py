"""Monthly portfolio engine with weight-based turnover and strict missing-data handling."""
from __future__ import annotations
import numpy as np
import pandas as pd

def _make_weights(monthly_rets, date, selected, weighting, vol_window):
    selected = pd.Index(selected).drop_duplicates()
    if len(selected) == 0:
        return pd.Series(dtype=float)
    if weighting == "equal":
        return pd.Series(1.0 / len(selected), index=selected, dtype=float)
    if weighting != "inv_vol":
        raise ValueError("weighting must be 'equal' or 'inv_vol'")
    vols = monthly_rets.loc[:date, selected].tail(vol_window).std(ddof=1)
    inv = 1.0 / vols.replace(0, np.nan)
    finite = inv[np.isfinite(inv)]
    if finite.empty or finite.sum() <= 0:
        return pd.Series(1.0 / len(selected), index=selected, dtype=float)
    inv = inv.fillna(finite.mean())
    total = inv.sum()
    return inv / total if total > 0 else pd.Series(1.0 / len(selected), index=selected, dtype=float)

def _turnover_full(prev_full, new_full):
    """One-way turnover 0.5*sum|w_new - w_prev| on FULL portfolio weights (risky names + '__CASH__')."""
    names = prev_full.index.union(new_full.index)
    diff = new_full.reindex(names, fill_value=0.0) - prev_full.reindex(names, fill_value=0.0)
    return 0.5 * float(diff.abs().sum())

def _period_outcome(rets, weights, policy):
    """Returns (gross_return, n_missing, filled_returns) for a basket held over one period."""
    valid = rets.reindex(weights.index)
    missing = valid.isna()
    n_missing = int(missing.sum())
    if n_missing:
        if policy == "raise":
            raise ValueError(f"{n_missing} held positions have missing next-period returns")
        if policy == "liquidate":
            valid = valid.fillna(-1.0)
        elif policy == "zero":
            valid = valid.fillna(0.0)
        else:
            raise ValueError("missing_return_policy must be 'liquidate', 'raise', or 'zero'")
    gross = float((weights * valid).sum())
    return gross, n_missing, valid

def _period_return(rets, weights, policy):
    gross, n_missing, _ = _period_outcome(rets, weights, policy)
    return gross, n_missing

def _full_weights(weights, exposure):
    """Basket weights scaled by exposure, plus the cash remainder."""
    risky = weights * float(exposure)
    out = risky.copy()
    out["__CASH__"] = 1.0 - float(risky.sum())
    return out

def _drift(weights, exposure, filled_rets):
    """
    Buy-and-hold drift over one period. Returns (prev_full, next_basket):
      prev_full   - full portfolio weights (incl. cash) at the END of the period, before any trading
      next_basket - basket weights renormalised to sum to 1 (what we hold if we do not rebalance)
    """
    risky_end = weights * float(exposure) * (1.0 + filled_rets.reindex(weights.index))
    cash_end = 1.0 - float((weights * float(exposure)).sum())
    total = float(risky_end.sum()) + cash_end
    if total <= 0:
        return pd.Series({"__CASH__": 1.0}), pd.Series(dtype=float)
    prev_full = risky_end / total
    prev_full["__CASH__"] = cash_end / total
    basket_total = float(risky_end.sum())
    next_basket = risky_end / basket_total if basket_total > 0 else pd.Series(dtype=float)
    return prev_full, next_basket

def run_strategy(
    monthly_prices,
    scores,
    n_pct,
    k_months,
    tx_cost_bps=10.0,
    slippage_bps=5.0,
    dd_threshold=0.15,
    dd_deleverage=0.5,
    require_positive_score=True,
    weighting="inv_vol",
    vol_window=6,
    enable_dd_overlay=False,
    missing_return_policy="liquidate",
    eligible=None,
):
    if not 0 < float(n_pct) <= 1:
        raise ValueError("n_pct must be in (0, 1]")
    if int(k_months) < 1:
        raise ValueError("k_months must be >= 1")
    monthly_prices = monthly_prices.sort_index()
    scores = scores.sort_index()
    monthly_rets = monthly_prices.pct_change(fill_method=None)
    all_months = monthly_prices.index
    reb_dates = [d for d in scores.index[::k_months] if d in all_months]
    if len(reb_dates) < 2:
        raise ValueError("Not enough rebalance dates")

    start_idx = all_months.get_loc(reb_dates[0])
    reb_set = set(reb_dates)
    weights = pd.Series(dtype=float)
    exposure = 1.0
    peak = 1.0
    equity = [1.0]
    dates_out = [all_months[start_idx]]
    prev_full = pd.Series({"__CASH__": 1.0})
    records = []

    for idx in range(start_idx, len(all_months) - 1):
        this_date = all_months[idx]
        next_date = all_months[idx + 1]
        rebalanced = this_date in reb_set

        if rebalanced:
            raw = scores.loc[this_date].dropna()
            if eligible is not None and this_date in eligible.index:
                elig = eligible.loc[this_date]
                raw = raw[elig.reindex(raw.index, fill_value=False)]
            candidates = raw[raw > 0] if require_positive_score else raw
            n_select = max(1, int(np.floor(len(raw) * float(n_pct))))
            selected = candidates.sort_values(ascending=False).head(n_select).index
            weights = _make_weights(monthly_rets, this_date, selected, weighting, vol_window)

        new_full = _full_weights(weights, exposure)
        turnover = _turnover_full(prev_full, new_full)
        cost = turnover * (float(tx_cost_bps) + float(slippage_bps)) / 10000.0
        equity[-1] *= max(0.0, 1.0 - cost)

        gross_ret, n_missing, filled = _period_outcome(monthly_rets.loc[next_date], weights, missing_return_policy)
        port_ret = exposure * gross_ret
        new_eq = equity[-1] * (1.0 + port_ret)
        # weights AFTER price moves, using the exposure that was actually held this period
        prev_full, drifted_basket = _drift(weights, exposure, filled)

        peak = max(peak, new_eq)
        dd = new_eq / peak - 1.0
        old_exposure = exposure
        if enable_dd_overlay:
            if dd <= -abs(float(dd_threshold)):
                exposure = float(dd_deleverage)
            elif new_eq >= peak:
                exposure = 1.0

        records.append({
            "date": this_date,
            "turnover": turnover,
            "cost": cost,
            "n_holdings": len(weights),
            "missing_returns": n_missing,
            "exposure_used": old_exposure,
            "drawdown": dd,
            "rebalanced": rebalanced,
        })
        weights = drifted_basket  # buy-and-hold until the next rebalance date
        equity.append(new_eq)
        dates_out.append(next_date)

    curve = pd.Series(equity, index=pd.DatetimeIndex(dates_out), name="equity")
    turnover_df = pd.DataFrame(records).set_index("date")
    return curve, turnover_df
