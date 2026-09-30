"""Research-only Phase 1: efficient, out-of-sample selection of j/k/n on development data."""
import numpy as np
import pandas as pd
import model_momentum
import strategy
import metrics

def _score_cache(prices, j_grid, eligibility=None):
    monthly = prices.resample("ME").last()
    cache = {}
    for j in j_grid:
        s = model_momentum.compute_momentum_scores(prices, j)
        if eligibility is not None:
            s = s.where(eligibility.reindex(index=s.index, columns=s.columns, fill_value=False))
        cache[j] = s
    return monthly, cache

def sweep(
    prices, j_grid, k_grid, n_grid, split_frac=0.70,
    eligibility=None, strategy_kwargs=None,
):
    strategy_kwargs = dict(strategy_kwargs or {})
    monthly, score_cache = _score_cache(prices, j_grid, eligibility)
    split_date = monthly.index[int(len(monthly) * split_frac)]
    rows = []
    for j in j_grid:
        for k in k_grid:
            for n in n_grid:
                try:
                    eq, turn = strategy.run_strategy(monthly, score_cache[j], n, k, eligible=eligibility, **strategy_kwargs)
                except ValueError:
                    continue
                train = metrics.metrics_for_window(eq, end=split_date)
                test = metrics.metrics_for_window(eq, start=split_date)
                rows.append({
                    "j": j, "k": k, "n": n,
                    "train_sharpe": train["Sharpe"], "test_sharpe": test["Sharpe"],
                    "train_calmar": train["Calmar"], "test_calmar": test["Calmar"],
                    "train_maxdd": train["MaxDD"], "test_maxdd": test["MaxDD"],
                    "avg_turnover": turn["turnover"].mean() if len(turn) else np.nan,
                })
    return pd.DataFrame(rows), split_date

def walk_forward(
    prices, j_grid, k_grid, n_grid, train_months=60, test_months=12,
    eligibility=None, strategy_kwargs=None,
):
    strategy_kwargs = dict(strategy_kwargs or {})
    monthly, score_cache = _score_cache(prices, j_grid, eligibility)
    all_months = monthly.index
    combos = {}
    for j in j_grid:
        for k in k_grid:
            for n in n_grid:
                try:
                    combos[(j, k, n)] = strategy.run_strategy(
                        monthly, score_cache[j], n, k, eligible=eligibility, **strategy_kwargs
                    )[0]
                except ValueError:
                    pass

    segments, chosen = [], []
    start = int(train_months)
    while start + test_months <= len(all_months):
        train_end, test_start = all_months[start - 1], all_months[start]
        test_end = all_months[start + test_months - 1]
        best_combo, best_sharpe = None, -np.inf
        for combo, eq in combos.items():
            sh = metrics.metrics_for_window(eq, end=train_end)["Sharpe"]
            if np.isfinite(sh) and sh > best_sharpe:
                best_combo, best_sharpe = combo, sh
        if best_combo is not None:
            seg = combos[best_combo].loc[test_start:test_end]
            if len(seg) >= 2:
                seg = seg / seg.iloc[0]
                segments.append(seg)
                chosen.append({
                    "test_start": test_start, "test_end": test_end,
                    "j": best_combo[0], "k": best_combo[1], "n": best_combo[2],
                    "train_sharpe": best_sharpe,
                })
        start += test_months

    if not segments:
        return None, pd.DataFrame(chosen)
    stitched, level = [segments[0]], float(segments[0].iloc[-1])
    for seg in segments[1:]:
        scaled = seg * level
        stitched.append(scaled.iloc[1:])
        level = float(scaled.iloc[-1])
    return pd.concat(stitched), pd.DataFrame(chosen)

def generate_feedback(sweep_df, top_k=8, sharpe_gap_warn=0.5):
    if sweep_df.empty:
        raise ValueError("sweep returned no valid parameter sets")
    df = sweep_df.copy()
    df["sharpe_gap"] = df["train_sharpe"] - df["test_sharpe"]
    df["overfit_flag"] = df["sharpe_gap"] > sharpe_gap_warn
    return df.sort_values("test_sharpe", ascending=False).head(top_k), df
