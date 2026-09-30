import numpy as np
import pandas as pd

def compute_metrics(equity_curve, periods_per_year=12):
    eq = pd.Series(equity_curve).dropna()
    if len(eq) < 2:
        return {k: np.nan for k in ["CAGR","AnnVol","Sharpe","Sortino","MaxDD","Calmar","WinRate","FinalEquity"]}
    rets = eq.pct_change(fill_method=None).dropna()
    n_years = len(rets) / periods_per_year
    final = float(eq.iloc[-1])
    cagr = final ** (1.0 / n_years) - 1.0 if n_years > 0 and final > 0 else np.nan
    ann_vol = rets.std(ddof=1) * np.sqrt(periods_per_year)
    sharpe = (rets.mean() * periods_per_year) / ann_vol if ann_vol > 0 else np.nan
    downside = rets[rets < 0]
    downside_vol = downside.std(ddof=1) * np.sqrt(periods_per_year) if len(downside) > 1 else np.nan
    sortino = (rets.mean() * periods_per_year) / downside_vol if downside_vol > 0 else np.nan
    drawdown = eq / eq.cummax() - 1.0
    max_dd = float(drawdown.min())
    calmar = cagr / abs(max_dd) if np.isfinite(cagr) and max_dd < 0 else np.nan
    return {
        "CAGR": cagr, "AnnVol": ann_vol, "Sharpe": sharpe,
        "Sortino": sortino, "MaxDD": max_dd, "Calmar": calmar,
        "WinRate": float((rets > 0).mean()), "FinalEquity": final,
    }

def metrics_for_window(equity_curve, start=None, end=None):
    sub = pd.Series(equity_curve).loc[start:end].dropna()
    if len(sub) < 3:
        return {k: np.nan for k in ["CAGR","AnnVol","Sharpe","Sortino","MaxDD","Calmar","WinRate","FinalEquity"]}
    return compute_metrics(sub / sub.iloc[0])
