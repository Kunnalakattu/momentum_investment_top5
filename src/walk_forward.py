"""
Walk-forward validation for the final strategy:
Momentum → Top-5 → 200DMA filter → HRP weighting.

The full backtest is already sequential (no lookahead). This module
slices the saved HRP return series into annual folds and reports
per-fold consistency metrics: CAGR, Sharpe, MaxDD, Calmar.
"""

import numpy as np
import pandas as pd

PERIODS = 12


def _fold_metrics(ret: pd.Series, rf: pd.Series) -> dict:
    """CAGR, Sharpe, MaxDD, Calmar for one fold."""
    ret = ret.dropna()
    if len(ret) < 2:
        return {"CAGR %": np.nan, "Sharpe": np.nan, "MaxDD %": np.nan, "Calmar": np.nan}

    rf_fold  = rf.reindex(ret.index).fillna(0)
    rf_mean  = float(rf_fold.mean())

    n_yr     = len(ret) / PERIODS
    cagr     = float((1 + ret).prod() ** (1 / n_yr) - 1) if n_yr > 0 else np.nan

    excess   = ret - rf_fold
    ann_vol  = float(ret.std() * np.sqrt(PERIODS))
    ann_rf   = rf_mean * PERIODS
    sharpe   = float((cagr - ann_rf) / ann_vol) if ann_vol > 0 else np.nan

    equity   = (1 + ret).cumprod()
    max_dd   = float((equity / equity.cummax() - 1).min())
    calmar   = float(cagr / abs(max_dd)) if max_dd < 0 else np.nan

    return {
        "CAGR %":  round(cagr  * 100, 2),
        "Sharpe":  round(sharpe, 3),
        "MaxDD %": round(max_dd * 100, 2),
        "Calmar":  round(calmar,  3),
    }


def run_walk_forward(
    hrp_returns: pd.Series,
    spy_returns: pd.Series,
    rf_monthly:  pd.Series,
) -> pd.DataFrame:
    """
    Slice the HRP return series into calendar-year folds.
    Returns a DataFrame with one row per fold + a 'Full Period' summary row.
    """
    hrp  = hrp_returns.dropna()
    spy  = spy_returns.reindex(hrp.index).dropna()
    years = sorted(hrp.index.year.unique())

    rows = []
    for yr in years:
        mask = hrp.index.year == yr
        h    = hrp[mask]
        s    = spy.reindex(h.index).dropna()
        rf   = rf_monthly.reindex(h.index).fillna(0)

        hm   = _fold_metrics(h, rf)
        sm   = _fold_metrics(s, rf)

        rows.append({
            "Fold":        str(yr),
            "N months":    len(h),
            "CAGR %":      hm["CAGR %"],
            "Sharpe":      hm["Sharpe"],
            "MaxDD %":     hm["MaxDD %"],
            "Calmar":      hm["Calmar"],
            "SPY CAGR %":  sm["CAGR %"],
            "SPY Sharpe":  sm["Sharpe"],
            "Active CAGR": round(hm["CAGR %"] - sm["CAGR %"], 2) if not np.isnan(hm["CAGR %"]) and not np.isnan(sm["CAGR %"]) else np.nan,
        })

    # Full period summary row
    hm_full  = _fold_metrics(hrp, rf_monthly)
    spy_full = _fold_metrics(spy, rf_monthly.reindex(spy.index).fillna(0))
    rows.append({
        "Fold":        "Full Period",
        "N months":    len(hrp),
        "CAGR %":      hm_full["CAGR %"],
        "Sharpe":      hm_full["Sharpe"],
        "MaxDD %":     hm_full["MaxDD %"],
        "Calmar":      hm_full["Calmar"],
        "SPY CAGR %":  spy_full["CAGR %"],
        "SPY Sharpe":  spy_full["Sharpe"],
        "Active CAGR": round(hm_full["CAGR %"] - spy_full["CAGR %"], 2),
    })

    df = pd.DataFrame(rows).set_index("Fold")
    return df


def print_walk_forward_table(df: pd.DataFrame) -> None:
    folds = df[df.index != "Full Period"]
    n_pos = (folds["CAGR %"]  > 0).sum()
    n_sh  = (folds["Sharpe"]  > 0).sum()
    n_beat = (folds["Active CAGR"] > 0).sum()
    n_tot = len(folds)

    print(f"\n{'='*85}")
    print(f"  WALK-FORWARD VALIDATION — Momentum → Top-5 → 200DMA → HRP")
    print(f"{'='*85}")
    print(f"  {'Fold':<12} {'N':>3}  {'CAGR%':>7}  {'Sharpe':>7}  {'MaxDD%':>7}  {'Calmar':>7}  "
          f"{'SPY%':>6}  {'Alpha':>7}")
    print(f"  {'─'*12} {'─'*3}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*6}  {'─'*7}")

    for fold, row in df.iterrows():
        is_full = fold == "Full Period"
        mark = ""
        if not is_full:
            mark = " ★" if row["Active CAGR"] > 0 else ""
        sep = "  " if is_full else "  "
        print(
            f"  {fold:<12} {int(row['N months']):>3}  "
            f"{row['CAGR %']:>7.2f}  "
            f"{row['Sharpe']:>7.3f}  "
            f"{row['MaxDD %']:>7.2f}  "
            f"{row['Calmar']:>7.3f}  "
            f"{row['SPY CAGR %']:>6.2f}  "
            f"{row['Active CAGR']:>7.2f}"
            f"{mark}"
            + ("  ← full" if is_full else "")
        )
        if is_full:
            print(f"  {'─'*12} {'─'*3}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*6}  {'─'*7}")

    print(f"\n  Consistency check ({n_tot} annual folds, excl. partial years):")
    print(f"    CAGR > 0       : {n_pos}/{n_tot}  ({n_pos/n_tot*100:.0f}%)")
    print(f"    Sharpe > 0     : {n_sh}/{n_tot}  ({n_sh/n_tot*100:.0f}%)")
    print(f"    Beat SPY       : {n_beat}/{n_tot}  ({n_beat/n_tot*100:.0f}%)")

    sh_vals = folds["Sharpe"].dropna()
    print(f"    Sharpe median  : {sh_vals.median():.3f}  (range [{sh_vals.min():.3f}, {sh_vals.max():.3f}])")
    cagr_vals = folds["CAGR %"].dropna()
    print(f"    CAGR  median   : {cagr_vals.median():.2f}%  (range [{cagr_vals.min():.2f}%, {cagr_vals.max():.2f}%])")
    print(f"{'='*85}")


def run_walk_forward_pipeline(
    prices:   pd.DataFrame,
    returns:  pd.DataFrame,
    proc_dir: str = "data/processed",
) -> pd.DataFrame:
    from src.backtest import load_backtest_returns
    from src.data import load_risk_free_rate

    rf_daily   = load_risk_free_rate(proc_dir)
    rf_monthly = (1 + rf_daily).resample("ME").prod() - 1

    bt_rets   = load_backtest_returns(proc_dir)
    hrp_ret   = bt_rets["D: HRP"].dropna()
    spy_ret   = prices.resample("ME").last()["SPY"].pct_change().dropna()

    df = run_walk_forward(hrp_ret, spy_ret, rf_monthly)
    print_walk_forward_table(df)
    return df
