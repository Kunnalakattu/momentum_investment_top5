"""
Walk-forward backtest: four weighting methods × two vol-targeting modes.

Strategies tested (all use Top-5 + 200DMA filter, 10bp one-way cost):
  A  : Equal weight
  B  : Inverse volatility
  C  : Risk parity (ERC)
  D  : HRP
  A+ : Equal weight + 10% vol target
  B+ : Inverse vol  + 10% vol target
  C+ : Risk parity  + 10% vol target
  D+ : HRP          + 10% vol target

Benchmark: VUSA buy-and-hold (S&P 500 UCITS).
"""

import numpy as np
import pandas as pd
import os
from pathlib import Path

from src.hypothesis_tests import (
    _hrp_weight, _inv_vol_weight, _risk_parity_weight,
    compute_me_forward_returns, LOOKBACK_MOS,
)
from src.portfolio import build_weight_matrix

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
VOL_TARGET    = 0.10    # 10% annualised
LEVERAGE_CAP  = 1.5     # max scale factor
COST_ONE_WAY  = 0.001   # 10 bp
N_TOP         = 5
PERIODS       = 12      # monthly


# ─────────────────────────────────────────────────────────────────────────────
# Extended metrics
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(ret: pd.Series, rf_monthly: pd.Series | None = None) -> dict:
    ret = ret.dropna()
    if ret.empty:
        return {}

    if rf_monthly is not None:
        rf = rf_monthly.reindex(ret.index).fillna(0)
    else:
        rf = pd.Series(0.0, index=ret.index)

    excess     = ret - rf
    ann_ret    = ret.mean()    * PERIODS
    ann_vol    = ret.std()     * np.sqrt(PERIODS)
    ann_rf     = rf.mean()     * PERIODS
    sharpe     = (ann_ret - ann_rf) / ann_vol if ann_vol > 0 else np.nan

    down       = ret[ret < rf]
    down_std   = down.std() * np.sqrt(PERIODS) if len(down) > 1 else np.nan
    sortino    = (ann_ret - ann_rf) / down_std if (down_std and down_std > 0) else np.nan

    equity     = (1 + ret).cumprod()
    drawdown   = equity / equity.cummax() - 1
    max_dd     = float(drawdown.min())
    calmar     = ann_ret / abs(max_dd) if max_dd < 0 else np.nan

    # Recovery: longest stretch to recover from a drawdown
    in_dd = drawdown < 0
    max_recovery = 0
    streak = 0
    for v in in_dd:
        streak = streak + 1 if v else 0
        max_recovery = max(max_recovery, streak)

    var_95     = float(np.percentile(ret, 5))
    cvar_95    = float(ret[ret <= var_95].mean()) if (ret <= var_95).any() else np.nan

    return {
        "Ann Return %":  round(ann_ret  * 100, 2),
        "Ann Vol %":     round(ann_vol  * 100, 2),
        "Sharpe":        round(float(sharpe),  3),
        "Sortino":       round(float(sortino), 3) if not np.isnan(sortino) else np.nan,
        "Max DD %":      round(max_dd   * 100, 2),
        "Calmar":        round(float(calmar),  3) if not np.isnan(calmar)  else np.nan,
        "VaR 95% %":    round(var_95   * 100, 2),
        "CVaR 95% %":   round(cvar_95  * 100, 2) if not np.isnan(cvar_95) else np.nan,
        "Hit Rate %":    round((ret > 0).mean() * 100, 1),
        "Best Month %":  round(ret.max() * 100, 2),
        "Worst Month %": round(ret.min() * 100, 2),
        "Max DD Dur":    int(max_recovery),
        "N months":      len(ret),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Volatility-targeting backtest engine
# ─────────────────────────────────────────────────────────────────────────────
def _run_backtest_vt(
    weights_df:   pd.DataFrame,
    fwd_returns:  pd.DataFrame,
    me_returns:   pd.DataFrame,
    rf_monthly:   pd.Series,
    cost_one_way: float = COST_ONE_WAY,
    vol_target:   float | None = None,
    leverage_cap: float = LEVERAGE_CAP,
    lookback:     int   = LOOKBACK_MOS,
) -> pd.Series:
    """
    Backtest engine with optional volatility targeting.
    Cash portion (when scale < 1) earns the risk-free rate.
    """
    dates  = weights_df.index.intersection(fwd_returns.index)
    prev_w = pd.Series(0.0, index=weights_df.columns)
    out    = {}

    for dt in dates:
        w   = weights_df.loc[dt].fillna(0.0)
        fwd = fwd_returns.loc[dt].reindex(w.index).fillna(0.0)
        rf  = float(rf_monthly.get(dt, 0.0))

        scale = 1.0
        if vol_target is not None and w.sum() > 0:
            active = w[w > 0].index.tolist()
            hist   = me_returns[me_returns.index < dt].tail(lookback)
            if active and len(hist) >= 12:
                sub_cov  = hist[active].cov()
                port_var = float(w[active] @ sub_cov.loc[active, active].values @ w[active])
                port_vol = np.sqrt(max(port_var * PERIODS, 0.0))
                if port_vol > 1e-6:
                    scale = min(vol_target / port_vol, leverage_cap)

        w_scaled = w * scale
        cash_ret = (1.0 - w_scaled.sum()) * rf    # uninvested portion earns rf

        turnover = (w_scaled - prev_w).abs().sum() / 2.0
        port_ret = float(w_scaled @ fwd) + cash_ret - turnover * cost_one_way
        out[dt]  = port_ret

        drifted = w_scaled * (1 + fwd)
        total   = drifted.sum()
        prev_w  = drifted / total if total > 0 else w_scaled

    return pd.Series(out)


# ─────────────────────────────────────────────────────────────────────────────
# Run all eight strategy variants
# ─────────────────────────────────────────────────────────────────────────────
def run_all_backtests(
    signals:    dict,
    prices:     pd.DataFrame,
    n_top:      int   = N_TOP,
    cost_bps:   float = 10.0,
    vol_target: float = VOL_TARGET,
) -> dict:
    """
    Returns dict: strategy_label → {"returns": Series, "metrics": dict, "weights": DataFrame}.
    """
    me_prices   = prices.resample("ME").last()
    me_returns  = me_prices.pct_change()
    fwd_returns = compute_me_forward_returns(prices)
    cost        = cost_bps / 10_000

    # Monthly risk-free rate from signal rf (resampled)
    from src.data import load_risk_free_rate
    rf_daily   = load_risk_free_rate()
    rf_monthly = (1 + rf_daily).resample("ME").prod() - 1

    # VUSA benchmark (S&P 500 UCITS, buy-and-hold)
    bench_col = "VUSA" if "VUSA" in me_returns.columns else me_returns.columns[0]
    spy_ret = me_returns[bench_col].dropna().rename("VUSA B&H")

    results = {"VUSA B&H": {"returns": spy_ret, "metrics": compute_metrics(spy_ret, rf_monthly),
                             "weights": None}}

    methods = ["equal", "inv_vol", "risk_parity", "hrp"]
    labels  = {"equal": "A: Equal", "inv_vol": "B: Inv Vol",
                "risk_parity": "C: Risk Parity", "hrp": "D: HRP"}

    for method in methods:
        label = labels[method]
        print(f"  Building weights: {label} …", end="\r")
        w_df = build_weight_matrix(signals, me_returns, n_top=n_top, method=method)

        # Without vol targeting
        ret = _run_backtest_vt(
            w_df, fwd_returns, me_returns, rf_monthly,
            cost_one_way=cost, vol_target=None,
        ).dropna()
        results[label] = {"returns": ret, "metrics": compute_metrics(ret, rf_monthly), "weights": w_df}

        # With vol targeting
        label_vt = label + " + VT"
        ret_vt = _run_backtest_vt(
            w_df, fwd_returns, me_returns, rf_monthly,
            cost_one_way=cost, vol_target=vol_target,
        ).dropna()
        results[label_vt] = {"returns": ret_vt, "metrics": compute_metrics(ret_vt, rf_monthly), "weights": w_df}

    print("  All strategies computed.                  ")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Print summary table
# ─────────────────────────────────────────────────────────────────────────────
def print_summary(results: dict) -> pd.DataFrame:
    key_metrics = ["Ann Return %", "Ann Vol %", "Sharpe", "Sortino",
                   "Max DD %", "Calmar", "Hit Rate %"]
    m_df = pd.DataFrame(
        {k: {m: v["metrics"].get(m, np.nan) for m in key_metrics}
         for k, v in results.items()}
    ).T
    m_df.index.name = "Strategy"

    print(f"\n{'='*90}")
    print(f"  BACKTEST SUMMARY  (Top-{N_TOP}, 200DMA filter, 10bp cost, {VOL_TARGET*100:.0f}% vol target where applied)")
    print(f"{'='*90}")
    print(m_df.to_string())
    return m_df


# ─────────────────────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────────────────────
def save_backtest_results(results: dict, directory: str = "data/processed") -> None:
    Path(directory).mkdir(parents=True, exist_ok=True)
    frames = {k: v["returns"] for k, v in results.items()}
    pd.DataFrame(frames).to_parquet(os.path.join(directory, "backtest_returns.parquet"))
    print(f"Saved backtest returns → {directory}/backtest_returns.parquet")


def load_backtest_returns(directory: str = "data/processed") -> pd.DataFrame:
    return pd.read_parquet(os.path.join(directory, "backtest_returns.parquet"))


# ─────────────────────────────────────────────────────────────────────────────
# Top-level pipeline
# ─────────────────────────────────────────────────────────────────────────────
def run_backtest_pipeline(
    signals: dict,
    prices:  pd.DataFrame,
    returns: pd.DataFrame,
    proc_dir: str = "data/processed",
) -> dict:
    print("\n" + "=" * 60)
    print("  RUNNING WALK-FORWARD BACKTEST — all 8 strategies")
    print("=" * 60)

    results = run_all_backtests(signals, prices)
    summary = print_summary(results)
    save_backtest_results(results, proc_dir)

    return results
