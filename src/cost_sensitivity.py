"""
Phase 15 — Transaction Cost Sensitivity

Sweeps one-way transaction cost from 5 to 100 bps.
For each cost level, computes net CAGR, Sharpe, MaxDD, Calmar.
Answers: "At what cost does the strategy stop being worth running?"

The cost is modelled as a monthly drag:
  net_return_month = gross_return - (annual_turnover/12 × 2 × cost_bps / 1e4)
"""

import numpy as np
import pandas as pd

PERIODS   = 12
COST_LEVELS_BPS = [0, 5, 10, 20, 30, 50, 75, 100]


def _metrics(ret: pd.Series, rf: pd.Series) -> dict:
    ret = ret.dropna()
    if len(ret) < 2:
        return {k: np.nan for k in ["CAGR %", "Sharpe", "MaxDD %", "Calmar", "Ann Vol %"]}
    n_yr    = len(ret) / PERIODS
    cagr    = float((1 + ret).prod() ** (1 / n_yr) - 1)
    ann_vol = float(ret.std() * np.sqrt(PERIODS))
    rf_mean = float(rf.reindex(ret.index).fillna(0).mean() * PERIODS)
    sharpe  = (cagr - rf_mean) / ann_vol if ann_vol > 0 else np.nan
    eq      = (1 + ret).cumprod()
    maxdd   = float((eq / eq.cummax() - 1).min())
    calmar  = cagr / abs(maxdd) if maxdd < 0 else np.nan
    return {
        "CAGR %":    round(cagr * 100, 2),
        "Ann Vol %": round(ann_vol * 100, 2),
        "Sharpe":    round(sharpe, 3) if not np.isnan(sharpe) else np.nan,
        "MaxDD %":   round(maxdd * 100, 2),
        "Calmar":    round(calmar, 3) if not np.isnan(calmar) else np.nan,
    }


def apply_cost(
    gross_ret:         pd.Series,
    annual_turnover:   float,
    one_way_cost_bps:  float,
) -> pd.Series:
    """
    Subtract monthly cost drag from gross returns.
    annual_turnover: fraction of portfolio turned over per year.
    Round-trip = 2 × one-way; distributed evenly each month.
    """
    monthly_cost = annual_turnover * 2 * one_way_cost_bps / 1e4 / PERIODS
    return gross_ret - monthly_cost


def compute_turnover(weights_df: pd.DataFrame) -> float:
    """Average annual ONE-WAY turnover (fraction of portfolio).
    sum(|Δw|) counts both buys and sells, so divide by 2 for one-way.
    """
    wt_changes = weights_df.diff().abs().sum(axis=1)
    monthly_to_oneway = float(wt_changes.mean()) / 2
    return monthly_to_oneway * PERIODS


def run_cost_sensitivity(
    hrp_ret:        pd.Series,
    spy_ret:        pd.Series,
    rf_monthly:     pd.Series,
    annual_turnover: float,
    cost_levels:    list[float] = COST_LEVELS_BPS,
) -> pd.DataFrame:
    """
    For each cost level: metrics for the net-of-cost strategy.
    Returns DataFrame indexed by cost level.
    """
    rows = []
    for cost_bps in cost_levels:
        net = apply_cost(hrp_ret, annual_turnover, cost_bps)
        m   = _metrics(net, rf_monthly)
        spy_m = _metrics(spy_ret.reindex(hrp_ret.index).fillna(0), rf_monthly)
        rows.append({
            "Cost (bps)":     cost_bps,
            "CAGR %":         m["CAGR %"],
            "Ann Vol %":      m["Ann Vol %"],
            "Sharpe":         m["Sharpe"],
            "MaxDD %":        m["MaxDD %"],
            "Calmar":         m["Calmar"],
            "vs SPY CAGR":    round(m["CAGR %"] - spy_m["CAGR %"], 2),
            "Viable":         (
                "✓ YES" if m["Sharpe"] > 0.7 and m["CAGR %"] > 0
                else "~ OK" if m["Sharpe"] > 0.4 and m["CAGR %"] > 0
                else "✗ NO"
            ),
        })
    return pd.DataFrame(rows).set_index("Cost (bps)")


def print_cost_sensitivity(df: pd.DataFrame, annual_turnover: float) -> None:
    print(f"\n{'='*75}")
    print(f"  TRANSACTION COST SENSITIVITY — Momentum → Top-5 → 200DMA → HRP")
    print(f"  Annual turnover: {annual_turnover*100:.0f}%  (one-way)")
    print(f"{'='*75}")
    print(f"  {'Cost':>9}  {'CAGR%':>7}  {'Sharpe':>7}  {'MaxDD%':>7}  "
          f"{'Calmar':>7}  {'vs SPY':>7}  Viable")
    print(f"  {'─'*9}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*8}")
    for cost, row in df.iterrows():
        star = "  ← gross (0 cost)" if cost == 0 else ""
        print(f"  {cost:>6.0f}bps  "
              f"{row['CAGR %']:>+6.2f}%  "
              f"{row['Sharpe']:>7.3f}  "
              f"{row['MaxDD %']:>+6.2f}%  "
              f"{row['Calmar']:>7.3f}  "
              f"{row['vs SPY CAGR']:>+6.2f}%  "
              f"{row['Viable']}{star}")

    # Find break-even cost
    positive = df[df["CAGR %"] > 0]
    if len(positive) > 0:
        max_viable_cost = positive.index.max()
        print(f"\n  Strategy still CAGR-positive up to {max_viable_cost}bps one-way cost")
    sharpe_ok = df[df["Sharpe"] >= 0.5]
    if len(sharpe_ok) > 0:
        max_sh_cost = sharpe_ok.index.max()
        print(f"  Sharpe >= 0.5 up to {max_sh_cost}bps one-way cost")
    print(f"{'='*75}")


def run_cost_sensitivity_pipeline(
    prices:   pd.DataFrame,
    returns:  pd.DataFrame,
    proc_dir: str = "data/processed",
) -> dict:
    from src.backtest import load_backtest_returns
    from src.data import load_risk_free_rate
    from src.signals import load_signals
    from src.portfolio import build_weight_matrix

    rf_daily   = load_risk_free_rate(proc_dir)
    rf_monthly = (1 + rf_daily).resample("ME").prod() - 1

    bt_rets = load_backtest_returns(proc_dir)
    hrp_ret = bt_rets["D: HRP"].dropna()
    bench_col = "SPY" if "SPY" in prices.columns else prices.columns[0]
    spy_ret = prices.resample("ME").last()[bench_col].pct_change().dropna()

    signals    = load_signals(proc_dir)
    me_returns = prices.resample("ME").last().pct_change().dropna()
    weights_df = build_weight_matrix(signals, me_returns, n_top=5, method="hrp")
    annual_to  = compute_turnover(weights_df)

    df = run_cost_sensitivity(hrp_ret, spy_ret, rf_monthly, annual_to)
    print_cost_sensitivity(df, annual_to)

    # Net returns at each cost level (for notebook)
    net_series = {
        cost: apply_cost(hrp_ret, annual_to, cost)
        for cost in COST_LEVELS_BPS
    }

    return {
        "cost_df":       df,
        "net_series":    net_series,
        "hrp_ret":       hrp_ret,
        "spy_ret":       spy_ret,
        "rf_monthly":    rf_monthly,
        "annual_turnover": annual_to,
    }
