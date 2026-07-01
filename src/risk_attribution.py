"""
Risk Attribution — Momentum → Top-5 → 200DMA → HRP

Decomposes portfolio performance into per-asset contributions:
  1. Return contribution   — w_i * r_i  (Brinson-style)
  2. Risk contribution     — Component Risk Contribution (CRC)
                             CRC_i = w_i * (Σw)_i / σ_p  (sums to σ_p)
  3. Drawdown contribution — sum of return contributions during drawdown months
"""

import numpy as np
import pandas as pd

LOOKBACK_MOS = 36
PERIODS      = 12


# ─────────────────────────────────────────────────────────────────────────────
# 1. Return attribution
# ─────────────────────────────────────────────────────────────────────────────
def compute_return_attribution(
    weights_df:  pd.DataFrame,
    fwd_returns: pd.DataFrame,
) -> pd.DataFrame:
    """Monthly return contribution per asset: w_i * r_i."""
    dates  = weights_df.index.intersection(fwd_returns.index)
    tickers = weights_df.columns
    rows = []
    for dt in dates:
        w = weights_df.loc[dt].fillna(0)
        r = fwd_returns.loc[dt].reindex(tickers).fillna(0)
        rows.append(w * r)
    return pd.DataFrame(rows, index=dates)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Risk attribution (Component Risk Contribution)
# ─────────────────────────────────────────────────────────────────────────────
def _crc(w: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """CRC_i = w_i * (Σw)_i / σ_p.  Sums to σ_p (annualised)."""
    port_var = float(w @ cov @ w)
    port_vol = np.sqrt(max(port_var, 0.0))
    if port_vol < 1e-10:
        return np.zeros_like(w)
    return w * (cov @ w) / port_vol


def compute_risk_attribution(
    weights_df: pd.DataFrame,
    me_returns: pd.DataFrame,
    lookback:   int = LOOKBACK_MOS,
) -> pd.DataFrame:
    """
    For each rebalance date: annualised CRC per asset.
    Also returns portfolio vol (sum of all CRCs).
    """
    tickers   = weights_df.columns
    crc_rows  = []
    port_vols = []

    for dt in weights_df.index:
        w = weights_df.loc[dt].reindex(tickers).fillna(0).values

        if w.sum() < 1e-6:
            crc_rows.append(pd.Series(0.0, index=tickers))
            port_vols.append(np.nan)
            continue

        hist = me_returns.loc[:dt].reindex(columns=tickers).fillna(0).iloc[-lookback:]
        if len(hist) < 3:
            crc_rows.append(pd.Series(0.0, index=tickers))
            port_vols.append(np.nan)
            continue

        cov      = hist.cov().values * PERIODS          # annualised
        crc      = _crc(w, cov)
        port_vol = float(crc.sum())

        crc_rows.append(pd.Series(crc, index=tickers))
        port_vols.append(port_vol)

    df         = pd.DataFrame(crc_rows, index=weights_df.index)
    df["_pvol"] = port_vols
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. Drawdown attribution
# ─────────────────────────────────────────────────────────────────────────────
def _find_max_dd_dates(portfolio_ret: pd.Series):
    """Return (peak_date, trough_date) of the worst drawdown."""
    eq         = (1 + portfolio_ret).cumprod()
    dd         = eq / eq.cummax() - 1
    trough_dt  = dd.idxmin()
    trough_pos = portfolio_ret.index.get_loc(trough_dt)
    peak_dt    = eq.iloc[: trough_pos + 1].idxmax()
    return peak_dt, trough_dt


def compute_drawdown_attribution(
    weights_df:    pd.DataFrame,
    fwd_returns:   pd.DataFrame,
    portfolio_ret: pd.Series,
) -> dict:
    """
    Attribution split three ways:
      - During the single worst drawdown period
      - All months where portfolio is below its high-water mark
      - All positive months (what's driving the gains?)
    """
    ret_contrib = compute_return_attribution(weights_df, fwd_returns)

    # Align portfolio_ret to contribution dates
    port = portfolio_ret.reindex(ret_contrib.index).dropna()
    eq   = (1 + port).cumprod()
    in_dd = (eq / eq.cummax() - 1) < -0.001

    peak_dt, trough_dt = _find_max_dd_dates(port)
    mdd_mask = (ret_contrib.index >= peak_dt) & (ret_contrib.index <= trough_dt)

    return {
        "ret_contrib":      ret_contrib,
        "max_dd_dates":     (peak_dt, trough_dt),
        "max_dd_contrib":   ret_contrib[mdd_mask].sum(),
        "all_dd_contrib":   ret_contrib[in_dd].sum(),
        "all_up_contrib":   ret_contrib[~in_dd].sum(),
        "in_dd_mask":       in_dd,
        "portfolio_eq":     eq,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Summary builders
# ─────────────────────────────────────────────────────────────────────────────
def build_attribution_summary(
    weights_df:   pd.DataFrame,
    ret_contrib:  pd.DataFrame,
    risk_contrib: pd.DataFrame,
    dd_res:       dict,
) -> pd.DataFrame:
    tickers = weights_df.columns

    # Return attribution
    cum_ret_contrib  = ret_contrib.sum()                       # total over period
    avg_mo_contrib   = ret_contrib.mean() * PERIODS            # annualised avg monthly
    total_port_ret   = ret_contrib.sum(axis=1).sum()

    # Frequency and avg weight when held
    held             = (weights_df > 0.001)
    freq             = held.mean() * 100                       # % of months in portfolio
    avg_wt_when_held = (weights_df * held).sum() / held.sum() * 100   # avg weight when held

    # Risk attribution
    pvol_col  = risk_contrib["_pvol"]
    crc_df    = risk_contrib.drop(columns=["_pvol"])
    avg_crc   = crc_df.mean() * 100                           # avg annualised CRC %
    # Risk share = CRC_i / σ_p
    risk_share = (crc_df.div(pvol_col.replace(0, np.nan), axis=0)).mean() * 100

    # Drawdown attribution
    mdd_contrib = dd_res["max_dd_contrib"] * 100
    all_dd      = dd_res["all_dd_contrib"] * 100

    summary = pd.DataFrame({
        "Cum Return Contrib %":  (cum_ret_contrib * 100).round(2),
        "Ann Return Contrib %":  (avg_mo_contrib  * 100).round(3),
        "Avg CRC % (ann)":       avg_crc.round(3),
        "Risk Share %":          risk_share.round(1),
        "Freq % (in portfolio)": freq.round(1),
        "Avg Wt When Held %":    avg_wt_when_held.round(1),
        "Max DD Contrib %":      mdd_contrib.round(2),
        "All DD Contrib %":      all_dd.round(2),
    }, index=tickers)

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Print output
# ─────────────────────────────────────────────────────────────────────────────
def print_attribution_tables(summary: pd.DataFrame, dd_res: dict) -> None:
    peak_dt, trough_dt = dd_res["max_dd_dates"]

    print(f"\n{'='*95}")
    print(f"  RISK ATTRIBUTION — Momentum → Top-5 → 200DMA → HRP")
    print(f"{'='*95}")

    # ── Return attribution ──
    print(f"\n  1. RETURN ATTRIBUTION")
    print(f"  {'ETF':<6} {'Cum%':>8}  {'Ann%':>7}  {'Freq%':>7}  {'AvgWt(held)':>12}  Role")
    print(f"  {'─'*6} {'─'*8}  {'─'*7}  {'─'*7}  {'─'*12}  {'─'*20}")
    ret_sorted = summary.sort_values("Cum Return Contrib %", ascending=False)
    for tkr, row in ret_sorted.iterrows():
        if row["Freq % (in portfolio)"] < 1:
            continue
        cum = row["Cum Return Contrib %"]
        role = "★ TOP CONTRIBUTOR" if cum > 5 else ("✗ DRAG" if cum < -2 else "  moderate")
        print(f"  {tkr:<6} {cum:>8.2f}%  {row['Ann Return Contrib %']:>6.2f}%  "
              f"{row['Freq % (in portfolio)']:>6.1f}%  {row['Avg Wt When Held %']:>11.1f}%  {role}")

    total_ret = summary["Cum Return Contrib %"].sum()
    print(f"  {'─'*6} {'─'*8}")
    print(f"  {'TOTAL':<6} {total_ret:>8.2f}%  (strategy cumulative return)")

    # ── Risk attribution ──
    print(f"\n  2. RISK ATTRIBUTION  (Component Risk Contribution)")
    print(f"  {'ETF':<6} {'AvgCRC%':>8}  {'RiskSh%':>8}  {'AvgWt%':>8}  {'Risk/Wt':>9}  Assessment")
    print(f"  {'─'*6} {'─'*8}  {'─'*8}  {'─'*8}  {'─'*9}  {'─'*20}")
    # Average portfolio weight (not just when held)
    risk_sorted = summary.sort_values("Risk Share %", ascending=False)
    for tkr, row in risk_sorted.iterrows():
        if row["Risk Share %"] < 0.5:
            continue
        rsh  = row["Risk Share %"]
        freq = row["Freq % (in portfolio)"]
        if freq < 1:
            continue
        # Risk/weight ratio: risk share relative to average weight
        avg_wt_all = freq * row["Avg Wt When Held %"] / 100  # avg weight across ALL months
        ratio = rsh / avg_wt_all if avg_wt_all > 0 else np.nan
        flag = "HIGH RISK/WT" if ratio > 1.5 else ("low risk/wt" if ratio < 0.7 else "balanced")
        print(f"  {tkr:<6} {row['Avg CRC % (ann)']:>7.2f}%  {rsh:>7.1f}%  {avg_wt_all:>7.1f}%  "
              f"{ratio:>9.2f}x  {flag}")

    # ── Drawdown attribution ──
    print(f"\n  3. DRAWDOWN ATTRIBUTION")
    print(f"     Max drawdown period: {peak_dt.date()} → {trough_dt.date()}")
    print(f"  {'ETF':<6} {'MDD Period%':>12}  {'All DD months%':>15}  Assessment")
    print(f"  {'─'*6} {'─'*12}  {'─'*15}  {'─'*20}")
    dd_sorted = summary.sort_values("Max DD Contrib %")
    for tkr, row in dd_sorted.iterrows():
        mdd = row["Max DD Contrib %"]
        alldd = row["All DD Contrib %"]
        if abs(mdd) < 0.05 and abs(alldd) < 0.1:
            continue
        flag = "BIGGEST DRAG" if mdd < -2 else ("drag" if mdd < -0.5 else "negligible")
        print(f"  {tkr:<6} {mdd:>+11.2f}%  {alldd:>+14.2f}%  {flag}")

    print(f"{'='*95}")


# ─────────────────────────────────────────────────────────────────────────────
# Top-level runner
# ─────────────────────────────────────────────────────────────────────────────
def run_risk_attribution(
    signals:  dict,
    prices:   pd.DataFrame,
    returns:  pd.DataFrame,
    proc_dir: str = "data/processed",
) -> dict:
    from src.portfolio import build_weight_matrix
    from src.hypothesis_tests import compute_me_forward_returns
    from src.backtest import load_backtest_returns

    me_prices  = prices.resample("ME").last()
    me_returns = me_prices.pct_change().dropna()
    fwd        = compute_me_forward_returns(prices)

    weights_df = build_weight_matrix(signals, me_returns, n_top=5, method="hrp")

    bt_rets      = load_backtest_returns(proc_dir)
    portfolio_ret = bt_rets["D: HRP"].dropna()

    ret_contrib  = compute_return_attribution(weights_df, fwd)
    risk_contrib = compute_risk_attribution(weights_df, me_returns)
    dd_res       = compute_drawdown_attribution(weights_df, fwd, portfolio_ret)

    summary = build_attribution_summary(weights_df, ret_contrib, risk_contrib, dd_res)
    print_attribution_tables(summary, dd_res)

    return {
        "summary":      summary,
        "weights":      weights_df,
        "ret_contrib":  ret_contrib,
        "risk_contrib": risk_contrib,
        "dd_res":       dd_res,
        "me_returns":   me_returns,
        "fwd":          fwd,
    }
