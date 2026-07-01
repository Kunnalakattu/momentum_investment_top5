"""
Phase 18 — Production Monitoring

Generates a monthly monitoring report covering:
  - Current portfolio (weights, momentum scores, signals)
  - Performance vs benchmark (MTD, YTD, since inception)
  - Rolling Sharpe and drawdown
  - Turnover vs budget
  - Trade sheet (what to buy/sell this month)

Quarterly extensions (run with quarterly=True):
  - Re-runs IC analysis
  - Re-runs factor attribution
  - Re-runs cost analysis
  - Re-runs risk attribution

Run via: python -m src.monitoring  (or imported into main.py)
"""

import numpy as np
import pandas as pd
from datetime import datetime

PERIODS = 12


# ─────────────────────────────────────────────────────────────────────────────
# Current portfolio state
# ─────────────────────────────────────────────────────────────────────────────
def get_current_portfolio(
    signals:    dict,
    prices:     pd.DataFrame,
    returns:    pd.DataFrame,
    proc_dir:   str = "data/processed",
) -> dict:
    """
    Returns the current month's portfolio: weights, momentum scores, signals.
    """
    from src.portfolio import build_weight_matrix

    me_prices  = prices.resample("ME").last()
    me_returns = me_prices.pct_change().dropna()

    weights_df = build_weight_matrix(signals, me_returns, n_top=5, method="hrp")

    latest_dt = weights_df.index[-1]
    curr_wts  = weights_df.loc[latest_dt].sort_values(ascending=False)
    curr_wts  = curr_wts[curr_wts > 0.001]

    # Momentum scores
    score = signals.get("score")
    if score is not None and latest_dt in score.index:
        scores = score.loc[latest_dt].sort_values(ascending=False)
    else:
        scores = pd.Series(dtype=float)

    # 200DMA signal
    trend = signals.get("trend")
    if trend is not None and latest_dt in trend.index:
        dma_pass = trend.loc[latest_dt].astype(bool)
    else:
        dma_pass = pd.Series(dtype=bool)

    # Previous month's weights (for trade sheet)
    if len(weights_df) >= 2:
        prev_wts = weights_df.iloc[-2].sort_values(ascending=False)
    else:
        prev_wts = pd.Series(dtype=float)

    trade_sheet = (curr_wts - prev_wts.reindex(curr_wts.index).fillna(0)).round(4)

    return {
        "date":          latest_dt,
        "weights":       curr_wts,
        "scores":        scores,
        "dma_pass":      dma_pass,
        "trade_sheet":   trade_sheet,
        "prev_weights":  prev_wts,
        "weights_df":    weights_df,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Performance summary
# ─────────────────────────────────────────────────────────────────────────────
def performance_summary(
    hrp_ret:    pd.Series,
    spy_ret:    pd.Series,
    rf_monthly: pd.Series,
    report_date: pd.Timestamp = None,
) -> dict:
    """MTD, YTD, 1Y, 3Y, 5Y, ITD CAGR and Sharpe for strategy vs VUSA."""
    if report_date is None:
        report_date = hrp_ret.index[-1]

    rf = rf_monthly.reindex(hrp_ret.index).fillna(0)

    def _m(ret, periods_back=None, start=None):
        if start is not None:
            ret = ret.loc[start:]
        elif periods_back is not None:
            ret = ret.iloc[-periods_back:]
        if len(ret) < 1:
            return {"CAGR %": np.nan, "Sharpe": np.nan, "MaxDD %": np.nan}
        n_yr = max(len(ret) / PERIODS, 1 / PERIODS)
        cagr = float((1 + ret).prod() ** (1 / n_yr) - 1)
        vol  = float(ret.std() * np.sqrt(PERIODS))
        rf_a = float(rf.reindex(ret.index).fillna(0).mean() * PERIODS)
        sh   = (cagr - rf_a) / vol if vol > 0 else np.nan
        eq   = (1 + ret).cumprod()
        dd   = float((eq / eq.cummax() - 1).min())
        return {"CAGR %": round(cagr*100,2), "Sharpe": round(sh,3), "MaxDD %": round(dd*100,2)}

    yr = report_date.year
    ytd_start = pd.Timestamp(f"{yr}-01-01")
    one_y_start = report_date - pd.DateOffset(years=1)
    three_y_start = report_date - pd.DateOffset(years=3)
    five_y_start = report_date - pd.DateOffset(years=5)

    spy = spy_ret.reindex(hrp_ret.index).fillna(0)

    return {
        "MTD":   {"HRP": _m(hrp_ret, periods_back=1), "VUSA": _m(spy, periods_back=1)},
        "YTD":   {"HRP": _m(hrp_ret, start=ytd_start), "VUSA": _m(spy, start=ytd_start)},
        "1Y":    {"HRP": _m(hrp_ret, start=one_y_start), "VUSA": _m(spy, start=one_y_start)},
        "3Y":    {"HRP": _m(hrp_ret, start=three_y_start), "VUSA": _m(spy, start=three_y_start)},
        "5Y":    {"HRP": _m(hrp_ret, start=five_y_start), "VUSA": _m(spy, start=five_y_start)},
        "ITD":   {"HRP": _m(hrp_ret), "VUSA": _m(spy)},
    }


# ─────────────────────────────────────────────────────────────────────────────
# Rolling metrics
# ─────────────────────────────────────────────────────────────────────────────
def rolling_monitor(
    hrp_ret:    pd.Series,
    spy_ret:    pd.Series,
    rf_monthly: pd.Series,
    window:     int = 12,
) -> pd.DataFrame:
    """Rolling Sharpe, drawdown, active return."""
    rf = rf_monthly.reindex(hrp_ret.index).fillna(0)
    rf_ann = rf.mean() * PERIODS
    spy = spy_ret.reindex(hrp_ret.index).fillna(0)

    rows = {}
    for i in range(window, len(hrp_ret) + 1):
        h  = hrp_ret.iloc[i-window:i]
        s  = spy.iloc[i-window:i]
        n_yr = window / PERIODS
        cagr_h = float((1 + h).prod() ** (1/n_yr) - 1)
        cagr_s = float((1 + s).prod() ** (1/n_yr) - 1)
        vol_h  = float(h.std() * np.sqrt(PERIODS))
        sh_h   = (cagr_h - rf_ann) / vol_h if vol_h > 0 else np.nan
        eq     = (1 + hrp_ret.iloc[:i]).cumprod()
        curr_dd = float((eq / eq.cummax() - 1).iloc[-1])
        rows[hrp_ret.index[i-1]] = {
            "Rolling Sharpe":    sh_h,
            "Rolling CAGR %":   cagr_h * 100,
            "Active CAGR %":    (cagr_h - cagr_s) * 100,
            "Current DD %":     curr_dd * 100,
        }
    return pd.DataFrame(rows).T


# ─────────────────────────────────────────────────────────────────────────────
# Print monthly report
# ─────────────────────────────────────────────────────────────────────────────
def print_monthly_report(
    port:     dict,
    perf:     dict,
    hrp_ret:  pd.Series,
    spy_ret:  pd.Series,
) -> None:
    today = datetime.now().strftime("%Y-%m-%d")
    print(f"\n{'='*70}")
    print(f"  MONTHLY MONITORING REPORT — {port['date'].strftime('%B %Y')}")
    print(f"  Generated: {today}")
    print(f"{'='*70}")

    # Performance table
    print(f"\n  PERFORMANCE SUMMARY")
    print(f"  {'Period':<8}  {'HRP CAGR':>10}  {'HRP Sharpe':>11}  {'HRP MaxDD':>10}  {'VUSA CAGR':>10}  {'Active':>8}")
    print(f"  {'─'*8}  {'─'*10}  {'─'*11}  {'─'*10}  {'─'*10}  {'─'*8}")
    for period, d in perf.items():
        h = d["HRP"]
        s = d["VUSA"]
        if np.isnan(h.get("CAGR %", np.nan)):
            continue
        active = round(h["CAGR %"] - s["CAGR %"], 2) if not np.isnan(s.get("CAGR %", np.nan)) else np.nan
        print(f"  {period:<8}  {h['CAGR %']:>+9.2f}%  {h['Sharpe']:>11.3f}  "
              f"{h['MaxDD %']:>+9.2f}%  {s['CAGR %']:>+9.2f}%  {active:>+7.2f}%")

    # Current portfolio
    print(f"\n  CURRENT PORTFOLIO ({port['date'].strftime('%Y-%m-%d')})")
    print(f"  {'ETF':<6}  {'Weight':>8}  {'Score':>8}  {'200DMA':>8}  {'Trade':>10}")
    print(f"  {'─'*6}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*10}")
    for tkr, wt in port["weights"].items():
        sc     = port["scores"].get(tkr, np.nan)
        dma    = port["dma_pass"].get(tkr, np.nan)
        trade  = port["trade_sheet"].get(tkr, 0)
        sc_str = f"{sc:.3f}" if not np.isnan(sc) else " n/a"
        dma_str = "PASS" if dma else "FAIL" if dma is not None else " n/a"
        trd_str = f"{trade*100:+.1f}%" if abs(trade) > 0.001 else " no change"
        print(f"  {tkr:<6}  {wt*100:>7.1f}%  {sc_str:>8}  {dma_str:>8}  {trd_str:>10}")

    # Current drawdown
    eq  = (1 + hrp_ret).cumprod()
    dd  = (eq / eq.cummax() - 1)
    curr_dd = float(dd.iloc[-1])
    max_dd  = float(dd.min())
    peak_eq = float(eq.cummax().iloc[-1])
    curr_eq = float(eq.iloc[-1])

    print(f"\n  DRAWDOWN STATUS")
    print(f"  Current drawdown  : {curr_dd*100:+.2f}%")
    print(f"  All-time max DD   : {max_dd*100:+.2f}%")
    if curr_dd < -0.05:
        print(f"  ⚠ IN DRAWDOWN: {curr_dd*100:.1f}% below peak")
    else:
        print(f"  ✓ Near all-time high")

    print(f"{'='*70}")


# ─────────────────────────────────────────────────────────────────────────────
# Top-level runner
# ─────────────────────────────────────────────────────────────────────────────
def run_monitoring(
    signals:  dict,
    prices:   pd.DataFrame,
    returns:  pd.DataFrame,
    proc_dir: str = "data/processed",
    quarterly: bool = False,
) -> dict:
    from src.backtest import load_backtest_returns
    from src.data import load_risk_free_rate

    rf_daily   = load_risk_free_rate(proc_dir)
    rf_monthly = (1 + rf_daily).resample("ME").prod() - 1

    bt_rets    = load_backtest_returns(proc_dir)
    hrp_ret    = bt_rets["D: HRP"].dropna()
    me_prices  = prices.resample("ME").last()
    bench_col  = "VUSA" if "VUSA" in me_prices.columns else me_prices.columns[0]
    spy_ret    = me_prices[bench_col].pct_change().dropna()

    port       = get_current_portfolio(signals, prices, returns, proc_dir)
    perf       = performance_summary(hrp_ret, spy_ret, rf_monthly)
    roll_df    = rolling_monitor(hrp_ret, spy_ret, rf_monthly)

    print_monthly_report(port, perf, hrp_ret, spy_ret)

    result = {
        "portfolio":     port,
        "performance":   perf,
        "rolling":       roll_df,
        "hrp_ret":       hrp_ret,
        "spy_ret":       spy_ret,
        "rf_monthly":    rf_monthly,
    }

    if quarterly:
        print("\n  [QUARTERLY] Re-running IC, factor attribution, cost, risk analysis...")
        from src.stats_validation import run_stats_validation
        from src.factor_attribution import run_factor_attribution
        from src.cost_sensitivity import run_cost_sensitivity_pipeline
        from src.risk_attribution import run_risk_attribution

        result["quarterly"] = {
            "stats":   run_stats_validation(signals, prices, returns, proc_dir),
            "factor":  run_factor_attribution(prices, returns, proc_dir),
            "costs":   run_cost_sensitivity_pipeline(prices, returns, proc_dir),
            "risk":    run_risk_attribution(signals, prices, returns, proc_dir),
        }

    return result
