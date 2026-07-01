"""
Strategy Health Dashboard

Runs six automated checks against pre-defined thresholds decided before
capital is committed. Produces a GREEN / AMBER / RED verdict for each metric
and an overall HEALTHY / WATCH / ACT signal.

Thresholds are fixed here — never adjusted retroactively.

    Metric               Green           Amber               Red
    ─────────────────────────────────────────────────────────────
    Max Drawdown         < hist worst    within 10% of worst > hist worst
    Rolling Sharpe 36m   > 0.7           0.5 – 0.7           < 0.5
    Info Coefficient     > 0.02          0 – 0.02            < 0 sustained
    Monthly Turnover     ≤ 1.5x avg      1.5x – 2.0x avg     > 2.0x avg
    Data Quality         0 missing days  < 5 missing days    ≥ 5 missing days
    ETF Availability     all present     minor gaps           price = 0 or NaN
"""

import numpy as np
import pandas as pd

PERIODS = 12

GREEN = "GREEN"
AMBER = "AMBER"
RED   = "RED"

_ICONS = {GREEN: "●", AMBER: "◑", RED: "○"}   # monochrome-safe


# ─────────────────────────────────────────────────────────────────────────────
# Individual checks
# ─────────────────────────────────────────────────────────────────────────────

def check_drawdown(hrp_ret: pd.Series) -> dict:
    """
    Compare current drawdown against the historical worst.
    Green  — at least 10% better than historical worst
    Amber  — within 10% of historical worst
    Red    — new all-time drawdown (worse than historical worst)
    """
    eq      = (1 + hrp_ret).cumprod()
    dd      = (eq / eq.cummax() - 1)
    hist_worst = float(dd.min())           # e.g. -0.129
    current_dd = float(dd.iloc[-1])        # e.g. -0.034

    buffer = abs(hist_worst) * 0.10        # 10% of worst

    if current_dd <= hist_worst:           # new all-time low
        status = RED
        action = "New all-time drawdown — review positions and signal integrity."
    elif current_dd <= hist_worst + buffer:
        status = AMBER
        action = f"Within 10% of historical worst ({hist_worst*100:.1f}%). Monitor closely."
    else:
        status = GREEN
        action = "Drawdown within normal range."

    return {
        "metric":    "Max Drawdown",
        "status":    status,
        "value":     f"{current_dd*100:.1f}%",
        "threshold": f"Hist worst: {hist_worst*100:.1f}%  |  Amber zone: {(hist_worst+buffer)*100:.1f}%",
        "action":    action,
        "raw":       {"current": current_dd, "hist_worst": hist_worst},
    }


def check_rolling_sharpe(hrp_ret: pd.Series, rf_monthly: pd.Series,
                          window: int = 36) -> dict:
    """
    Rolling Sharpe over last `window` months.
    Green  — > 0.7
    Amber  — 0.5 – 0.7
    Red    — < 0.5
    """
    if len(hrp_ret) < window:
        window = len(hrp_ret)

    ret = hrp_ret.iloc[-window:]
    rf  = rf_monthly.reindex(ret.index).fillna(0)
    n_yr = window / PERIODS
    cagr  = float((1 + ret).prod() ** (1 / n_yr) - 1)
    vol   = float(ret.std() * np.sqrt(PERIODS))
    rf_a  = float(rf.mean() * PERIODS)
    sharpe = (cagr - rf_a) / vol if vol > 0 else np.nan

    if np.isnan(sharpe):
        status = AMBER
        action = "Not enough data to compute rolling Sharpe."
    elif sharpe >= 0.7:
        status = GREEN
        action = "Rolling Sharpe healthy."
    elif sharpe >= 0.5:
        status = AMBER
        action = "Rolling Sharpe degraded. Check for regime change or data issues."
    else:
        status = RED
        action = "Rolling Sharpe below 0.5. Strategy may be failing in current environment."

    return {
        "metric":    f"Rolling Sharpe ({window}m)",
        "status":    status,
        "value":     f"{sharpe:.3f}" if not np.isnan(sharpe) else "n/a",
        "threshold": "Green > 0.7  |  Amber 0.5–0.7  |  Red < 0.5",
        "action":    action,
        "raw":       {"sharpe": sharpe, "window": window},
    }


def check_ic(signals: dict, prices: pd.DataFrame,
             lookback_months: int = 12) -> dict:
    """
    Information Coefficient over the most recent `lookback_months`.
    Green  — mean IC > 0.02
    Amber  — 0 < mean IC ≤ 0.02
    Red    — mean IC ≤ 0  (signal has no predictive power recently)
    """
    try:
        from src.stats_validation import ic_full_analysis
        from src.hypothesis_tests import compute_me_forward_returns

        fwd = compute_me_forward_returns(prices)

        # Restrict signals and fwd to the last N months
        sig_sub = {}
        for k, v in signals.items():
            if isinstance(v, pd.DataFrame) and len(v) > lookback_months:
                sig_sub[k] = v.iloc[-lookback_months:]
            else:
                sig_sub[k] = v
        fwd_sub = fwd.reindex(sig_sub.get("score", fwd).index) if "score" in sig_sub else fwd

        res     = ic_full_analysis(sig_sub, fwd_sub)
        mean_ic = float(res.get("mean_ic", np.nan))
        p_val   = float(res.get("p_value", 1.0))
        icir    = float(res.get("icir_annual", np.nan))

        if np.isnan(mean_ic):
            status = AMBER
            action = "IC could not be computed — check signals."
        elif mean_ic > 0.02:
            status = GREEN
            action = "Signal has positive predictive power."
        elif mean_ic > 0:
            status = AMBER
            action = "IC near zero. Signal is weak recently — watch over next quarter."
        else:
            status = RED
            action = "IC negative. Signal has lost predictive power. Run annual review early."

        value = f"{mean_ic:.4f}" if not np.isnan(mean_ic) else "n/a"
        thresh = f"Green > 0.02  |  Amber 0–0.02  |  Red ≤ 0  (p={p_val:.3f})"

    except Exception as e:
        status = AMBER
        value  = "error"
        thresh = "n/a"
        action = f"IC check failed: {e}"
        mean_ic = np.nan
        icir    = np.nan

    return {
        "metric":    f"Info Coefficient ({lookback_months}m)",
        "status":    status,
        "value":     value,
        "threshold": thresh,
        "action":    action,
        "raw":       {"mean_ic": mean_ic},
    }


def check_turnover(weights_df: pd.DataFrame, lookback_months: int = 3) -> dict:
    """
    Compare recent monthly turnover against historical average.
    Green  — recent ≤ 1.5× historical average
    Amber  — 1.5× – 2.0× historical average
    Red    — > 2.0× historical average
    """
    monthly_to = weights_df.diff().abs().sum(axis=1) / 2   # one-way
    hist_avg   = float(monthly_to.iloc[:-lookback_months].mean()) if len(monthly_to) > lookback_months else float(monthly_to.mean())
    recent_avg = float(monthly_to.iloc[-lookback_months:].mean())

    ratio = recent_avg / hist_avg if hist_avg > 0 else 1.0

    if ratio <= 1.5:
        status = GREEN
        action = "Turnover within expected range."
    elif ratio <= 2.0:
        status = AMBER
        action = f"Turnover {ratio:.1f}× historical avg. Check for signal instability or data anomalies."
    else:
        status = RED
        action = f"Turnover {ratio:.1f}× historical avg. Check for data error or ETF corporate actions."

    return {
        "metric":    "Monthly Turnover",
        "status":    status,
        "value":     f"{recent_avg*100:.1f}%/mo (recent)  vs  {hist_avg*100:.1f}%/mo (hist avg)",
        "threshold": "Green ≤1.5×avg  |  Amber 1.5–2×avg  |  Red >2×avg",
        "action":    action,
        "raw":       {"recent_avg": recent_avg, "hist_avg": hist_avg, "ratio": ratio},
    }


def check_data_quality(prices: pd.DataFrame, lookback_days: int = 30) -> dict:
    """
    Count missing values and detect price anomalies in recent data.
    Green  — 0 missing trading days, no price jumps > 25% in a single day
    Amber  — 1–4 missing days OR suspicious price moves in non-volatile ETFs
    Red    — ≥ 5 missing days OR confirmed data corruption
    """
    recent   = prices.iloc[-lookback_days:]
    missing  = int(recent.isna().any(axis=1).sum())
    stale    = int((prices.iloc[-5:].std() == 0).sum())   # ETFs with no movement

    # Detect suspicious daily returns (> 25% in a day for low-vol ETFs)
    daily_ret = prices.pct_change().iloc[-lookback_days:]
    suspicious = (daily_ret.abs() > 0.25).any().sum()

    issues = []
    if missing >= 5:
        status = RED
        issues.append(f"{missing} missing trading days in last {lookback_days}d")
    elif missing > 0:
        status = AMBER
        issues.append(f"{missing} missing trading day(s) in last {lookback_days}d")
    else:
        status = GREEN

    if suspicious > 0:
        status = max(status, AMBER, key=[GREEN, AMBER, RED].index)
        issues.append(f"{suspicious} ETF(s) with >25% single-day move")

    if stale > 0:
        status = max(status, AMBER, key=[GREEN, AMBER, RED].index)
        issues.append(f"{stale} ETF(s) with no price movement (last 5 days)")

    value  = "No issues" if not issues else "; ".join(issues)
    action = ("Data looks clean." if status == GREEN else
              "Review flagged ETFs before trading." if status == AMBER else
              "Do NOT trade until data issues are resolved.")

    return {
        "metric":    "Data Quality",
        "status":    status,
        "value":     value,
        "threshold": "Green: 0 missing  |  Amber: 1–4  |  Red: ≥5 or corrupted",
        "action":    action,
        "raw":       {"missing": missing, "suspicious": suspicious, "stale": stale},
    }


def check_etf_availability(prices: pd.DataFrame, target_tickers: list[str]) -> dict:
    """
    Confirm every ETF in the target universe has recent, non-zero prices.
    Green  — all ETFs tradable, prices current
    Amber  — 1 ETF with thin/stale data
    Red    — any ETF with zero/missing price (possibly delisted)
    """
    latest = prices.ffill().iloc[-1]
    issues = []

    for tkr in target_tickers:
        if tkr not in latest.index or pd.isna(latest[tkr]):
            issues.append(f"{tkr}: no price data")
        elif latest[tkr] <= 0:
            issues.append(f"{tkr}: price = ${latest[tkr]:.2f} (check delisting)")

    # Check for ETFs that haven't traded in > 5 days
    for tkr in target_tickers:
        if tkr in prices.columns:
            last_valid = prices[tkr].dropna()
            if len(last_valid) > 0:
                days_since = (prices.index[-1] - last_valid.index[-1]).days
                if days_since > 5:
                    issues.append(f"{tkr}: last price {days_since}d ago")

    if len(issues) == 0:
        status = GREEN
        value  = f"All {len(target_tickers)} ETFs available"
        action = "No ETF issues."
    elif len(issues) == 1:
        status = AMBER
        value  = issues[0]
        action = "Monitor. If ETF is delisted, replace per predefined rules."
    else:
        status = RED
        value  = "; ".join(issues)
        action = "One or more ETFs unavailable. Do NOT trade until resolved."

    return {
        "metric":    "ETF Availability",
        "status":    status,
        "value":     value,
        "threshold": "Green: all present  |  Amber: 1 issue  |  Red: delisted/illiquid",
        "action":    action,
        "raw":       {"issues": issues},
    }


# ─────────────────────────────────────────────────────────────────────────────
# Overall verdict
# ─────────────────────────────────────────────────────────────────────────────

def overall_verdict(checks: list[dict]) -> tuple[str, str]:
    counts = {GREEN: 0, AMBER: 0, RED: 0}
    for c in checks:
        counts[c["status"]] += 1

    if counts[RED] > 0:
        verdict = "ACT NOW"
        msg     = "One or more RED metrics. Investigate before next trade."
    elif counts[AMBER] >= 2:
        verdict = "WATCH"
        msg     = "Multiple AMBER metrics. Review before next rebalance."
    elif counts[AMBER] == 1:
        verdict = "WATCH"
        msg     = "One AMBER metric. Note and monitor over next month."
    else:
        verdict = "HEALTHY"
        msg     = "All metrics green. Proceed with normal monthly rebalance."

    return verdict, msg


# ─────────────────────────────────────────────────────────────────────────────
# Print report
# ─────────────────────────────────────────────────────────────────────────────

def print_health_report(checks: list[dict], report_date: pd.Timestamp = None) -> None:
    from datetime import datetime
    date_str = (report_date.strftime("%Y-%m-%d") if report_date
                else datetime.now().strftime("%Y-%m-%d"))

    verdict, msg = overall_verdict(checks)
    counts = {s: sum(1 for c in checks if c["status"] == s) for s in [GREEN, AMBER, RED]}

    print(f"\n{'='*72}")
    print(f"  STRATEGY HEALTH DASHBOARD")
    print(f"  {date_str}")
    print(f"{'='*72}")
    print(f"\n  {'Metric':<28}  {'Status':<10}  {'Value'}")
    print(f"  {'─'*28}  {'─'*10}  {'─'*30}")

    for c in checks:
        icon = _ICONS[c["status"]]
        print(f"  {c['metric']:<28}  {icon} {c['status']:<8}  {c['value']}")
        print(f"  {'':28}  {'':10}  Threshold : {c['threshold']}")
        if c["status"] != GREEN:
            print(f"  {'':28}  {'':10}  Action    : {c['action']}")
        print()

    print(f"{'─'*72}")
    print(f"  {counts[GREEN]} GREEN  |  {counts[AMBER]} AMBER  |  {counts[RED]} RED")
    print()
    verdict_line = f"  ► {verdict}: {msg}"
    print(verdict_line)
    print(f"{'='*72}")


# ─────────────────────────────────────────────────────────────────────────────
# Top-level runner
# ─────────────────────────────────────────────────────────────────────────────

def run_health_check(
    signals:    dict,
    prices:     pd.DataFrame,
    returns:    pd.DataFrame,
    proc_dir:   str = "data/processed",
) -> dict:
    from src.backtest import load_backtest_returns
    from src.data import load_risk_free_rate
    from src.signals import load_signals
    from src.portfolio import build_weight_matrix

    rf_daily   = load_risk_free_rate(proc_dir)
    rf_monthly = (1 + rf_daily).resample("ME").prod() - 1

    bt = load_backtest_returns(proc_dir)
    hrp_ret = bt["D: HRP"].dropna()

    me_returns = prices.resample("ME").last().pct_change().dropna()
    weights_df = build_weight_matrix(signals, me_returns, n_top=5, method="hrp")

    target_tickers = list(prices.columns)

    checks = [
        check_drawdown(hrp_ret),
        check_rolling_sharpe(hrp_ret, rf_monthly, window=36),
        check_ic(signals, prices, lookback_months=12),
        check_turnover(weights_df, lookback_months=3),
        check_data_quality(prices, lookback_days=30),
        check_etf_availability(prices, target_tickers),
    ]

    print_health_report(checks, report_date=prices.resample("ME").last().index[-1])

    verdict, msg = overall_verdict(checks)
    return {
        "checks":  checks,
        "verdict": verdict,
        "message": msg,
    }
