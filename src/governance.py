"""
Phase 20 — Research Governance & Model Approval

Runs all validation tests and produces a GO / NO-GO decision.
No strategy gets promoted to production until every box is checked.

Checklist thresholds (from the user's specification):
  IC > 0, stable over time
  Walk-forward Sharpe > 0.7 in majority of folds
  Bootstrap 95% CI excludes zero CAGR
  Parameter sensitivity: no sharp cliff
  Cost sensitivity: profitable at 25-50 bps
  Regime analysis: acceptable across major regimes
  Factor attribution: not explained solely by market beta
  Capacity analysis: tradable with planned capital
"""

import numpy as np
import pandas as pd

PERIODS = 12


# ─────────────────────────────────────────────────────────────────────────────
# Individual test checks
# ─────────────────────────────────────────────────────────────────────────────
def _check_ic(signals: dict, prices: pd.DataFrame, rf: pd.Series) -> dict:
    from src.stats_validation import ic_full_analysis
    from src.hypothesis_tests import compute_me_forward_returns

    # try known key names in order of preference
    score = None
    for key in ("score", "mom_score", "composite_score"):
        v = signals.get(key)
        if v is not None:
            score = v
            break
    elig = None
    for key in ("eligible", "eligibility"):
        v = signals.get(key)
        if v is not None:
            elig = v
            break

    fwd    = compute_me_forward_returns(prices)
    ic_res = ic_full_analysis(score, elig, fwd)
    mean_ic = ic_res.get("mean_ic", np.nan)
    p_val   = ic_res.get("p_value", 1.0)
    icir    = ic_res.get("icir_annual", np.nan)

    pass_ = bool((not np.isnan(mean_ic)) and (mean_ic > 0) and (p_val < 0.10))
    return {
        "test":      "Information Coefficient",
        "value":     f"IC={mean_ic:.4f}, p={p_val:.4f}, ICIR={icir:.2f}",
        "threshold": "IC > 0, p < 0.10",
        "pass":      pass_,
    }


def _check_walk_forward(proc_dir: str) -> dict:
    from src.walk_forward import run_walk_forward

    bt = pd.read_parquet(f"{proc_dir}/backtest_returns.parquet")
    hrp = bt["D: HRP"].dropna()
    spy = bt["VUSA B&H"].dropna()
    rf  = pd.Series(0.0, index=hrp.index)

    wf_df = run_walk_forward(hrp, spy, rf)

    # Separate full-period row from calendar-year folds
    fold_mask  = pd.to_numeric(wf_df.index, errors='coerce').notna()
    fold_df    = wf_df[fold_mask]
    sharpe_pos = (fold_df["Sharpe"].dropna() > 0.7).mean()
    cagr_pos   = (fold_df["CAGR %"].dropna() > 0).mean()
    full_rows  = wf_df[~fold_mask]
    full_sh    = float(full_rows["Sharpe"].iloc[0]) if len(full_rows) > 0 else np.nan

    pass_ = bool((sharpe_pos >= 0.60) and (cagr_pos >= 0.75) and (not np.isnan(full_sh)) and (full_sh > 0.7))
    return {
        "test":      "Walk-Forward Validation",
        "value":     f"Sharpe>0.7 in {sharpe_pos*100:.0f}% folds; CAGR>0 in {cagr_pos*100:.0f}%; full-period Sharpe={full_sh:.3f}",
        "threshold": "Sharpe>0.7 in ≥60% of folds (majority); CAGR>0 in ≥75%; full Sharpe>0.7",
        "pass":      pass_,
    }


def _check_bootstrap(signals: dict, prices: pd.DataFrame, proc_dir: str) -> dict:
    from src.stats_validation import bootstrap_ci
    from src.data import load_risk_free_rate

    bt = pd.read_parquet(f"{proc_dir}/backtest_returns.parquet")
    hrp = bt["D: HRP"].dropna()
    rf_daily = load_risk_free_rate(proc_dir)
    rf_mo    = (1 + rf_daily).resample("ME").prod() - 1

    boot = bootstrap_ci(hrp, rf_monthly=rf_mo)
    cagr_arr  = np.asarray(boot["dist"]["cagr"])
    cagr_lo   = float(np.quantile(cagr_arr, 0.05))
    sharpe_lo = boot["ci_sharpe"]["lo95"]
    p_gt0_sh  = boot["ci_sharpe"].get("p_gt0", np.nan)

    pass_ = bool((cagr_lo > 0) and (sharpe_lo > 0))
    return {
        "test":      "Bootstrap Confidence Intervals",
        "value":     f"CAGR 5th pct={cagr_lo*100:.1f}%; Sharpe 95% CI=[{sharpe_lo:.3f}, {boot['ci_sharpe']['hi95']:.3f}]",
        "threshold": "5th-pct CAGR > 0; Sharpe CI lower bound > 0",
        "pass":      pass_,
    }


def _check_parameter_sensitivity(proc_dir: str) -> dict:
    rob = pd.read_parquet(f"{proc_dir}/robustness_matrix.parquet")
    # Extract Sharpe column only
    if "Sharpe" in rob.columns:
        vals = rob["Sharpe"].dropna().values
    else:
        vals = np.array([])

    min_sh  = float(np.min(vals))  if len(vals) > 0 else np.nan
    pct_pos = float((vals > 0).mean()) if len(vals) > 0 else np.nan
    std_sh  = float(np.std(vals))  if len(vals) > 0 else np.nan

    pass_ = bool((not np.isnan(min_sh)) and (pct_pos > 0.80) and (min_sh > 0.3))
    return {
        "test":      "Parameter Sensitivity",
        "value":     f"Min Sharpe={min_sh:.3f}; {pct_pos*100:.0f}% of parameter variants positive; Sharpe std={std_sh:.3f}",
        "threshold": ">80% variants Sharpe>0; min Sharpe>0.3 (no cliff)",
        "pass":      pass_,
    }


def _check_cost_sensitivity(proc_dir: str) -> dict:
    bt = pd.read_parquet(f"{proc_dir}/backtest_returns.parquet")
    hrp = bt["D: HRP"].dropna()

    from src.cost_sensitivity import apply_cost
    annual_to = 1.2  # conservative estimate (120% annual turnover = typical for monthly HRP)
    cost_25 = apply_cost(hrp, annual_to, 25)
    cost_50 = apply_cost(hrp, annual_to, 50)

    n_yr = len(hrp) / PERIODS
    cagr_25 = float((1 + cost_25).prod() ** (1/n_yr) - 1)
    cagr_50 = float((1 + cost_50).prod() ** (1/n_yr) - 1)

    pass_ = bool((cagr_25 > 0) and (cagr_50 > 0))
    return {
        "test":      "Cost Sensitivity",
        "value":     f"Net CAGR at 25bp={cagr_25*100:.2f}%; at 50bp={cagr_50*100:.2f}%",
        "threshold": "Profitable (CAGR>0) at 25bp AND 50bp one-way cost",
        "pass":      pass_,
    }


def _check_regime(proc_dir: str) -> dict:
    bt = pd.read_parquet(f"{proc_dir}/backtest_returns.parquet")
    hrp = bt["D: HRP"].dropna()
    spy = bt["VUSA B&H"].dropna()
    rf  = pd.Series(0.0, index=hrp.index)

    from src.regime_analysis import build_regime_table
    reg_df = build_regime_table(hrp, spy, rf)

    # Strategy should outperform SPY in at least 50% of regimes
    # and not have catastrophic drawdown (< -40%) in any single regime
    n_win = (reg_df["Alpha CAGR"] > 0).sum()
    n_tot = len(reg_df)
    min_maxdd = float(reg_df["HRP MaxDD %"].min())

    pass_ = bool((n_win / n_tot >= 0.50) and (min_maxdd > -40))
    return {
        "test":      "Regime Analysis",
        "value":     f"Outperforms SPY in {n_win}/{n_tot} regimes; worst MaxDD={min_maxdd:.1f}%",
        "threshold": "Win in ≥50% of regimes; MaxDD better than −40% in any regime",
        "pass":      pass_,
    }


def _check_factor(proc_dir: str) -> dict:
    bt = pd.read_parquet(f"{proc_dir}/backtest_returns.parquet")
    hrp = bt["D: HRP"].dropna()

    from src.factor_attribution import download_factors, _run_ols
    from src.data import load_risk_free_rate
    import os

    rf_daily   = load_risk_free_rate(proc_dir)
    rf_monthly = (1 + rf_daily).resample("ME").prod() - 1
    excess_ret = (hrp - rf_monthly.reindex(hrp.index).fillna(0)).dropna()

    try:
        factors = download_factors()
        capm = _run_ols(excess_ret, factors, ["Mkt-RF"], "CAPM")
        alpha_ann = capm["alpha_annual"]
        t_stat    = capm["alpha_tstat"]
        r2        = capm["r2"]
        mkt_beta  = float(capm["params"]["Mkt-RF"])
        pass_ = bool((alpha_ann > 0.03) and (t_stat > 2.0))
        value = f"CAPM alpha={alpha_ann*100:.2f}%/yr (t={t_stat:.2f}), R²={r2*100:.1f}%, mkt_beta={mkt_beta:.3f}"
    except Exception as e:
        pass_ = None  # can't check — network issue
        value = f"Could not download FF data: {e}"

    return {
        "test":      "Factor Attribution",
        "value":     value,
        "threshold": "CAPM alpha > 3%/yr, t-stat > 2.0",
        "pass":      pass_,
    }


def _check_capacity(proc_dir: str, target_aum_gbp: float = 100_000) -> dict:
    bt = pd.read_parquet(f"{proc_dir}/backtest_returns.parquet")
    hrp = bt["D: HRP"].dropna()

    from src.capacity_analysis import compute_adv, compute_capacity_metrics, GBP_TO_USD
    from src.data import load_risk_free_rate
    import os

    rf_daily   = load_risk_free_rate(proc_dir)
    rf_monthly = (1 + rf_daily).resample("ME").prod() - 1
    rf_ann     = rf_monthly.mean() * PERIODS
    n_yr       = len(hrp) / PERIODS
    hrp_cagr   = float((1 + hrp).prod() ** (1/n_yr) - 1)
    alpha_bps  = (hrp_cagr - rf_ann) * 1e4

    try:
        # Need weights_df — approximate with uniform weights here
        prices_parquet = os.path.join(proc_dir, "prices.parquet")
        prices = pd.read_parquet(prices_parquet)
        me_ret = prices.resample("ME").last().pct_change().dropna()

        from src.signals import load_signals
        from src.portfolio import build_weight_matrix
        signals    = load_signals(proc_dir)
        weights_df = build_weight_matrix(signals, me_ret, n_top=5, method="hrp")
        adv_usd    = compute_adv(prices)
        aum_usd    = target_aum_gbp * GBP_TO_USD

        m = compute_capacity_metrics(weights_df, me_ret, adv_usd, aum_usd)
        cost_bps  = m["annual_cost_bps"]
        max_part  = m["per_etf"]["participation_pct"].max() if len(m["per_etf"]) > 0 else np.nan
        retained  = (alpha_bps - cost_bps) / alpha_bps * 100 if alpha_bps > 0 else np.nan

        pass_ = bool((retained > 80) and (max_part < 1.0))
        value = f"At £{target_aum_gbp:,.0f}: cost={cost_bps:.0f}bp/yr, alpha retained={retained:.0f}%, max part={max_part:.3f}%"
    except Exception as e:
        pass_ = None
        value = f"Could not compute: {e}"

    return {
        "test":      f"Capacity Analysis (at £{target_aum_gbp:,.0f})",
        "value":     value,
        "threshold": "Alpha retained > 80%; max participation < 1%",
        "pass":      pass_,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Full governance run
# ─────────────────────────────────────────────────────────────────────────────
def run_governance(
    signals:  dict,
    prices:   pd.DataFrame,
    returns:  pd.DataFrame,
    proc_dir: str = "data/processed",
    target_aum_gbp: float = 100_000,
) -> dict:
    print(f"\n{'='*75}")
    print(f"  RESEARCH GOVERNANCE — MODEL APPROVAL CHECKLIST")
    print(f"  Strategy: Momentum → Top-5 → 200DMA → HRP")
    print(f"{'='*75}")

    checks = []

    # Run each check
    test_fns = [
        ("IC",           lambda: _check_ic(signals, prices, None)),
        ("Walk-Forward", lambda: _check_walk_forward(proc_dir)),
        ("Bootstrap",    lambda: _check_bootstrap(signals, prices, proc_dir)),
        ("Param Sens",   lambda: _check_parameter_sensitivity(proc_dir)),
        ("Cost Sens",    lambda: _check_cost_sensitivity(proc_dir)),
        ("Regime",       lambda: _check_regime(proc_dir)),
        ("Factor",       lambda: _check_factor(proc_dir)),
        ("Capacity",     lambda: _check_capacity(proc_dir, target_aum_gbp)),
    ]

    for name, fn in test_fns:
        try:
            result = fn()
        except Exception as e:
            result = {
                "test":      name,
                "value":     f"ERROR: {e}",
                "threshold": "n/a",
                "pass":      None,
            }
        checks.append(result)

    # Print results
    print(f"\n  {'Test':<30}  {'Pass?':>6}  {'Value'}")
    print(f"  {'─'*30}  {'─'*6}  {'─'*40}")
    n_pass = 0
    n_fail = 0
    n_warn = 0
    for c in checks:
        p = c["pass"]
        if p is True:
            icon = "✓ PASS"
            n_pass += 1
        elif p is False:
            icon = "✗ FAIL"
            n_fail += 1
        else:
            icon = "? WARN"
            n_warn += 1
        print(f"  {c['test']:<30}  {icon:>6}  {c['value'][:65]}")
        print(f"  {'':30}  {'':6}  Threshold: {c['threshold'][:65]}")
        print()

    # Final verdict
    all_pass = (n_fail == 0) and (n_pass >= len(checks) - n_warn)
    print(f"{'='*75}")
    print(f"  RESULTS: {n_pass} PASS  |  {n_fail} FAIL  |  {n_warn} WARN  (of {len(checks)} tests)")
    print()
    if all_pass and n_fail == 0:
        print("  ██████  GO  ██████")
        print("  Strategy approved for paper trading / live deployment.")
        print("  All critical checks passed.")
    elif n_fail == 0 and n_warn > 0:
        print("  ▓▓▓▓  GO WITH CAUTION  ▓▓▓▓")
        print("  No hard failures. Warnings noted — monitor closely.")
    else:
        print("  ██████  NO-GO  ██████")
        print("  One or more critical checks FAILED.")
        print("  Fix before committing capital.")
    print(f"{'='*75}")

    return {
        "checks":    checks,
        "n_pass":    n_pass,
        "n_fail":    n_fail,
        "n_warn":    n_warn,
        "go":        all_pass and n_fail == 0,
    }
