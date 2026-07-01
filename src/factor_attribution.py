"""
Factor Attribution — Momentum → Top-5 → 200DMA → HRP

Runs OLS regressions against standard factor models:
  CAPM      : Mkt-RF
  FF3       : Mkt-RF, SMB, HML
  FF5       : Mkt-RF, SMB, HML, RMW, CMA
  FF5 + MOM : Mkt-RF, SMB, HML, RMW, CMA, Mom

Data: Fama-French monthly factors from Kenneth French's data library.
      HRL estimates use HAC standard errors (Newey-West, 3 lags).
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

PERIODS = 12


# ─────────────────────────────────────────────────────────────────────────────
# Factor data
# ─────────────────────────────────────────────────────────────────────────────
def download_factors(start: str = "2005-01-01") -> pd.DataFrame:
    """
    Download FF5 + Momentum monthly factors.
    Returns DataFrame with DatetimeIndex (month-end), values in decimal (not %).
    """
    import pandas_datareader.data as web

    ff5 = web.DataReader("F-F_Research_Data_5_Factors_2x3", "famafrench", start=start)[0]
    mom = web.DataReader("F-F_Momentum_Factor",             "famafrench", start=start)[0]

    factors = ff5.join(mom, how="inner") / 100  # % → decimal

    # Convert PeriodIndex → DatetimeIndex (month-end, no intra-day component)
    factors.index = factors.index.to_timestamp(how="S") + pd.offsets.MonthEnd(0)

    # Rename for clarity
    factors = factors.rename(columns={"Mom": "MOM"})
    return factors


# ─────────────────────────────────────────────────────────────────────────────
# Single OLS regression
# ─────────────────────────────────────────────────────────────────────────────
def _run_ols(
    excess_ret:   pd.Series,
    factors:      pd.DataFrame,
    factor_cols:  list[str],
    model_name:   str,
) -> dict:
    """OLS with HAC standard errors (Newey-West, 3 lags)."""
    common = excess_ret.index.intersection(factors.index)
    y = excess_ret.loc[common].dropna()
    X = sm.add_constant(factors.loc[y.index, factor_cols])
    fit = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 3})

    alpha_mo = float(fit.params["const"])
    return {
        "model_name":    model_name,
        "factor_cols":   factor_cols,
        "n_obs":         len(y),
        "alpha_monthly": alpha_mo,
        "alpha_annual":  alpha_mo * PERIODS,
        "alpha_tstat":   float(fit.tvalues["const"]),
        "alpha_pval":    float(fit.pvalues["const"]),
        "r2":            float(fit.rsquared),
        "adj_r2":        float(fit.rsquared_adj),
        "params":        fit.params,
        "tvalues":       fit.tvalues,
        "pvalues":       fit.pvalues,
        "conf_int":      fit.conf_int(),
        "fit":           fit,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Rolling alpha (CAPM)
# ─────────────────────────────────────────────────────────────────────────────
def rolling_alpha(
    excess_ret: pd.Series,
    factors:    pd.DataFrame,
    window:     int = 24,
) -> pd.Series:
    """Rolling CAPM alpha (annualised)."""
    common = excess_ret.index.intersection(factors.index)
    y  = excess_ret.loc[common].dropna()
    mkt = factors.loc[y.index, "Mkt-RF"]

    alphas = {}
    for i in range(window, len(y) + 1):
        yi   = y.iloc[i - window: i]
        xi   = sm.add_constant(mkt.iloc[i - window: i])
        try:
            fit  = sm.OLS(yi, xi).fit()
            alphas[y.index[i - 1]] = float(fit.params["const"]) * PERIODS
        except Exception:
            alphas[y.index[i - 1]] = np.nan

    return pd.Series(alphas)


# ─────────────────────────────────────────────────────────────────────────────
# Rolling betas (FF5 + MOM)
# ─────────────────────────────────────────────────────────────────────────────
def rolling_betas(
    excess_ret:  pd.Series,
    factors:     pd.DataFrame,
    factor_cols: list[str],
    window:      int = 36,
) -> pd.DataFrame:
    """Rolling OLS betas."""
    common = excess_ret.index.intersection(factors.index)
    y   = excess_ret.loc[common].dropna()
    X_f = factors.loc[y.index, factor_cols]

    rows = {}
    for i in range(window, len(y) + 1):
        yi = y.iloc[i - window: i]
        Xi = sm.add_constant(X_f.iloc[i - window: i])
        try:
            fit = sm.OLS(yi, Xi).fit()
            rows[y.index[i - 1]] = fit.params[factor_cols]
        except Exception:
            rows[y.index[i - 1]] = pd.Series(np.nan, index=factor_cols)

    return pd.DataFrame(rows).T


# ─────────────────────────────────────────────────────────────────────────────
# Return decomposition: alpha + factor contributions = excess return
# ─────────────────────────────────────────────────────────────────────────────
def return_decomposition(
    excess_ret:  pd.Series,
    factors:     pd.DataFrame,
    full_model:  dict,
) -> pd.DataFrame:
    """
    Annualised contribution of each factor and alpha to the total excess return.
    factor_contrib_i = beta_i * mean(factor_i) * 12
    """
    factor_cols = full_model["factor_cols"]
    params      = full_model["params"]
    common      = excess_ret.index.intersection(factors.index)

    factor_means  = factors.loc[common, factor_cols].mean() * PERIODS
    contributions = {}
    for f in factor_cols:
        contributions[f] = float(params[f]) * float(factor_means[f])

    contributions["Alpha"] = full_model["alpha_annual"]
    total_ann = excess_ret.loc[common].mean() * PERIODS

    return pd.Series(contributions), total_ann


# ─────────────────────────────────────────────────────────────────────────────
# Print tables
# ─────────────────────────────────────────────────────────────────────────────
FACTOR_LABELS = {
    "Mkt-RF": "Market (MKT-RF)",
    "SMB":    "Size (SMB)",
    "HML":    "Value (HML)",
    "RMW":    "Quality (RMW)",
    "CMA":    "Investment (CMA)",
    "MOM":    "Momentum (MOM)",
}


def _sig(p: float) -> str:
    return "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.10 else "(n.s.)"))


def print_factor_results(models: list[dict]) -> None:
    print(f"\n{'='*80}")
    print(f"  FACTOR ATTRIBUTION — Momentum → Top-5 → 200DMA → HRP")
    print(f"{'='*80}")

    # ── Model comparison ──
    print(f"\n  Model Comparison  (alpha = return unexplained by factors)")
    print(f"  {'Model':<16} {'N':>4}  {'Alpha/yr':>9}  {'t-stat':>7}  {'p-val':>7}  {'R²':>6}  {'AdjR²':>7}  Sig")
    print(f"  {'─'*16} {'─'*4}  {'─'*9}  {'─'*7}  {'─'*7}  {'─'*6}  {'─'*7}  {'─'*10}")
    for m in models:
        p = m["alpha_pval"]
        print(f"  {m['model_name']:<16} {m['n_obs']:>4}  "
              f"{m['alpha_annual']*100:>+8.2f}%  "
              f"{m['alpha_tstat']:>7.3f}  "
              f"{p:>7.4f}  "
              f"{m['r2']*100:>5.1f}%  "
              f"{m['adj_r2']*100:>6.1f}%  {_sig(p)}")

    # ── Full model factor loadings ──
    full = models[-1]
    print(f"\n  Factor Loadings — {full['model_name']}")
    print(f"  {'Factor':<20} {'Beta':>7}  {'t-stat':>7}  {'p-val':>7}  {'95% CI':>20}  Sig  Interpretation")
    print(f"  {'─'*20} {'─'*7}  {'─'*7}  {'─'*7}  {'─'*20}  {'─'*4}  {'─'*30}")

    interp = {
        "Mkt-RF": lambda b: f"{'Low' if b<0.3 else 'Moderate' if b<0.6 else 'High'} market exposure",
        "SMB":    lambda b: f"{'Small-cap' if b>0.1 else 'Large-cap' if b<-0.1 else 'Cap-neutral'} tilt",
        "HML":    lambda b: f"{'Value' if b>0.1 else 'Growth' if b<-0.1 else 'Style-neutral'} tilt",
        "RMW":    lambda b: f"{'Quality/profitability' if b>0.1 else 'Junk' if b<-0.1 else 'Profitability-neutral'} tilt",
        "CMA":    lambda b: f"{'Conservative' if b>0.1 else 'Aggressive invest.' if b<-0.1 else 'Invest.-neutral'} tilt",
        "MOM":    lambda b: f"{'Momentum-driven' if b>0.2 else 'Contrarian' if b<-0.1 else 'Momentum-neutral'}",
    }

    for f in full["factor_cols"]:
        beta = float(full["params"][f])
        t    = float(full["tvalues"][f])
        p    = float(full["pvalues"][f])
        lo   = float(full["conf_int"].loc[f, 0])
        hi   = float(full["conf_int"].loc[f, 1])
        label = FACTOR_LABELS.get(f, f)
        desc  = interp[f](beta) if f in interp else ""
        print(f"  {label:<20} {beta:>7.3f}  {t:>7.3f}  {p:>7.4f}  [{lo:>7.3f}, {hi:>7.3f}]  "
              f"{_sig(p):<4}  {desc}")

    print(f"  {'Alpha':<20} {full['alpha_annual']*100:>+6.2f}%/yr  "
          f"{full['alpha_tstat']:>7.3f}  {full['alpha_pval']:>7.4f}  "
          f"{'  True strategy alpha':>22}  {_sig(full['alpha_pval'])}")
    print(f"{'='*80}")


def print_decomposition(
    decomp: pd.Series,
    total_ann: float,
    rf_ann: float,
) -> None:
    print(f"\n  Return Decomposition (annualised):")
    print(f"  {'Source':<24} {'Contribution':>14}  {'% of excess ret':>16}")
    print(f"  {'─'*24} {'─'*14}  {'─'*16}")
    excess_ann = total_ann - rf_ann
    for src, val in decomp.items():
        pct = val / excess_ann * 100 if abs(excess_ann) > 1e-6 else np.nan
        flag = ""
        if src == "Alpha":
            flag = " ← pure alpha"
        elif src == "Mkt-RF":
            flag = " ← market beta"
        elif src == "MOM":
            flag = " ← momentum premium"
        print(f"  {FACTOR_LABELS.get(src, src):<24} {val*100:>+13.2f}%  {pct:>15.1f}%{flag}")
    print(f"  {'─'*24} {'─'*14}")
    print(f"  {'Total excess return':<24} {excess_ann*100:>+13.2f}%")
    print(f"  {'Risk-free rate':<24} {rf_ann*100:>+13.2f}%")
    print(f"  {'Gross return':<24} {total_ann*100:>+13.2f}%")


# ─────────────────────────────────────────────────────────────────────────────
# Top-level runner
# ─────────────────────────────────────────────────────────────────────────────
def run_factor_attribution(
    prices:   pd.DataFrame,
    returns:  pd.DataFrame,
    proc_dir: str = "data/processed",
) -> dict:
    from src.backtest import load_backtest_returns
    from src.data import load_risk_free_rate

    rf_daily   = load_risk_free_rate(proc_dir)
    rf_monthly = (1 + rf_daily).resample("ME").prod() - 1

    bt_rets   = load_backtest_returns(proc_dir)
    hrp_ret   = bt_rets["D: HRP"].dropna()

    factors   = download_factors()

    # Excess return = strategy return - risk-free rate
    excess_ret = (hrp_ret - rf_monthly.reindex(hrp_ret.index).fillna(0)).dropna()

    rf_ann  = rf_monthly.reindex(excess_ret.index).mean() * PERIODS
    total_ann = hrp_ret.reindex(excess_ret.index).mean() * PERIODS

    # Four models
    MODELS = [
        ("CAPM",       ["Mkt-RF"]),
        ("FF3",        ["Mkt-RF", "SMB", "HML"]),
        ("FF5",        ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]),
        ("FF5 + MOM",  ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "MOM"]),
    ]

    model_results = [_run_ols(excess_ret, factors, cols, name) for name, cols in MODELS]

    print_factor_results(model_results)

    full_model = model_results[-1]
    decomp, _ = return_decomposition(excess_ret, factors, full_model)
    print_decomposition(decomp, total_ann, rf_ann)

    roll_alpha_series = rolling_alpha(excess_ret, factors)
    roll_betas_df     = rolling_betas(excess_ret, factors, ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "MOM"])

    return {
        "models":       model_results,
        "factors":      factors,
        "excess_ret":   excess_ret,
        "hrp_ret":      hrp_ret,
        "rf_monthly":   rf_monthly,
        "rolling_alpha":roll_alpha_series,
        "rolling_betas":roll_betas_df,
        "decomp":       decomp,
        "total_ann":    total_ann,
        "rf_ann":       rf_ann,
    }
