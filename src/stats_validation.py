"""
Statistical validation of the momentum strategy.

Block A — IC deep-dive     : Spearman IC, ICIR, yearly stability, bootstrap CI
Block B — Bootstrap         : 10,000 resamplings → Sharpe / CAGR / MaxDD CI
Block C — Monte Carlo       : 10,000 permutations of return sequence
Block D — Reality check     : remove 10 / 20 / 30 % of trades randomly

All simulations use a seeded RNG (seed=42) for reproducibility.
"""

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
PERIODS        = 12          # monthly data
N_BOOT         = 10_000
N_MC           = 10_000
N_REALITY      = 5_000
REMOVAL_RATES  = [0.10, 0.20, 0.30]
RNG            = np.random.default_rng(42)


# ─────────────────────────────────────────────────────────────────────────────
# Vectorised metric helpers (no DataFrame overhead — fast in loops)
# ─────────────────────────────────────────────────────────────────────────────
def _sharpe_arr(ret: np.ndarray, rf: float = 0.0) -> float:
    excess = ret - rf
    vol = excess.std()
    return float(excess.mean() * PERIODS / (vol * np.sqrt(PERIODS))) if vol > 0 else np.nan


def _cagr_arr(ret: np.ndarray) -> float:
    n_yr = len(ret) / PERIODS
    return float((1 + ret).prod() ** (1 / n_yr) - 1) if n_yr > 0 else np.nan


def _maxdd_arr(ret: np.ndarray) -> float:
    eq  = np.cumprod(1 + ret)
    mdd = np.minimum.accumulate(eq / np.maximum.accumulate(eq)) - 1
    return float(mdd.min())


def _sortino_arr(ret: np.ndarray, rf: float = 0.0) -> float:
    excess = ret - rf
    down   = excess[excess < 0]
    if len(down) < 2:
        return np.nan
    dd_vol = down.std() * np.sqrt(PERIODS)
    return float(excess.mean() * PERIODS / dd_vol) if dd_vol > 0 else np.nan


# ─────────────────────────────────────────────────────────────────────────────
# Block A — IC Analysis
# ─────────────────────────────────────────────────────────────────────────────
def ic_full_analysis(
    signals:     dict,
    fwd_returns: pd.DataFrame,
    n_boot:      int = N_BOOT,
) -> dict:
    """
    Monthly Spearman IC between momentum score and next-period return.
    Returns: ic_series, annual_ic, ICIR, t_stat, p_value, bootstrap CI,
             normality tests, autocorrelation.
    """
    score = signals["score"]
    dates = score.index.intersection(fwd_returns.index)

    ic_map = {}
    for dt in dates:
        s  = score.loc[dt].dropna()
        fr = fwd_returns.loc[dt].reindex(s.index).dropna()
        common = s.index.intersection(fr.index)
        if len(common) < 4:
            continue
        rho, _ = scipy_stats.spearmanr(s[common].values, fr[common].values)
        ic_map[dt] = float(rho)

    ic  = pd.Series(ic_map).dropna()
    n   = len(ic)
    mu  = ic.mean()
    sig = ic.std()

    # t-stat, p-value (two-sided)
    t_stat   = mu / (sig / np.sqrt(n))
    p_value  = 2 * scipy_stats.t.sf(abs(t_stat), df=n - 1)

    # ICIR (monthly version — analogous to Sharpe of IC)
    icir_monthly = mu / sig if sig > 0 else np.nan
    icir_annual  = icir_monthly * np.sqrt(PERIODS) if not np.isnan(icir_monthly) else np.nan

    # Annual IC
    annual_ic = ic.groupby(ic.index.year).mean()

    # Normality tests
    jb_stat, jb_p      = scipy_stats.jarque_bera(ic.values)
    _, sw_p            = scipy_stats.shapiro(ic.values[:500])   # Shapiro capped at 5000
    ac1                = float(ic.autocorr(lag=1))

    # Bootstrap CI for mean IC
    ic_arr  = ic.values
    boot_mu = np.array([
        RNG.choice(ic_arr, size=n, replace=True).mean()
        for _ in range(n_boot)
    ])
    ci_lo, ci_hi = np.percentile(boot_mu, [2.5, 97.5])

    result = {
        "ic_series":      ic,
        "annual_ic":      annual_ic,
        "n_obs":          n,
        "mean_ic":        round(mu, 5),
        "ic_std":         round(sig, 5),
        "icir_monthly":   round(icir_monthly, 4),
        "icir_annual":    round(icir_annual, 4),
        "t_stat":         round(t_stat, 4),
        "p_value":        round(p_value, 6),
        "ic_gt0_pct":     round((ic > 0).mean() * 100, 1),
        "boot_mean_ic_lo": round(ci_lo, 5),
        "boot_mean_ic_hi": round(ci_hi, 5),
        "normality_jb_p":  round(jb_p, 4),
        "normality_sw_p":  round(sw_p, 4),
        "autocorr_lag1":   round(ac1, 4),
        "boot_mu_dist":    boot_mu,
    }

    print(f"\n{'─'*55}")
    print("Block A — Information Coefficient")
    print(f"{'─'*55}")
    print(f"  N obs            : {n}")
    print(f"  Mean IC          : {mu:.5f}")
    print(f"  IC Std           : {sig:.5f}")
    print(f"  ICIR (monthly)   : {icir_monthly:.4f}")
    print(f"  ICIR (annual)    : {icir_annual:.4f}")
    print(f"  t-stat           : {t_stat:.4f}")
    print(f"  p-value          : {p_value:.6f}  {'***' if p_value<0.01 else '**' if p_value<0.05 else '*' if p_value<0.10 else ''}")
    print(f"  IC > 0           : {result['ic_gt0_pct']:.1f}%")
    print(f"  95% CI (boot)    : [{ci_lo:.5f}, {ci_hi:.5f}]")
    print(f"  Jarque-Bera p    : {jb_p:.4f}  {'(non-normal)' if jb_p<0.05 else '(normal-ish)'}")
    print(f"  Shapiro-Wilk p   : {sw_p:.4f}")
    print(f"  Autocorr (lag-1) : {ac1:.4f}  {'(significant)' if abs(ac1)>0.1 else '(negligible)'}")
    print(f"\n  Annual IC (mean per year):")
    for yr, v in annual_ic.items():
        bar = "█" * int(abs(v) * 60) if abs(v) < 1 else "█" * 20
        sign = "+" if v >= 0 else "-"
        print(f"    {yr}: {sign}{abs(v):.4f}  {bar}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Block B — Bootstrap confidence intervals
# ─────────────────────────────────────────────────────────────────────────────
def bootstrap_ci(
    returns:    pd.Series,
    rf_monthly: pd.Series | None = None,
    n_sims:     int = N_BOOT,
) -> dict:
    """
    Resample monthly returns 10,000× with replacement.
    Reports 95% CI for Sharpe, CAGR, MaxDD and Sortino.
    """
    n      = len(returns)
    ret_a  = returns.values.astype(float)
    rf_val = rf_monthly.reindex(returns.index).fillna(0).mean() if rf_monthly is not None else 0.0

    # Draw all bootstrap samples at once: (n_sims × n)
    idx       = RNG.integers(0, n, size=(n_sims, n))
    boot_rets = ret_a[idx]           # shape: (n_sims, n)

    # Vectorised Sharpe and CAGR
    excess    = boot_rets - rf_val
    mu_boot   = excess.mean(axis=1) * PERIODS
    vol_boot  = boot_rets.std(axis=1) * np.sqrt(PERIODS)
    sharpes   = np.where(vol_boot > 0, mu_boot / vol_boot, np.nan)
    cagrs     = (1 + boot_rets.mean(axis=1)) ** PERIODS - 1

    # Max DD — must loop (path-dependent)
    max_dds = np.empty(n_sims)
    for i in range(n_sims):
        eq  = np.cumprod(1 + boot_rets[i])
        max_dds[i] = float((eq / np.maximum.accumulate(eq)).min()) - 1

    # Sortino — partial vectorisation
    sortinos = np.empty(n_sims)
    for i in range(n_sims):
        sortinos[i] = _sortino_arr(boot_rets[i], rf=rf_val)

    def _ci(arr):
        a = np.asarray(arr)[~np.isnan(arr)]
        return {
            "mean":   round(float(a.mean()), 4),
            "lo95":   round(float(np.percentile(a, 2.5)), 4),
            "hi95":   round(float(np.percentile(a, 97.5)), 4),
            "p_gt0":  round(float((a > 0).mean()), 4),
        }

    obs_sh  = _sharpe_arr(ret_a, rf=rf_val)
    obs_cg  = _cagr_arr(ret_a)
    obs_md  = _maxdd_arr(ret_a)
    obs_so  = _sortino_arr(ret_a, rf=rf_val)

    ci_sh = _ci(sharpes)
    ci_cg = _ci(cagrs)
    ci_md = _ci(max_dds)
    ci_so = _ci(sortinos)

    print(f"\n{'─'*55}")
    print(f"Block B — Bootstrap  ({n_sims:,} samples, 95% CI)")
    print(f"{'─'*55}")
    print(f"  {'Metric':<12} {'Observed':>10} {'Mean boot':>10} {'95% CI':>22}  {'P>0':>6}")
    print(f"  {'─'*12} {'─'*10} {'─'*10} {'─'*22}  {'─'*6}")
    for label, obs, ci in [
        ("Sharpe",  obs_sh, ci_sh),
        ("CAGR",    obs_cg, ci_cg),
        ("MaxDD",   obs_md, ci_md),
        ("Sortino", obs_so, ci_so),
    ]:
        print(f"  {label:<12} {obs:>10.4f} {ci['mean']:>10.4f} "
              f"  [{ci['lo95']:>8.4f}, {ci['hi95']:>8.4f}]  {ci['p_gt0']:>6.3f}")

    return {
        "observed":  {"sharpe": obs_sh, "cagr": obs_cg, "maxdd": obs_md, "sortino": obs_so},
        "ci_sharpe": ci_sh, "ci_cagr": ci_cg, "ci_maxdd": ci_md, "ci_sortino": ci_so,
        "dist":      {"sharpe": sharpes, "cagr": cagrs, "maxdd": max_dds, "sortino": sortinos},
        "n_sims":    n_sims,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Block C — Monte Carlo  (two sub-tests)
# ─────────────────────────────────────────────────────────────────────────────
def monte_carlo_permutation(
    returns:     pd.Series,
    signals:     dict,
    fwd_returns: pd.DataFrame,
    rf_monthly:  pd.Series | None = None,
    spy_returns: pd.Series | None = None,
    n_sims:      int = N_MC,
    n_top:       int = 5,
) -> dict:
    """
    Two complementary permutation tests.

    C1 — Temporal shuffle: permute the return ORDER → MaxDD distribution.
         Sharpe is order-invariant so this is NOT the right test for Sharpe.
         Tests: "Is our low drawdown due to lucky timing?"

    C2 — Cross-sectional: randomly SELECT assets at each month (ignore score).
         Sharpe DOES vary because different assets are picked.
         Tests: "Does the momentum score add selection skill vs random picking?"
    """
    n      = len(returns)
    ret_a  = returns.values.astype(float)
    rf_val = rf_monthly.reindex(returns.index).fillna(0).mean() if rf_monthly is not None else 0.0
    obs_sh = _sharpe_arr(ret_a, rf=rf_val)
    obs_md = _maxdd_arr(ret_a)

    # ── C1: Temporal permutation → MaxDD distribution ──────────────────────
    idx_perm = np.argsort(RNG.random(size=(n_sims, n)), axis=1)
    perm_rets = ret_a[idx_perm]   # (n_sims, n)
    perm_mdd  = np.array([
        float((np.cumprod(1 + perm_rets[i]) /
               np.maximum.accumulate(np.cumprod(1 + perm_rets[i]))).min()) - 1
        for i in range(n_sims)
    ])
    # p-value: what fraction of random orderings have MaxDD >= observed (i.e. less severe)?
    p_mdd = float((perm_mdd >= obs_md).mean())

    # ── C2: Cross-sectional permutation → Sharpe distribution ──────────────
    score = signals["score"]
    elig  = signals["eligible"]
    dates = score.index.intersection(fwd_returns.index)

    # Pre-compute eligible forward-return arrays for every date
    elig_rets_list   = []
    actual_rets_list = []
    for dt in dates:
        s  = score.loc[dt].dropna()
        e  = elig.loc[dt].reindex(s.index).fillna(False)
        fr = fwd_returns.loc[dt].reindex(s.index)
        # Build a joint DataFrame so score and fwd_return are perfectly aligned
        joint = pd.DataFrame({"score": s, "elig": e, "fwd": fr}).dropna(subset=["fwd"])
        joint = joint[joint["elig"].astype(bool)]
        er = joint["fwd"].values.astype(float)
        elig_rets_list.append(er)
        # Actual: equal-weight top-N by score (for fair comparison with random)
        K = len(er)
        if K == 0:
            actual_rets_list.append(0.0)
        elif K <= n_top:
            actual_rets_list.append(float(er.mean()))
        else:
            top_idx = np.argsort(joint["score"].values)[-n_top:]
            actual_rets_list.append(float(er[top_idx].mean()))

    actual_ew_ret = np.array(actual_rets_list)
    obs_sh_ew = _sharpe_arr(actual_ew_ret, rf=rf_val)

    # Vectorised random selection for each date
    sim_rets = np.zeros((n_sims, len(dates)))
    for j, er in enumerate(elig_rets_list):
        K = len(er)
        if K == 0:
            continue
        elif K <= n_top:
            sim_rets[:, j] = er.mean()
        else:
            # Random draw of n_top from K, for all sims at once
            rand_idx = np.argsort(RNG.random(size=(n_sims, K)), axis=1)[:, :n_top]
            sim_rets[:, j] = er[rand_idx].mean(axis=1)

    excess_sim = sim_rets - rf_val
    mu_sim  = excess_sim.mean(axis=1) * PERIODS
    vol_sim = sim_rets.std(axis=1) * np.sqrt(PERIODS)
    rand_sh = np.where(vol_sim > 0, mu_sim / vol_sim, np.nan)

    p_cs  = float((rand_sh >= obs_sh_ew).mean())
    pct_rank_cs = float((rand_sh < obs_sh_ew).mean())

    bench_sh = None
    if spy_returns is not None:
        b = spy_returns.reindex(returns.index).dropna().values.astype(float)
        bench_sh = _sharpe_arr(b, rf=rf_val)

    print(f"\n{'─'*60}")
    print(f"Block C — Monte Carlo  ({n_sims:,} sims each)")
    print(f"{'─'*60}")
    print(f"  C1 — Temporal permutation (MaxDD test):")
    print(f"    Observed MaxDD           : {obs_md*100:.2f}%")
    print(f"    Mean random MaxDD        : {perm_mdd.mean()*100:.2f}%")
    print(f"    P(random MaxDD ≥ observed): {p_mdd:.4f}  "
          f"{'*** (lucky timing excluded)' if p_mdd < 0.05 else '(timing may contribute)'}")
    print(f"\n  C2 — Cross-sectional permutation (signal skill test):")
    print(f"    Observed Sharpe (EW top-N)  : {obs_sh_ew:.4f}")
    print(f"    Mean random Sharpe          : {np.nanmean(rand_sh):.4f}")
    print(f"    Std random Sharpe           : {np.nanstd(rand_sh):.4f}")
    print(f"    Empirical p-value           : {p_cs:.6f}  "
          f"{'***' if p_cs<0.01 else '**' if p_cs<0.05 else '*' if p_cs<0.10 else '(n.s.)'}")
    print(f"    Percentile rank             : {pct_rank_cs*100:.2f}th percentile")
    if bench_sh:
        p_beat_spy = float((rand_sh >= bench_sh).mean())
        print(f"    SPY Sharpe                  : {bench_sh:.4f}")
        print(f"    P(random ≥ SPY Sharpe)      : {p_beat_spy:.4f}")

    return {
        "c1_temporal": {
            "observed_maxdd": obs_md,
            "perm_maxdd":     perm_mdd,
            "p_value":        p_mdd,
            "mean_random":    float(perm_mdd.mean()),
        },
        "c2_crosssectional": {
            "observed_sharpe_ew": obs_sh_ew,
            "rand_sharpes":       rand_sh,
            "p_value":            p_cs,
            "percentile_rank":    pct_rank_cs,
            "mean_random":        float(np.nanmean(rand_sh)),
            "std_random":         float(np.nanstd(rand_sh)),
            "benchmark_sharpe":   bench_sh,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Block D — Reality check (random trade removal)
# ─────────────────────────────────────────────────────────────────────────────
def reality_check(
    returns:        pd.Series,
    rf_monthly:     pd.Series | None = None,
    spy_returns:    pd.Series | None = None,
    removal_rates:  list = REMOVAL_RATES,
    n_sims:         int  = N_REALITY,
) -> dict:
    """
    Randomly set X% of months to risk-free return.
    Simulates missed signals, execution failures, data gaps.
    Asks: would the strategy still beat SPY if we missed that fraction of trades?
    """
    n      = len(returns)
    ret_a  = returns.values.astype(float)
    rf_a   = rf_monthly.reindex(returns.index).fillna(0).values if rf_monthly is not None else np.zeros(n)
    rf_val = rf_a.mean()

    obs_sh  = _sharpe_arr(ret_a, rf=rf_val)
    spy_sh  = _sharpe_arr(spy_returns.reindex(returns.index).fillna(0).values, rf=rf_val) if spy_returns is not None else None

    results = {}
    print(f"\n{'─'*55}")
    print(f"Block D — Reality Check  ({n_sims:,} sims per rate)")
    print(f"{'─'*55}")
    print(f"  Observed Sharpe   : {obs_sh:.4f}")
    if spy_sh:
        print(f"  SPY Sharpe        : {spy_sh:.4f}")
    print(f"\n  {'Removal':>10}  {'Mean Sh':>8}  {'5th pct':>8}  {'95th pct':>8}  "
          f"{'P>0':>6}  {'P>SPY':>7}  {'Still profitable':>16}")
    print(f"  {'─'*10}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*6}  {'─'*7}  {'─'*16}")

    for rate in removal_rates:
        n_rm = int(n * rate)
        shs, cagrs, mdds = [], [], []

        for _ in range(n_sims):
            mod      = ret_a.copy()
            rm_idx   = RNG.choice(n, size=n_rm, replace=False)
            mod[rm_idx] = rf_a[rm_idx]          # earn only rf on removed months
            shs.append(_sharpe_arr(mod, rf=rf_val))
            cagrs.append(_cagr_arr(mod))
            mdds.append(_maxdd_arr(mod))

        shs_a = np.array(shs)
        p_gt0    = float((shs_a > 0).mean())
        p_spy    = float((shs_a > spy_sh).mean()) if spy_sh else np.nan
        mean_sh  = float(np.nanmean(shs_a))
        lo5, hi95 = np.nanpercentile(shs_a, [5, 95])

        still_ok = "YES ✓" if lo5 > 0 else ("WEAK △" if mean_sh > 0 else "NO ✗")

        print(f"  {int(rate*100):>9}%  {mean_sh:>8.3f}  {lo5:>8.3f}  {hi95:>8.3f}  "
              f"{p_gt0:>6.3f}  {p_spy:>7.3f}  {still_ok:>16}")

        results[f"{int(rate*100)}pct"] = {
            "rate":       rate,
            "mean_sh":    mean_sh,
            "lo5":        lo5,
            "hi95":       hi95,
            "p_gt0":      p_gt0,
            "p_beat_spy": p_spy,
            "dist_sharpe":shs_a,
            "dist_cagr":  np.array(cagrs),
            "dist_maxdd": np.array(mdds),
        }

    return {
        "observed_sharpe": obs_sh,
        "spy_sharpe":      spy_sh,
        "results":         results,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Top-level runner
# ─────────────────────────────────────────────────────────────────────────────
def run_stats_validation(
    signals:  dict,
    prices:   pd.DataFrame,
    returns:  pd.DataFrame,
    proc_dir: str = "data/processed",
) -> dict:
    from src.data import load_risk_free_rate
    from src.backtest import load_backtest_returns
    from src.hypothesis_tests import compute_me_forward_returns

    rf_daily   = load_risk_free_rate(proc_dir)
    rf_monthly = (1 + rf_daily).resample("ME").prod() - 1

    bt_rets   = load_backtest_returns(proc_dir)
    hrp_ret   = bt_rets["D: HRP"].dropna()
    bench_col = "SPY" if "SPY" in prices.columns else prices.columns[0]
    spy_ret   = prices.resample("ME").last()[bench_col].pct_change().dropna()
    fwd       = compute_me_forward_returns(prices)

    print("\n" + "=" * 60)
    print("  STATISTICAL VALIDATION")
    print("=" * 60)

    out = {}
    out["ic"]      = ic_full_analysis(signals, fwd)
    out["boot"]    = bootstrap_ci(hrp_ret, rf_monthly)
    out["mc"]      = monte_carlo_permutation(
        hrp_ret, signals=signals, fwd_returns=fwd,
        rf_monthly=rf_monthly, spy_returns=spy_ret,
    )
    out["reality"] = reality_check(hrp_ret, rf_monthly, spy_returns=spy_ret)

    print(f"\n{'='*60}\n  VALIDATION COMPLETE\n{'='*60}")
    return out
