"""
Five hypothesis tests for the ETF momentum strategy.

Test 1 — Information Coefficient     : does score predict next-month return?
Test 2 — Top vs Bottom               : does top group beat bottom group?
Test 3 — Trend filter                : does 200DMA filter improve risk-adjusted returns?
Test 4 — Portfolio construction      : equal / inv-vol / risk-parity / HRP comparison
Test 5 — Cost sensitivity            : does alpha survive realistic transaction costs?
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy.optimize as sco
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
N_TOP            = 4       # assets in top/bottom group
LOOKBACK_MOS     = 36      # months of history for covariance estimation
PERIODS_PER_YEAR = 12

CRISIS_PERIODS = {
    "GFC":    ("2008-09-01", "2009-03-31"),
    "COVID":  ("2020-02-01", "2020-04-30"),
    "2022BB": ("2022-01-01", "2022-10-31"),
}

# ─────────────────────────────────────────────────────────────────────────────
# Performance metrics
# ─────────────────────────────────────────────────────────────────────────────
def _metrics(ret: pd.Series, rf_annual: float = 0.0) -> dict:
    ann_ret = ret.mean() * PERIODS_PER_YEAR
    ann_vol = ret.std()  * np.sqrt(PERIODS_PER_YEAR)
    sharpe  = (ann_ret - rf_annual) / ann_vol if ann_vol > 0 else np.nan
    equity  = (1 + ret).cumprod()
    max_dd  = float((equity / equity.cummax() - 1).min())
    calmar  = ann_ret / abs(max_dd) if max_dd < 0 else np.nan
    return {
        "Ann Return %": round(ann_ret * 100, 2),
        "Ann Vol %":    round(ann_vol * 100, 2),
        "Sharpe":       round(float(sharpe), 3),
        "Max DD %":     round(max_dd * 100, 2),
        "Calmar":       round(float(calmar), 3) if not np.isnan(calmar) else np.nan,
        "N months":     len(ret),
    }

# ─────────────────────────────────────────────────────────────────────────────
# Forward returns (look-ahead free)
# ─────────────────────────────────────────────────────────────────────────────
def compute_me_forward_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """At each month-end t: return from t → t+1."""
    me = prices.resample("ME").last()
    return me.pct_change().shift(-1)

# ─────────────────────────────────────────────────────────────────────────────
# Backtest engine
# ─────────────────────────────────────────────────────────────────────────────
def _run_backtest(
    weights_df: pd.DataFrame,
    fwd_returns: pd.DataFrame,
    cost_one_way: float = 0.0,
) -> pd.Series:
    dates    = weights_df.index.intersection(fwd_returns.index)
    prev_w   = pd.Series(0.0, index=weights_df.columns)
    out      = {}

    for dt in dates:
        w   = weights_df.loc[dt].fillna(0.0)
        fwd = fwd_returns.loc[dt].reindex(w.index).fillna(0.0)

        turnover = (w - prev_w).abs().sum() / 2.0
        port_ret = float(w @ fwd) - turnover * cost_one_way
        out[dt]  = port_ret

        drifted = w * (1 + fwd)
        total   = drifted.sum()
        prev_w  = drifted / total if total > 0 else w

    return pd.Series(out)

# ─────────────────────────────────────────────────────────────────────────────
# Weighting schemes
# ─────────────────────────────────────────────────────────────────────────────
def _equal_weight(tickers: list) -> pd.Series:
    if not tickers:
        return pd.Series(dtype=float)
    return pd.Series(1.0 / len(tickers), index=tickers)


def _inv_vol_weight(tickers: list, vol_series: pd.Series) -> pd.Series:
    vols = vol_series.reindex(tickers).replace(0, np.nan).dropna()
    if vols.empty:
        return _equal_weight(tickers)
    w = 1.0 / vols
    return (w / w.sum()).reindex(tickers).fillna(0.0)


def _risk_parity_weight(tickers: list, cov: pd.DataFrame) -> pd.Series:
    """Equal Risk Contribution via SLSQP."""
    sub = cov.loc[tickers, tickers].values.astype(float)
    n   = len(tickers)

    def obj(w):
        sigma = float(np.sqrt(max(w @ sub @ w, 1e-20)))
        rc    = w * (sub @ w) / sigma
        avg   = sigma / n
        return float(np.sum((rc - avg) ** 2))

    w0     = np.ones(n) / n
    bounds = [(1e-4, 1.0)] * n
    cons   = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    res    = sco.minimize(obj, w0, method="SLSQP", bounds=bounds,
                          constraints=cons, options={"maxiter": 300, "ftol": 1e-10})
    w_out = np.maximum(res.x if res.success else w0, 0)
    w_out /= w_out.sum()
    return pd.Series(w_out, index=tickers)


# ─── HRP helpers ────────────────────────────────────────────────────────────
def _leaves(link: np.ndarray, cid: int, N: int) -> list:
    if cid < N:
        return [int(cid)]
    row = link[cid - N]
    return _leaves(link, int(row[0]), N) + _leaves(link, int(row[1]), N)


def _cluster_var(cov: np.ndarray, idx: list) -> float:
    if len(idx) == 1:
        return cov[idx[0], idx[0]]
    s = cov[np.ix_(idx, idx)]
    iv = 1.0 / np.diag(s)
    w  = iv / iv.sum()
    return float(w @ s @ w)


def _hrp_weight(tickers: list, rolling_ret: pd.DataFrame) -> pd.Series:
    available = [t for t in tickers if t in rolling_ret.columns]
    sub = rolling_ret[available].dropna()
    if sub.shape[0] < 5 or len(available) < 2:
        return _equal_weight(tickers)

    corr = sub.corr().clip(-0.9999, 0.9999)
    cov  = sub.cov().values
    N    = len(available)

    dist = np.sqrt(np.maximum(0.5 * (1 - corr.values), 0))
    np.fill_diagonal(dist, 0)
    link     = sch.linkage(squareform(dist), method="single")
    sort_ix  = _leaves(link.astype(int), 2 * N - 2, N)
    ordered  = [available[i] for i in sort_ix]

    weights  = pd.Series(1.0, index=ordered)
    clusters = [sort_ix]

    while clusters:
        nxt = []
        for cl in clusters:
            if len(cl) < 2:
                continue
            mid = len(cl) // 2
            lft, rgt = cl[:mid], cl[mid:]
            vl, vr   = _cluster_var(cov, lft), _cluster_var(cov, rgt)
            alpha     = 1.0 - vl / (vl + vr)
            weights[[available[i] for i in lft]] *= alpha
            weights[[available[i] for i in rgt]] *= (1.0 - alpha)
            nxt.extend([lft, rgt])
        clusters = [c for c in nxt if len(c) > 1]

    weights /= weights.sum()
    return weights.reindex(tickers).fillna(0.0)

# ─────────────────────────────────────────────────────────────────────────────
# Weight builder (one month)
# ─────────────────────────────────────────────────────────────────────────────
def _build_weights(
    score_row:   pd.Series,
    elig_row:    pd.Series,
    vol_row:     pd.Series,
    rolling_ret: pd.DataFrame,
    method:      str  = "equal",
    n_top:       int  = N_TOP,
    use_filter:  bool = True,
) -> pd.Series:
    all_tickers = score_row.index.tolist()
    w           = pd.Series(0.0, index=all_tickers)

    cands = score_row.dropna()
    if use_filter:
        cands = cands[elig_row.reindex(cands.index).fillna(False)]
    if cands.empty:
        return w

    selected = cands.nlargest(n_top).index.tolist()
    if not selected:
        return w

    if method == "equal":
        sub_w = _equal_weight(selected)
    elif method == "inv_vol":
        sub_w = _inv_vol_weight(selected, vol_row)
    elif method == "risk_parity":
        if len(selected) < 2 or rolling_ret.empty:
            sub_w = _equal_weight(selected)
        else:
            avail = [t for t in selected if t in rolling_ret.columns]
            if len(avail) < 2:
                sub_w = _equal_weight(selected)
            else:
                cov = rolling_ret[avail].dropna().cov()
                sub_w = _risk_parity_weight(avail, cov).reindex(selected).fillna(0.0)
    elif method == "hrp":
        sub_w = _hrp_weight(selected, rolling_ret) if len(selected) >= 2 else _equal_weight(selected)
    else:
        raise ValueError(f"Unknown method: {method}")

    w[selected] = sub_w
    return w

# ═════════════════════════════════════════════════════════════════════════════
# Test 1 — Information Coefficient
# ═════════════════════════════════════════════════════════════════════════════
def test_ic(signals: dict, fwd_returns: pd.DataFrame) -> dict:
    score = signals["score"]
    dates = score.index.intersection(fwd_returns.index)

    ic_map = {}
    for dt in dates:
        s  = score.loc[dt].dropna()
        fr = fwd_returns.loc[dt].reindex(s.index).dropna()
        common = s.index.intersection(fr.index)
        if len(common) < 4:
            continue
        rho, _ = stats.spearmanr(s[common], fr[common])
        ic_map[dt] = rho

    ic       = pd.Series(ic_map)
    t_stat, p_val = stats.ttest_1samp(ic.dropna(), 0)
    annual_ic = ic.groupby(ic.index.year).mean().rename("Mean IC")

    print(f"\n{'─'*45}")
    print("Test 1 — Information Coefficient")
    print(f"{'─'*45}")
    print(f"  Mean IC  : {ic.mean():.4f}")
    print(f"  IC Std   : {ic.std():.4f}")
    print(f"  t-stat   : {t_stat:.3f}  (p={p_val:.4f})")
    print(f"  IC > 0   : {(ic > 0).mean()*100:.1f}%")
    print(f"  N obs    : {len(ic)}")
    print("\n  Annual IC:")
    print(annual_ic.round(4).to_string())

    return {
        "ic_series": ic, "annual_ic": annual_ic,
        "mean_ic": ic.mean(), "ic_std": ic.std(),
        "t_stat": t_stat, "p_value": p_val,
    }

# ═════════════════════════════════════════════════════════════════════════════
# Test 2 — Top vs Bottom
# ═════════════════════════════════════════════════════════════════════════════
def test_top_bottom(signals: dict, fwd_returns: pd.DataFrame, n: int = N_TOP) -> dict:
    score = signals["score"]
    dates = score.index.intersection(fwd_returns.index)

    top_w = pd.DataFrame(0.0, index=dates, columns=score.columns)
    bot_w = pd.DataFrame(0.0, index=dates, columns=score.columns)

    for dt in dates:
        s = score.loc[dt].dropna()
        if len(s) < 2 * n:
            continue
        top_w.loc[dt, s.nlargest(n).index]  = 1.0 / n
        bot_w.loc[dt, s.nsmallest(n).index] = 1.0 / n

    top_ret  = _run_backtest(top_w, fwd_returns).dropna()
    bot_ret  = _run_backtest(bot_w, fwd_returns).dropna()
    spread   = (top_ret - bot_ret).dropna()
    t_stat, p_val = stats.ttest_1samp(spread, 0)

    print(f"\n{'─'*45}")
    print(f"Test 2 — Top vs Bottom (N={n} each)")
    print(f"{'─'*45}")
    summary = pd.DataFrame({
        "Top":    _metrics(top_ret),
        "Bottom": _metrics(bot_ret),
        "Spread": _metrics(spread),
    })
    print(summary.to_string())
    print(f"\n  H0: spread=0  →  t={t_stat:.3f}, p={p_val:.4f}")

    return {
        "top_returns": top_ret, "bot_returns": bot_ret, "spread": spread,
        "metrics": summary,
        "equity_top": (1 + top_ret).cumprod(),
        "equity_bot": (1 + bot_ret).cumprod(),
        "t_stat": t_stat, "p_value": p_val,
    }

# ═════════════════════════════════════════════════════════════════════════════
# Test 3 — Trend filter
# ═════════════════════════════════════════════════════════════════════════════
def test_trend_filter(signals: dict, fwd_returns: pd.DataFrame, n: int = N_TOP) -> dict:
    score = signals["score"]
    elig  = signals["eligible"]
    vol   = signals["vol_60d"]
    dates = score.index.intersection(fwd_returns.index)

    results = {}
    for label, use_f in [("No filter", False), ("200DMA filter", True)]:
        w_df = pd.DataFrame(0.0, index=dates, columns=score.columns)
        for dt in dates:
            w_df.loc[dt] = _build_weights(
                score.loc[dt], elig.loc[dt], vol.loc[dt],
                rolling_ret=pd.DataFrame(), method="equal",
                n_top=n, use_filter=use_f,
            )
        ret = _run_backtest(w_df, fwd_returns).dropna()
        results[label] = {"returns": ret, "metrics": _metrics(ret)}

    crisis = {}
    for period, (s, e) in CRISIS_PERIODS.items():
        row = {}
        for label in results:
            r = results[label]["returns"]
            sub = r[(r.index >= s) & (r.index <= e)]
            row[label] = round(((1 + sub).prod() - 1) * 100, 2) if len(sub) else np.nan
        crisis[period] = row

    print(f"\n{'─'*45}")
    print("Test 3 — Trend Filter")
    print(f"{'─'*45}")
    print(pd.DataFrame({k: v["metrics"] for k, v in results.items()}).to_string())
    print("\n  Crisis total returns (%):")
    print(pd.DataFrame(crisis).T.to_string())

    return {
        "results":        results,
        "crisis_returns": pd.DataFrame(crisis).T,
    }

# ═════════════════════════════════════════════════════════════════════════════
# Test 4 — Portfolio construction
# ═════════════════════════════════════════════════════════════════════════════
def test_portfolio_construction(
    signals: dict,
    fwd_returns: pd.DataFrame,
    me_returns: pd.DataFrame,
    n: int = N_TOP,
) -> dict:
    score = signals["score"]
    elig  = signals["eligible"]
    vol   = signals["vol_60d"]
    dates = score.index.intersection(fwd_returns.index)

    results = {}
    for method in ["equal", "inv_vol", "risk_parity", "hrp"]:
        w_df = pd.DataFrame(0.0, index=dates, columns=score.columns)
        for dt in dates:
            hist = me_returns[me_returns.index < dt].tail(LOOKBACK_MOS)
            w_df.loc[dt] = _build_weights(
                score.loc[dt], elig.loc[dt], vol.loc[dt],
                rolling_ret=hist, method=method, n_top=n, use_filter=True,
            )
        ret = _run_backtest(w_df, fwd_returns).dropna()
        results[method] = {"returns": ret, "weights": w_df, "metrics": _metrics(ret)}

    print(f"\n{'─'*45}")
    print("Test 4 — Portfolio Construction")
    print(f"{'─'*45}")
    print(pd.DataFrame({k: v["metrics"] for k, v in results.items()}).to_string())

    return results

# ═════════════════════════════════════════════════════════════════════════════
# Test 5 — Cost sensitivity
# ═════════════════════════════════════════════════════════════════════════════
def test_cost_sensitivity(
    signals: dict,
    fwd_returns: pd.DataFrame,
    me_returns: pd.DataFrame,
    costs_bps: list = [5, 10, 25, 50],
    n: int = N_TOP,
) -> dict:
    score = signals["score"]
    elig  = signals["eligible"]
    vol   = signals["vol_60d"]
    dates = score.index.intersection(fwd_returns.index)

    # Build weights once (equal-weight + 200DMA filter)
    w_df = pd.DataFrame(0.0, index=dates, columns=score.columns)
    for dt in dates:
        w_df.loc[dt] = _build_weights(
            score.loc[dt], elig.loc[dt], vol.loc[dt],
            rolling_ret=pd.DataFrame(), method="equal",
            n_top=n, use_filter=True,
        )

    results = {}
    for bps in costs_bps:
        ret = _run_backtest(w_df, fwd_returns, cost_one_way=bps / 10_000).dropna()
        results[f"{bps}bp"] = {"returns": ret, "metrics": _metrics(ret)}

    # Turnover
    prev_w = pd.Series(0.0, index=w_df.columns)
    turnovers = []
    for dt in dates:
        w   = w_df.loc[dt].fillna(0.0)
        turnovers.append((w - prev_w).abs().sum() / 2.0)
        fwd = fwd_returns.loc[dt].reindex(w.index).fillna(0.0)
        drifted = w * (1 + fwd)
        s = drifted.sum()
        prev_w = drifted / s if s > 0 else w

    avg_to = float(np.mean(turnovers))

    print(f"\n{'─'*45}")
    print("Test 5 — Cost Sensitivity")
    print(f"{'─'*45}")
    print(pd.DataFrame({k: v["metrics"] for k, v in results.items()}).to_string())
    print(f"\n  Avg monthly one-way turnover : {avg_to*100:.1f}%")
    print(f"  Annualised two-way turnover  : {avg_to*2*12*100:.0f}%")

    return {"results": results, "avg_turnover": avg_to}

# ═════════════════════════════════════════════════════════════════════════════
# Master runner
# ═════════════════════════════════════════════════════════════════════════════
def run_all_tests(prices: pd.DataFrame, returns: pd.DataFrame, signals: dict) -> dict:
    me_returns  = prices.resample("ME").last().pct_change()
    fwd_returns = compute_me_forward_returns(prices)

    print("\n" + "=" * 50)
    print("  MOMENTUM STRATEGY — HYPOTHESIS TESTS")
    print("=" * 50)

    t1 = test_ic(signals, fwd_returns)
    t2 = test_top_bottom(signals, fwd_returns)
    t3 = test_trend_filter(signals, fwd_returns)
    t4 = test_portfolio_construction(signals, fwd_returns, me_returns)
    t5 = test_cost_sensitivity(signals, fwd_returns, me_returns)

    print("\n" + "=" * 50 + "\n  ALL TESTS COMPLETE\n" + "=" * 50)
    return {"t1_ic": t1, "t2_top_bottom": t2, "t3_trend": t3,
            "t4_construction": t4, "t5_costs": t5}
