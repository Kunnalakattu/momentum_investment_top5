"""
Robustness analysis — vary one parameter at a time, hold all others at base.

Tests:
  1. Momentum windows   — 5 weighting combinations
  2. Trend filter DMA   — 150 / 180 / 200 / 250
  3. Covariance window  — 6 / 12 / 24 / 36 months
  4. Rebalance freq     — Monthly / 6-week / Quarterly
  5. Universe size      — 10 / 15 (base) / 20 / 30 ETFs

All tests use HRP + Top-5 + 10bp cost unless the parameter under test changes it.
Output: a robustness matrix showing Sharpe, Sortino, Calmar, Max DD.
"""

import warnings
import numpy as np
import pandas as pd
import yfinance as yf
import os
from pathlib import Path

from src.backtest import compute_metrics, _run_backtest_vt, COST_ONE_WAY, LEVERAGE_CAP
from src.hypothesis_tests import _hrp_weight, _inv_vol_weight, LOOKBACK_MOS
from src.data import clean_prices, estimate_bid_ask_spreads, load_prices, compute_returns

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
REPORT_COLS = ["Sharpe", "Sortino", "Calmar", "Max DD %", "Ann Return %", "Ann Vol %"]
BASE_LABEL  = "BASE"
N_TOP       = 5
METHOD      = "hrp"
START_DATE  = "2010-01-04"

# UCITS universe subsets for robustness testing (tickers without .L suffix)
UNIVERSE_10 = ["VUSA", "EQQQ", "IWDA", "EIMI", "SGLN", "AGGH", "IGLA", "IWDP", "EUNK", "CMOD"]
UNIVERSE_15 = UNIVERSE_10 + ["VWRP", "IJPN", "WNRG", "BNKS", "JNKS"]
UNIVERSE_20_ADD = ["QDVE", "HEAL", "NDIA", "XCS6", "R2US"]
UNIVERSE_30_ADD = UNIVERSE_20_ADD + ["WSML", "LTMC", "LQDS", "JPEA", "IEAC", "JEUG", "QDIV", "QDVR", "QDVS", "QNTG"]


# ─────────────────────────────────────────────────────────────────────────────
# Core: compute daily signals with custom parameters
# ─────────────────────────────────────────────────────────────────────────────
def _daily_signals(
    prices:     pd.DataFrame,
    returns:    pd.DataFrame,
    w3:         float = 0.2,
    w6:         float = 0.3,
    w12:        float = 0.5,
    dma_period: int   = 200,
) -> dict:
    dma     = prices.rolling(dma_period, min_periods=int(dma_period * 0.8)).mean()
    mom_3m  = prices / prices.shift(63)  - 1
    mom_6m  = prices / prices.shift(126) - 1
    mom_12m = prices / prices.shift(252) - 1
    score   = w3 * mom_3m + w6 * mom_6m + w12 * mom_12m
    elig    = prices > dma
    vol     = returns.rolling(60, min_periods=50).std() * np.sqrt(252)
    return {"score": score, "eligible": elig, "vol_60d": vol}


def _resample_signals(daily: dict, freq: str) -> dict:
    return {k: v.resample(freq).last() for k, v in daily.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Core: forward returns at arbitrary rebalance dates
# ─────────────────────────────────────────────────────────────────────────────
def _fwd_returns(prices: pd.DataFrame, rebal_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Return between each consecutive pair of rebalance dates."""
    px = prices.reindex(rebal_dates, method="ffill")
    rows = {}
    for i in range(len(px) - 1):
        rows[px.index[i]] = px.iloc[i + 1] / px.iloc[i] - 1
    return pd.DataFrame(rows).T


def _rebal_dates_6w(index: pd.DatetimeIndex, warmup: int = 300) -> pd.DatetimeIndex:
    """Last trading day at ~6-week intervals (≥42 calendar days apart)."""
    dates, last = [], index[warmup]
    for dt in index[warmup:]:
        if (dt - last).days >= 42:
            dates.append(dt)
            last = dt
    return pd.DatetimeIndex(dates)


# ─────────────────────────────────────────────────────────────────────────────
# Core: weight builder at custom rebalance dates
# ─────────────────────────────────────────────────────────────────────────────
def _build_weight_df(
    signals:     dict,
    me_returns:  pd.DataFrame,
    n_top:       int  = N_TOP,
    lookback:    int  = LOOKBACK_MOS,
) -> pd.DataFrame:
    score = signals["score"]
    elig  = signals["eligible"]
    vol   = signals["vol_60d"]
    dates = score.index
    tickers = score.columns.tolist()
    w_df  = pd.DataFrame(0.0, index=dates, columns=tickers)

    for dt in dates:
        s_row = score.loc[dt].dropna()
        e_row = elig.loc[dt].reindex(s_row.index).fillna(False)
        v_row = vol.loc[dt].reindex(s_row.index)

        cands    = s_row[e_row]
        selected = cands.nlargest(n_top).index.tolist()
        if not selected:
            continue

        if len(selected) >= 2:
            hist  = me_returns[me_returns.index < dt].tail(lookback)
            avail = [t for t in selected if t in hist.columns]
            sub_w = _hrp_weight(selected, hist[avail].rename(columns=lambda c: c))
        else:
            sub_w = pd.Series(1.0, index=selected)

        w_df.loc[dt, selected] = sub_w

    return w_df


# ─────────────────────────────────────────────────────────────────────────────
# Core: run one variant and return metrics dict
# ─────────────────────────────────────────────────────────────────────────────
def _run_variant(
    prices:      pd.DataFrame,
    returns:     pd.DataFrame,
    rf_monthly:  pd.Series,
    w3:          float = 0.2,
    w6:          float = 0.3,
    w12:         float = 0.5,
    dma_period:  int   = 200,
    lookback:    int   = LOOKBACK_MOS,
    freq:        str   = "ME",
    n_top:       int   = N_TOP,
) -> dict:
    daily      = _daily_signals(prices, returns, w3=w3, w6=w6, w12=w12, dma_period=dma_period)
    me_returns = prices.resample("ME").last().pct_change()

    if freq == "ME":
        sig    = _resample_signals(daily, "ME")
        fwd    = _fwd_returns(prices, sig["score"].index)
    elif freq == "6W":
        rdates = _rebal_dates_6w(prices.index)
        sig    = {k: v.reindex(rdates, method="ffill") for k, v in daily.items()}
        fwd    = _fwd_returns(prices, rdates)
    else:
        sig    = _resample_signals(daily, freq)
        fwd    = _fwd_returns(prices, sig["score"].index)

    w_df = _build_weight_df(sig, me_returns, n_top=n_top, lookback=lookback)
    rf   = rf_monthly.reindex(w_df.index).ffill().fillna(0)
    ret  = _run_backtest_vt(w_df, fwd, me_returns, rf,
                             cost_one_way=COST_ONE_WAY, vol_target=None).dropna()
    return compute_metrics(ret, rf_monthly)


# ─────────────────────────────────────────────────────────────────────────────
# Test 1 — Momentum windows
# ─────────────────────────────────────────────────────────────────────────────
def test_momentum_windows(prices, returns, rf_monthly) -> pd.DataFrame:
    variants = {
        "A: 12M only":        (0.0, 0.0, 1.0),
        "B: 6M only":         (0.0, 1.0, 0.0),
        "C: 3M only":         (1.0, 0.0, 0.0),
        "D: 6M + 12M":        (0.0, 0.4, 0.6),
        f"E: 3+6+12 ({BASE_LABEL})": (0.2, 0.3, 0.5),
    }
    rows = {}
    for label, (w3, w6, w12) in variants.items():
        print(f"    {label} …", end="\r")
        rows[label] = _run_variant(prices, returns, rf_monthly, w3=w3, w6=w6, w12=w12)
    return pd.DataFrame(rows).T[REPORT_COLS]


# ─────────────────────────────────────────────────────────────────────────────
# Test 2 — Trend filter DMA
# ─────────────────────────────────────────────────────────────────────────────
def test_trend_filter(prices, returns, rf_monthly) -> pd.DataFrame:
    variants = {
        "150DMA":           150,
        "180DMA":           180,
        f"200DMA ({BASE_LABEL})": 200,
        "250DMA":           250,
    }
    rows = {}
    for label, dma in variants.items():
        print(f"    {label} …", end="\r")
        rows[label] = _run_variant(prices, returns, rf_monthly, dma_period=dma)
    return pd.DataFrame(rows).T[REPORT_COLS]


# ─────────────────────────────────────────────────────────────────────────────
# Test 3 — Covariance window
# ─────────────────────────────────────────────────────────────────────────────
def test_cov_window(prices, returns, rf_monthly) -> pd.DataFrame:
    variants = {
        "6M lookback":          6,
        "12M lookback":        12,
        "24M lookback":        24,
        f"36M lookback ({BASE_LABEL})": 36,
    }
    rows = {}
    for label, months in variants.items():
        print(f"    {label} …", end="\r")
        rows[label] = _run_variant(prices, returns, rf_monthly, lookback=months)
    return pd.DataFrame(rows).T[REPORT_COLS]


# ─────────────────────────────────────────────────────────────────────────────
# Test 4 — Rebalance frequency
# ─────────────────────────────────────────────────────────────────────────────
def test_rebalance_freq(prices, returns, rf_monthly) -> pd.DataFrame:
    variants = {
        f"Monthly ({BASE_LABEL})": "ME",
        "~6-Weekly":              "6W",
        "Quarterly":              "QE",
    }
    rows = {}
    for label, freq in variants.items():
        print(f"    {label} …", end="\r")
        rows[label] = _run_variant(prices, returns, rf_monthly, freq=freq)
    return pd.DataFrame(rows).T[REPORT_COLS]


# ─────────────────────────────────────────────────────────────────────────────
# Test 5 — Universe size
# ─────────────────────────────────────────────────────────────────────────────
def _download_extra_tickers(tickers: list, start: str, proc_dir: str) -> pd.DataFrame:
    """Download and cache extra tickers for universe extension tests."""
    cache = os.path.join(proc_dir, "extra_universe_prices.parquet")
    if os.path.exists(cache):
        cached = pd.read_parquet(cache)
        if all(t in cached.columns for t in tickers):
            return cached[tickers]

    print(f"    Downloading {len(tickers)} additional tickers …")
    raw = yf.download(tickers, start=start, auto_adjust=True,
                      progress=False, group_by="ticker", threads=True)
    if isinstance(raw.columns, pd.MultiIndex):
        raw = raw.swaplevel(axis=1).sort_index(axis=1)
    px = raw["Close"].ffill(limit=5) if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]].ffill(limit=5)

    if os.path.exists(cache):
        existing = pd.read_parquet(cache)
        px = existing.combine_first(px)
    px.to_parquet(cache)
    return px[tickers]


def test_universe_size(prices: pd.DataFrame, rf_monthly: pd.Series, proc_dir: str) -> pd.DataFrame:
    rows = {}
    universe_defs = {
        "10 ETFs":           UNIVERSE_10,
        f"15 ETFs ({BASE_LABEL})": UNIVERSE_15,
        "20 ETFs":           UNIVERSE_15 + UNIVERSE_20_ADD,
        "30 ETFs":           UNIVERSE_15 + UNIVERSE_30_ADD,
    }

    for label, tickers in universe_defs.items():
        print(f"    {label} …", end="\r")
        new_tickers = [t for t in tickers if t not in prices.columns]

        if new_tickers:
            extra = _download_extra_tickers(new_tickers, START_DATE, proc_dir)
            # extra is already a clean price DataFrame — just forward-fill short gaps
            extra = extra.where(extra > 0).ffill(limit=5)
            px = prices.reindex(columns=tickers, fill_value=np.nan).combine_first(extra)
        else:
            px = prices[tickers]

        # Only keep tickers present in this universe
        px = px[[t for t in tickers if t in px.columns]].dropna(how="all")
        rt = compute_returns(px)
        rows[label] = _run_variant(px, rt, rf_monthly, n_top=min(N_TOP, len(px.columns) // 3))

    return pd.DataFrame(rows).T[REPORT_COLS]


# ─────────────────────────────────────────────────────────────────────────────
# Build and print the full robustness matrix
# ─────────────────────────────────────────────────────────────────────────────
def _section(name: str, df: pd.DataFrame) -> pd.DataFrame:
    """Tag rows with category name for multi-level display."""
    df = df.copy()
    df.index = pd.MultiIndex.from_tuples([(name, v) for v in df.index],
                                          names=["Category", "Variant"])
    return df


def print_robustness_matrix(matrix: pd.DataFrame) -> None:
    print(f"\n{'═'*85}")
    print(f"  ROBUSTNESS MATRIX")
    print(f"{'═'*85}")
    fmt = {
        "Sharpe":      "{:>7.3f}",
        "Sortino":     "{:>7.3f}",
        "Calmar":      "{:>7.3f}",
        "Max DD %":    "{:>8.2f}",
        "Ann Return %":"{:>9.2f}",
        "Ann Vol %":   "{:>8.2f}",
    }
    header = f"  {'Category':<22} {'Variant':<28} " + "  ".join(
        f"{c:>{int(s[3:-1])}}" for c, s in [
            ("Sharpe", "{:>7}"), ("Sortino", "{:>7}"), ("Calmar", "{:>7}"),
            ("Max DD%", "{:>8}"), ("Ann Ret%", "{:>9}"), ("Ann Vol%", "{:>8}"),
        ]
    )
    print(header)

    last_cat = None
    for (cat, var), row in matrix.iterrows():
        is_base = BASE_LABEL in var
        prefix  = "★ " if is_base else "  "
        cat_str = cat if cat != last_cat else ""
        last_cat = cat

        vals = "  ".join([
            f"{row['Sharpe']:>7.3f}",
            f"{row['Sortino']:>7.3f}",
            f"{row['Calmar']:>7.3f}",
            f"{row['Max DD %']:>8.2f}",
            f"{row['Ann Return %']:>9.2f}",
            f"{row['Ann Vol %']:>8.2f}",
        ])
        print(f"{prefix}{cat_str:<22} {var:<28} {vals}")

        if cat != last_cat or (cat == last_cat and BASE_LABEL in var):
            pass
        # Separator between categories
        if cat != matrix.index[-1][0] and last_cat == cat and var == matrix.xs(cat, level=0).index[-1]:
            print(f"  {'─'*81}")

    print(f"{'═'*85}")
    print(f"  ★ = base case")


# ─────────────────────────────────────────────────────────────────────────────
# Top-level pipeline
# ─────────────────────────────────────────────────────────────────────────────
def run_robustness_analysis(
    prices:   pd.DataFrame,
    returns:  pd.DataFrame,
    proc_dir: str = "data/processed",
) -> pd.DataFrame:
    from src.data import load_risk_free_rate
    rf_daily   = load_risk_free_rate(proc_dir)
    rf_monthly = (1 + rf_daily).resample("ME").prod() - 1

    print("\n" + "=" * 60)
    print("  ROBUSTNESS ANALYSIS — 5 parameter dimensions")
    print("=" * 60)

    tests = {}

    print("\n[1/5] Momentum windows …")
    tests["Momentum Windows"] = test_momentum_windows(prices, returns, rf_monthly)

    print("\n[2/5] Trend filter DMA …")
    tests["Trend Filter DMA"] = test_trend_filter(prices, returns, rf_monthly)

    print("\n[3/5] Covariance window …")
    tests["Covariance Window"] = test_cov_window(prices, returns, rf_monthly)

    print("\n[4/5] Rebalance frequency …")
    tests["Rebalance Frequency"] = test_rebalance_freq(prices, returns, rf_monthly)

    print("\n[5/5] Universe size …")
    tests["Universe Size"] = test_universe_size(prices, rf_monthly, proc_dir)

    matrix = pd.concat([_section(name, df) for name, df in tests.items()])
    print_robustness_matrix(matrix)

    # Save
    out = os.path.join(proc_dir, "robustness_matrix.parquet")
    matrix.to_parquet(out)
    print(f"\nSaved → {out}")

    return matrix
