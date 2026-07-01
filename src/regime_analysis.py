"""
Regime Analysis — Momentum → Top-5 → 200DMA → HRP

Splits history into macro regimes and answers:
  "Where does momentum work and where does it fail?"

Pattern:
  WORKS  — trending markets (bull or bear), high dispersion, clear leaders
  FAILS  — sharp reversals (momentum crashes), narrow breadth, mean-reversion episodes
"""

import numpy as np
import pandas as pd
from collections import OrderedDict

PERIODS = 12

# ─────────────────────────────────────────────────────────────────────────────
# Regime definitions  (inclusive, month-end index)
# ─────────────────────────────────────────────────────────────────────────────
REGIMES = OrderedDict([
    ("Pre-GFC Bull",    ("2005-01", "2007-08")),   # 32 mo  — normal trending bull
    ("GFC Bear",        ("2007-09", "2009-02")),   # 18 mo  — trending crash → WORKS
    ("GFC Reversal",    ("2009-03", "2009-12")),   # 10 mo  — sharp mean-reversion → FAILS
    ("QE Bull",         ("2010-01", "2019-12")),   # 120 mo — slow grind, low dispersion
    ("COVID Crash",     ("2020-01", "2020-03")),   #  3 mo  — sharp crash → WORKS
    ("COVID Recovery",  ("2020-04", "2021-12")),   # 21 mo  — risk-on reversal → FAILS
    ("Inflation Bear",  ("2022-01", "2022-12")),   # 12 mo  — trending crash → WORKS
    ("Recovery",        ("2023-01", "2026-06")),   # 42 mo  — current cycle
])

# Character labels for annotation
REGIME_TYPE = {
    "Pre-GFC Bull":   "Trending",
    "GFC Bear":       "Trending↓  (WORKS)",
    "GFC Reversal":   "Reversal   (FAILS)",
    "QE Bull":        "Low-vol grind",
    "COVID Crash":    "Trending↓  (WORKS)",
    "COVID Recovery": "Reversal   (FAILS)",
    "Inflation Bear": "Trending↓  (WORKS)",
    "Recovery":       "Mixed",
}

REGIME_COLOR = {
    "Pre-GFC Bull":   "#4CAF50",
    "GFC Bear":       "#2196F3",
    "GFC Reversal":   "#F44336",
    "QE Bull":        "#8BC34A",
    "COVID Crash":    "#2196F3",
    "COVID Recovery": "#F44336",
    "Inflation Bear": "#2196F3",
    "Recovery":       "#9C27B0",
}


# ─────────────────────────────────────────────────────────────────────────────
# Metric helper
# ─────────────────────────────────────────────────────────────────────────────
def _metrics(ret: pd.Series, rf: pd.Series) -> dict:
    ret = ret.dropna()
    if len(ret) < 2:
        return {k: np.nan for k in ["CAGR %", "Sharpe", "MaxDD %", "Calmar", "Hit %", "Ann Vol %"]}

    rf_aligned = rf.reindex(ret.index).fillna(0)
    rf_mean    = float(rf_aligned.mean())
    n_yr       = len(ret) / PERIODS

    cagr     = float((1 + ret).prod() ** (1 / n_yr) - 1)
    ann_vol  = float(ret.std() * np.sqrt(PERIODS))
    ann_rf   = rf_mean * PERIODS
    sharpe   = (cagr - ann_rf) / ann_vol if ann_vol > 0 else np.nan

    equity   = (1 + ret).cumprod()
    max_dd   = float((equity / equity.cummax() - 1).min())
    calmar   = cagr / abs(max_dd) if max_dd < 0 else np.nan
    hit_rate = float((ret > 0).mean() * 100)

    return {
        "CAGR %":    round(cagr    * 100, 2),
        "Ann Vol %": round(ann_vol * 100, 2),
        "Sharpe":    round(sharpe,  3) if not np.isnan(sharpe) else np.nan,
        "MaxDD %":   round(max_dd  * 100, 2),
        "Calmar":    round(calmar,  3) if not np.isnan(calmar) else np.nan,
        "Hit %":     round(hit_rate, 1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Regime slicing
# ─────────────────────────────────────────────────────────────────────────────
def build_regime_table(
    hrp_ret:    pd.Series,
    spy_ret:    pd.Series,
    rf_monthly: pd.Series,
) -> pd.DataFrame:
    rows = []
    for name, (start, end) in REGIMES.items():
        h  = hrp_ret.loc[start:end].dropna()
        s  = spy_ret.reindex(h.index).dropna()
        rf = rf_monthly.reindex(h.index).fillna(0)

        hm = _metrics(h, rf)
        sm = _metrics(s, rf)

        alpha_cagr   = round(hm["CAGR %"] - sm["CAGR %"], 2) if not (np.isnan(hm["CAGR %"]) or np.isnan(sm["CAGR %"])) else np.nan
        sharpe_diff  = round(hm["Sharpe"] - sm["Sharpe"],  3) if not (np.isnan(hm["Sharpe"]) or np.isnan(sm["Sharpe"])) else np.nan

        rows.append({
            "Regime":        name,
            "Type":          REGIME_TYPE[name],
            "N months":      len(h),
            "HRP CAGR %":    hm["CAGR %"],
            "HRP Sharpe":    hm["Sharpe"],
            "HRP MaxDD %":   hm["MaxDD %"],
            "HRP Calmar":    hm["Calmar"],
            "HRP Hit %":     hm["Hit %"],
            "SPY CAGR %":    sm["CAGR %"],
            "SPY Sharpe":    sm["Sharpe"],
            "Alpha CAGR":    alpha_cagr,
            "Sharpe Diff":   sharpe_diff,
        })

    df = pd.DataFrame(rows).set_index("Regime")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Rolling analysis
# ─────────────────────────────────────────────────────────────────────────────
def rolling_metrics(
    hrp_ret:    pd.Series,
    spy_ret:    pd.Series,
    rf_monthly: pd.Series,
    window:     int = 12,
) -> pd.DataFrame:
    """Rolling window Sharpe and active return (alpha)."""
    rf = rf_monthly.reindex(hrp_ret.index).fillna(0)
    spy = spy_ret.reindex(hrp_ret.index).fillna(0)
    rf_mean_ann = rf.mean() * PERIODS

    roll_hrp_sh  = []
    roll_spy_sh  = []
    roll_alpha   = []
    roll_hrp_cagr = []
    dates        = []

    for i in range(window, len(hrp_ret) + 1):
        h  = hrp_ret.iloc[i - window: i]
        s  = spy.iloc[i - window: i]
        n_yr = window / PERIODS

        cagr_h   = float((1 + h).prod() ** (1 / n_yr) - 1)
        cagr_s   = float((1 + s).prod() ** (1 / n_yr) - 1)
        vol_h    = float(h.std() * np.sqrt(PERIODS))
        vol_s    = float(s.std() * np.sqrt(PERIODS))
        sh_h     = (cagr_h - rf_mean_ann) / vol_h if vol_h > 0 else np.nan
        sh_s     = (cagr_s - rf_mean_ann) / vol_s if vol_s > 0 else np.nan

        roll_hrp_sh.append(sh_h)
        roll_spy_sh.append(sh_s)
        roll_alpha.append((cagr_h - cagr_s) * 100)
        roll_hrp_cagr.append(cagr_h * 100)
        dates.append(hrp_ret.index[i - 1])

    return pd.DataFrame({
        "HRP Sharpe":  roll_hrp_sh,
        "SPY Sharpe":  roll_spy_sh,
        "Active CAGR": roll_alpha,
        "HRP CAGR":    roll_hrp_cagr,
    }, index=pd.DatetimeIndex(dates))


# ─────────────────────────────────────────────────────────────────────────────
# Print table
# ─────────────────────────────────────────────────────────────────────────────
def print_regime_table(df: pd.DataFrame) -> None:
    print(f"\n{'='*95}")
    print(f"  REGIME ANALYSIS — Momentum → Top-5 → 200DMA → HRP")
    print(f"{'='*95}")
    print(f"  {'Regime':<20} {'N':>3}  {'HRP%':>7}  {'SH':>6}  {'MaxDD':>7}  "
          f"{'SPY%':>7}  {'Alpha':>7}  {'ΔSharpe':>8}  Type")
    print(f"  {'─'*20} {'─'*3}  {'─'*7}  {'─'*6}  {'─'*7}  "
          f"{'─'*7}  {'─'*7}  {'─'*8}  {'─'*25}")

    for regime, row in df.iterrows():
        alpha    = row["Alpha CAGR"]
        sdiff    = row["Sharpe Diff"]
        outcome  = "WINS ★" if alpha > 0 else "FAILS ✗"
        print(
            f"  {regime:<20} {int(row['N months']):>3}  "
            f"{row['HRP CAGR %']:>7.2f}  "
            f"{row['HRP Sharpe']:>6.3f}  "
            f"{row['HRP MaxDD %']:>7.2f}  "
            f"{row['SPY CAGR %']:>7.2f}  "
            f"{alpha:>+7.2f}  "
            f"{sdiff:>+8.3f}  "
            f"{row['Type']}"
        )

    print(f"\n  Where momentum WINS vs SPY:")
    wins  = df[df["Alpha CAGR"] > 0]
    fails = df[df["Alpha CAGR"] <= 0]
    for r, row in wins.iterrows():
        print(f"    ★ {r:<20}  alpha={row['Alpha CAGR']:+.1f}%  ({row['Type']})")
    print(f"\n  Where momentum FAILS vs SPY:")
    for r, row in fails.iterrows():
        print(f"    ✗ {r:<20}  alpha={row['Alpha CAGR']:+.1f}%  ({row['Type']})")
    print(f"{'='*95}")


# ─────────────────────────────────────────────────────────────────────────────
# Top-level runner
# ─────────────────────────────────────────────────────────────────────────────
def run_regime_analysis(
    prices:   pd.DataFrame,
    returns:  pd.DataFrame,
    proc_dir: str = "data/processed",
) -> dict:
    from src.backtest import load_backtest_returns
    from src.data import load_risk_free_rate

    rf_daily   = load_risk_free_rate(proc_dir)
    rf_monthly = (1 + rf_daily).resample("ME").prod() - 1

    bt_rets = load_backtest_returns(proc_dir)
    hrp_ret = bt_rets["D: HRP"].dropna()
    bench_col = "SPY" if "SPY" in prices.columns else prices.columns[0]
    spy_ret = prices.resample("ME").last()[bench_col].pct_change().dropna()

    regime_df  = build_regime_table(hrp_ret, spy_ret, rf_monthly)
    roll_df    = rolling_metrics(hrp_ret, spy_ret, rf_monthly, window=12)

    print_regime_table(regime_df)

    return {
        "regime_table":  regime_df,
        "rolling":       roll_df,
        "hrp_ret":       hrp_ret,
        "spy_ret":       spy_ret,
        "rf_monthly":    rf_monthly,
    }
