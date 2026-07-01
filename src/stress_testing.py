"""
Stress Testing — Momentum → Top-5 → 200DMA → HRP

Answers: "Would the strategy survive?" for 10 historical crises.

Per-event metrics:
  - Total return (strategy vs SPY)
  - Max drawdown (strategy vs SPY)
  - Protection ratio — how much of SPY's crash was avoided?
  - Portfolio composition — what did the strategy actually hold?
  - Survivability verdict: THRIVED / SURVIVED / PROTECTED / LAGGED / FAILED

Full-period tail risk:
  - Historical VaR at 5% and 1%
  - CVaR / Expected Shortfall at 5% and 1%
  - Tail ratio (CVaR strategy / CVaR SPY)
"""

import numpy as np
import pandas as pd
from collections import OrderedDict

PERIODS = 12

# ─────────────────────────────────────────────────────────────────────────────
# Universe classification
# ─────────────────────────────────────────────────────────────────────────────
ASSET_CLASSES = {
    "Equity":      ["SPY", "QQQ", "IWM", "VGK", "EEM"],
    "Bonds":       ["TLT", "IEF", "SHY", "BIL"],
    "Commodities": ["GLD", "SLV", "DBC", "USO", "UNG"],
    "Real Estate": ["VNQ"],
}


# ─────────────────────────────────────────────────────────────────────────────
# Stress scenario definitions
# ─────────────────────────────────────────────────────────────────────────────
STRESS_EVENTS = OrderedDict([
    ("GFC Crash",
     ("2007-09", "2009-02",
      "Lehman collapse, credit freeze. S&P fell −57% peak-to-trough.")),
    ("GFC Reversal",
     ("2009-03", "2009-12",
      "V-shaped recovery. Momentum crashes: worst assets led the bounce.")),
    ("Flash Crash 2010",
     ("2010-05", "2010-06",
      "Algorithmic flash crash. Single-day -9.2% then full recovery within weeks.")),
    ("Euro Crisis 2011",
     ("2011-07", "2011-10",
      "European sovereign debt crisis. Peripheral bonds blow out, risk-off.")),
    ("Oil Collapse",
     ("2014-07", "2016-01",
      "Oil price collapsed −75%. Commodity bust = strategy's worst ever drawdown.")),
    ("China Scare 2015",
     ("2015-08", "2015-09",
      "China RMB devaluation, circuit breakers, global EM sell-off.")),
    ("COVID Crash",
     ("2020-01", "2020-03",
      "Pandemic shock. Fastest bear market in history: −34% SPY in 33 days.")),
    ("Commodity Boom",
     ("2020-11", "2022-06",
      "Post-COVID commodity supercycle: oil, metals, agriculture all surged.")),
    ("Inflation Bear",
     ("2022-01", "2022-12",
      "Fed hiked 425bp. Rare scenario: stocks AND bonds both fell simultaneously.")),
    ("AI Bull 2023-24",
     ("2023-01", "2024-12",
      "AI-driven equity rally. Narrow leadership (Mag-7). Cross-asset momentum lagged?")),
])

STRESS_COLOR = {
    "GFC Crash":         "#2196F3",
    "GFC Reversal":      "#F44336",
    "Flash Crash 2010":  "#FF9800",
    "Euro Crisis 2011":  "#9C27B0",
    "Oil Collapse":      "#795548",
    "China Scare 2015":  "#FF5722",
    "COVID Crash":       "#2196F3",
    "Commodity Boom":    "#4CAF50",
    "Inflation Bear":    "#E91E63",
    "AI Bull 2023-24":   "#00BCD4",
}


# ─────────────────────────────────────────────────────────────────────────────
# Tail risk helpers
# ─────────────────────────────────────────────────────────────────────────────
def var_cvar(returns: pd.Series, level: float = 0.05) -> tuple[float, float]:
    """Historical VaR and CVaR at given level (annualised from monthly)."""
    r = returns.dropna()
    var  = float(r.quantile(level))
    cvar = float(r[r <= var].mean())
    return var * np.sqrt(PERIODS), cvar * np.sqrt(PERIODS)


def compute_tail_risk(
    hrp_ret:    pd.Series,
    spy_ret:    pd.Series,
    rf_monthly: pd.Series,
) -> dict:
    """Full-period tail risk table."""
    common = hrp_ret.index.intersection(spy_ret.index)
    h = hrp_ret.loc[common]
    s = spy_ret.reindex(common).fillna(0)

    h_var5,  h_cvar5  = var_cvar(h, 0.05)
    h_var1,  h_cvar1  = var_cvar(h, 0.01)
    s_var5,  s_cvar5  = var_cvar(s, 0.05)
    s_var1,  s_cvar1  = var_cvar(s, 0.01)

    worst5_hrp = h.nsmallest(max(1, int(len(h) * 0.05)))
    worst5_spy = s.nsmallest(max(1, int(len(s) * 0.05)))

    return {
        "hrp_var5":    h_var5,
        "hrp_cvar5":   h_cvar5,
        "hrp_var1":    h_var1,
        "hrp_cvar1":   h_cvar1,
        "spy_var5":    s_var5,
        "spy_cvar5":   s_cvar5,
        "spy_var1":    s_var1,
        "spy_cvar1":   s_cvar1,
        "tail_ratio5": h_cvar5 / s_cvar5 if s_cvar5 < 0 else np.nan,
        "tail_ratio1": h_cvar1 / s_cvar1 if s_cvar1 < 0 else np.nan,
        "worst5_hrp":  worst5_hrp,
        "worst5_spy":  worst5_spy,
        "monthly_ret_hrp": h,
        "monthly_ret_spy": s,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Per-scenario metrics
# ─────────────────────────────────────────────────────────────────────────────
def _verdict(strat_ret: float, spy_ret: float) -> str:
    diff = strat_ret - spy_ret
    if strat_ret > 0.05 and diff > 0.10:
        return "THRIVED ★★"
    elif strat_ret > 0 and diff > 0:
        return "SURVIVED ✓"
    elif strat_ret > 0 and diff <= 0:
        return "LAGGED ~"
    elif diff > 0:
        return "PROTECTED ↓↑"
    else:
        return "FAILED ✗"


def _maxdd(returns: pd.Series) -> float:
    eq = (1 + returns).cumprod()
    return float((eq / eq.cummax() - 1).min())


def _class_weights(weights_df: pd.DataFrame, start: str, end: str) -> dict:
    """Average weight per asset class during the stress period."""
    slice_ = weights_df.loc[start:end]
    if len(slice_) == 0:
        return {ac: 0.0 for ac in ASSET_CLASSES}
    out = {}
    for ac, tickers in ASSET_CLASSES.items():
        cols = [t for t in tickers if t in slice_.columns]
        out[ac] = float(slice_[cols].sum(axis=1).mean()) if cols else 0.0
    return out


def compute_stress_metrics(
    hrp_ret:    pd.Series,
    spy_ret:    pd.Series,
    rf_monthly: pd.Series,
    weights_df: pd.DataFrame,
    name:       str,
    start:      str,
    end:        str,
    description: str,
) -> dict:
    h   = hrp_ret.loc[start:end].dropna()
    s   = spy_ret.reindex(h.index).fillna(0)
    rf  = rf_monthly.reindex(h.index).fillna(0)

    if len(h) == 0:
        return {}

    n_yr      = max(len(h) / PERIODS, 1 / PERIODS)
    strat_ret = float((1 + h).prod() - 1)
    spy_ret_  = float((1 + s).prod() - 1)
    strat_dd  = _maxdd(h)
    spy_dd    = _maxdd(s)

    # Protection ratio: how much of SPY's loss did strategy avoid?
    # 100% = broke even while SPY lost; >100% = made money while SPY lost
    if spy_ret_ < 0:
        protection = (strat_ret - spy_ret_) / abs(spy_ret_) * 100
    else:
        protection = np.nan

    # Monthly VaR and CVaR during stress period
    h_var5, h_cvar5 = var_cvar(h, 0.05)
    s_var5, s_cvar5 = var_cvar(s, 0.05)

    # Annualised Sharpe during period
    rf_mean = rf.mean()
    excess_h = h - rf
    sh = float((excess_h.mean() - 0) / h.std() * np.sqrt(PERIODS)) if h.std() > 0 else np.nan

    # Top 3 held assets during period
    weights_slice = weights_df.loc[start:end] if len(weights_df.loc[start:end]) > 0 else pd.DataFrame()
    if len(weights_slice) > 0:
        top_assets = weights_slice.mean().sort_values(ascending=False).head(5)
        top_str = ", ".join(f"{t}({w*100:.0f}%)" for t, w in top_assets.items() if w > 0.01)
    else:
        top_str = "n/a"

    class_wts = _class_weights(weights_df, start, end)

    return {
        "name":            name,
        "description":     description,
        "n_months":        len(h),
        "strat_total_ret": strat_ret,
        "spy_total_ret":   spy_ret_,
        "excess_ret":      strat_ret - spy_ret_,
        "strat_dd":        strat_dd,
        "spy_dd":          spy_dd,
        "dd_saved":        strat_dd - spy_dd,
        "protection_pct":  protection,
        "strat_sharpe":    sh,
        "verdict":         _verdict(strat_ret, spy_ret_),
        "top_assets":      top_str,
        "class_equity":    class_wts.get("Equity", 0),
        "class_bonds":     class_wts.get("Bonds", 0),
        "class_commodities": class_wts.get("Commodities", 0),
        "class_realestate":  class_wts.get("Real Estate", 0),
        "hrp_returns":     h,
        "spy_returns":     s,
    }


def build_stress_table(
    hrp_ret:    pd.Series,
    spy_ret:    pd.Series,
    rf_monthly: pd.Series,
    weights_df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict]]:
    rows = []
    details = []
    for name, (start, end, desc) in STRESS_EVENTS.items():
        m = compute_stress_metrics(hrp_ret, spy_ret, rf_monthly, weights_df, name, start, end, desc)
        if not m:
            continue
        rows.append({
            "Scenario":      m["name"],
            "N Mo":          m["n_months"],
            "HRP %":         round(m["strat_total_ret"] * 100, 1),
            "SPY %":         round(m["spy_total_ret"] * 100, 1),
            "Excess %":      round(m["excess_ret"] * 100, 1),
            "HRP MaxDD %":   round(m["strat_dd"] * 100, 1),
            "SPY MaxDD %":   round(m["spy_dd"] * 100, 1),
            "DD Saved %":    round(m["dd_saved"] * 100, 1),
            "Protection %":  round(m["protection_pct"], 0) if not np.isnan(m["protection_pct"]) else np.nan,
            "Equity Wt %":   round(m["class_equity"] * 100, 0),
            "Bond Wt %":     round(m["class_bonds"] * 100, 0),
            "Cmdty Wt %":    round(m["class_commodities"] * 100, 0),
            "Verdict":       m["verdict"],
        })
        details.append(m)

    df = pd.DataFrame(rows).set_index("Scenario")
    return df, details


# ─────────────────────────────────────────────────────────────────────────────
# Print output
# ─────────────────────────────────────────────────────────────────────────────
def print_stress_results(stress_df: pd.DataFrame, tail_risk: dict) -> None:
    print(f"\n{'='*110}")
    print(f"  STRESS TESTING — Momentum → Top-5 → 200DMA → HRP")
    print(f"{'='*110}")
    print(f"  {'Scenario':<22} {'N':>3}  {'HRP%':>7}  {'SPY%':>7}  {'Excess':>7}  "
          f"{'HRP DD':>7}  {'SPY DD':>7}  {'Eq%':>5}  {'Bd%':>5}  {'Cm%':>5}  Verdict")
    print(f"  {'─'*22} {'─'*3}  {'─'*7}  {'─'*7}  {'─'*7}  "
          f"{'─'*7}  {'─'*7}  {'─'*5}  {'─'*5}  {'─'*5}  {'─'*14}")

    for name, row in stress_df.iterrows():
        prot = f"  [{row['Protection %']:+.0f}% prot]" if not np.isnan(row['Protection %']) else ""
        print(f"  {name:<22} {int(row['N Mo']):>3}  "
              f"{row['HRP %']:>+6.1f}%  "
              f"{row['SPY %']:>+6.1f}%  "
              f"{row['Excess %']:>+6.1f}%  "
              f"{row['HRP MaxDD %']:>+6.1f}%  "
              f"{row['SPY MaxDD %']:>+6.1f}%  "
              f"{row['Equity Wt %']:>4.0f}%  "
              f"{row['Bond Wt %']:>4.0f}%  "
              f"{row['Cmdty Wt %']:>4.0f}%  "
              f"{row['Verdict']}{prot}")

    # Tail risk
    print(f"\n  Tail Risk — Full Period (annualised)")
    print(f"  {'Metric':<30} {'HRP':>10}  {'SPY':>10}  {'Ratio':>8}")
    print(f"  {'─'*30} {'─'*10}  {'─'*10}  {'─'*8}")
    print(f"  {'VaR (5%, monthly ann)':30} {tail_risk['hrp_var5']*100:>+9.1f}%  "
          f"{tail_risk['spy_var5']*100:>+9.1f}%  "
          f"{tail_risk['hrp_var5']/tail_risk['spy_var5']:>8.2f}x")
    print(f"  {'CVaR/ES (5%, monthly ann)':30} {tail_risk['hrp_cvar5']*100:>+9.1f}%  "
          f"{tail_risk['spy_cvar5']*100:>+9.1f}%  "
          f"{tail_risk['tail_ratio5']:>8.2f}x")
    print(f"  {'VaR (1%, monthly ann)':30} {tail_risk['hrp_var1']*100:>+9.1f}%  "
          f"{tail_risk['spy_var1']*100:>+9.1f}%  "
          f"{tail_risk['hrp_var1']/tail_risk['spy_var1']:>8.2f}x")
    print(f"  {'CVaR/ES (1%, monthly ann)':30} {tail_risk['hrp_cvar1']*100:>+9.1f}%  "
          f"{tail_risk['spy_cvar1']*100:>+9.1f}%  "
          f"{tail_risk['tail_ratio1']:>8.2f}x")

    thrived   = (stress_df["Verdict"].str.startswith("THRIVED")).sum()
    survived  = (stress_df["Verdict"].str.startswith("SURVIVED")).sum()
    protected = (stress_df["Verdict"].str.startswith("PROTECTED")).sum()
    lagged    = (stress_df["Verdict"].str.startswith("LAGGED")).sum()
    failed    = (stress_df["Verdict"].str.startswith("FAILED")).sum()
    total     = len(stress_df)

    print(f"\n  Survivability Score:  {thrived} thrived  |  {survived} survived  |  "
          f"{protected} protected  |  {lagged} lagged  |  {failed} failed  (of {total} scenarios)")
    print(f"{'='*110}")


# ─────────────────────────────────────────────────────────────────────────────
# Top-level runner
# ─────────────────────────────────────────────────────────────────────────────
def run_stress_testing(
    prices:   pd.DataFrame,
    returns:  pd.DataFrame,
    proc_dir: str = "data/processed",
) -> dict:
    from src.backtest import load_backtest_returns
    from src.data import load_risk_free_rate
    from src.portfolio import build_weight_matrix
    from src.signals import load_signals

    rf_daily   = load_risk_free_rate(proc_dir)
    rf_monthly = (1 + rf_daily).resample("ME").prod() - 1

    bt_rets   = load_backtest_returns(proc_dir)
    hrp_ret   = bt_rets["D: HRP"].dropna()

    me_prices = prices.resample("ME").last()
    me_returns = me_prices.pct_change().dropna()
    spy_ret   = me_returns["SPY"]

    signals   = load_signals(proc_dir)
    weights_df = build_weight_matrix(signals, me_returns, n_top=5, method="hrp")

    stress_df, details = build_stress_table(hrp_ret, spy_ret, rf_monthly, weights_df)
    tail_risk          = compute_tail_risk(hrp_ret, spy_ret, rf_monthly)

    print_stress_results(stress_df, tail_risk)

    return {
        "stress_table":  stress_df,
        "details":       details,
        "tail_risk":     tail_risk,
        "hrp_ret":       hrp_ret,
        "spy_ret":       spy_ret,
        "rf_monthly":    rf_monthly,
        "weights_df":    weights_df,
    }
