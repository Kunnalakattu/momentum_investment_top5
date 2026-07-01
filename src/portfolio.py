"""
Portfolio selection: rank eligible ETFs by momentum score, select top N, apply HRP weights.

Every month-end:
  1. Filter: keep only assets with price > 200DMA
  2. Rank: sort by composite momentum score (descending)
  3. Select: top N (compare N=5 vs N=7)
  4. Weight: HRP using 36-month rolling covariance
"""

import numpy as np
import pandas as pd

from src.hypothesis_tests import (
    _hrp_weight,
    _inv_vol_weight,
    _risk_parity_weight,
    _run_backtest,
    compute_me_forward_returns,
    _metrics,
    LOOKBACK_MOS,
)


# ─────────────────────────────────────────────────────────────────────────────
# Core selection logic
# ─────────────────────────────────────────────────────────────────────────────

def select_assets(
    score_row: pd.Series,
    elig_row:  pd.Series,
    n_top:     int,
) -> tuple[list, list, list]:
    """
    Returns (selected, eligible_ranked, skipped_ineligible).
    - selected            : top-N eligible tickers, ranked by score
    - eligible_ranked     : all eligible tickers ranked by score
    - skipped_ineligible  : tickers skipped only because of 200DMA filter
    """
    all_scored   = score_row.dropna().sort_values(ascending=False)
    eligible     = elig_row.reindex(all_scored.index).fillna(False)

    # Tickers that rank in top-N by score but are ineligible
    top_n_raw           = all_scored.head(n_top).index.tolist()
    skipped_ineligible  = [t for t in top_n_raw if not eligible[t]]

    eligible_ranked = all_scored[eligible].index.tolist()
    selected        = eligible_ranked[:n_top]

    return selected, eligible_ranked, skipped_ineligible


def build_weights(
    score_row:   pd.Series,
    elig_row:    pd.Series,
    vol_row:     pd.Series,
    rolling_ret: pd.DataFrame,
    n_top:       int,
    method:      str = "hrp",
) -> pd.Series:
    """Full pipeline for one month-end: select → weight."""
    all_tickers = score_row.index.tolist()
    w = pd.Series(0.0, index=all_tickers)

    selected, _, _ = select_assets(score_row, elig_row, n_top)
    if not selected:
        return w

    if method == "hrp" and len(selected) >= 2:
        sub_w = _hrp_weight(selected, rolling_ret)
    elif method == "inv_vol":
        sub_w = _inv_vol_weight(selected, vol_row)
    elif method == "risk_parity" and len(selected) >= 2:
        avail = [t for t in selected if t in rolling_ret.columns]
        cov   = rolling_ret[avail].dropna().cov()
        sub_w = _risk_parity_weight(avail, cov).reindex(selected).fillna(0.0)
    else:
        sub_w = pd.Series(1.0 / len(selected), index=selected)

    w[selected] = sub_w
    return w


# ─────────────────────────────────────────────────────────────────────────────
# Backtest: build full weight matrix over history
# ─────────────────────────────────────────────────────────────────────────────

def build_weight_matrix(
    signals:    dict,
    me_returns: pd.DataFrame,
    n_top:      int,
    method:     str = "hrp",
) -> pd.DataFrame:
    score = signals["score"]
    elig  = signals["eligible"]
    vol   = signals["vol_60d"]

    w_df = pd.DataFrame(0.0, index=score.index, columns=score.columns)
    for dt in score.index:
        hist = me_returns[me_returns.index < dt].tail(LOOKBACK_MOS)
        w_df.loc[dt] = build_weights(
            score.loc[dt], elig.loc[dt], vol.loc[dt],
            rolling_ret=hist, n_top=n_top, method=method,
        )
    return w_df


# ─────────────────────────────────────────────────────────────────────────────
# Comparison: top-5 vs top-7
# ─────────────────────────────────────────────────────────────────────────────

def run_n_comparison(
    signals:  dict,
    prices:   pd.DataFrame,
    method:   str = "hrp",
    ns:       list = [5, 7],
) -> dict:
    """Compare portfolio performance for different N values."""
    me_returns  = prices.resample("ME").last().pct_change()
    fwd_returns = compute_me_forward_returns(prices)

    results = {}
    for n in ns:
        w_df = build_weight_matrix(signals, me_returns, n_top=n, method=method)
        ret  = _run_backtest(w_df, fwd_returns).dropna()
        results[f"Top-{n}"] = {
            "returns":   ret,
            "weights":   w_df,
            "metrics":   _metrics(ret),
            "n_top":     n,
        }

    print(f"\n{'='*55}")
    print(f"  Portfolio selection comparison ({method.upper()} weights)")
    print(f"{'='*55}")
    m_df = pd.DataFrame({k: v["metrics"] for k, v in results.items()})
    print(m_df.to_string())

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Live portfolio — current month-end snapshot
# ─────────────────────────────────────────────────────────────────────────────

def current_portfolio(
    signals:  dict,
    prices:   pd.DataFrame,
    n_top:    int = 5,
    method:   str = "hrp",
) -> pd.DataFrame:
    """
    Print and return the current portfolio (latest month-end).
    Shows full ranking table: score, momentum breakdown, eligibility, selection status.
    """
    score = signals["score"]
    elig  = signals["eligible"]
    vol   = signals["vol_60d"]

    last  = score.index[-1]
    me_returns = prices.resample("ME").last().pct_change()
    hist  = me_returns[me_returns.index < last].tail(LOOKBACK_MOS)

    score_row = score.loc[last]
    elig_row  = elig.loc[last]
    vol_row   = vol.loc[last]

    selected, eligible_ranked, skipped = select_assets(score_row, elig_row, n_top)
    weights = build_weights(score_row, elig_row, vol_row, hist, n_top=n_top, method=method)

    # Full ranking table
    all_ranked = score_row.dropna().sort_values(ascending=False)
    rows = []
    for rank, tkr in enumerate(all_ranked.index, 1):
        is_elig   = bool(elig_row.get(tkr, False))
        is_sel    = tkr in selected
        rows.append({
            "Rank":      rank,
            "Ticker":    tkr,
            "Score":     round(score_row[tkr], 4),
            "3M %":      round(signals["mom_3m"].loc[last, tkr]  * 100, 1),
            "6M %":      round(signals["mom_6m"].loc[last, tkr]  * 100, 1),
            "12M %":     round(signals["mom_12m"].loc[last, tkr] * 100, 1),
            "Vol % pa":  round(vol_row.get(tkr, float("nan"))    * 100, 1),
            "Eligible":  "YES" if is_elig else "NO",
            "Weight %":  round(weights[tkr] * 100, 2) if is_sel else 0.0,
            "Status":    ("SELECTED" if is_sel
                          else ("INELIGIBLE" if not is_elig
                                else "RANKED OUT")),
        })

    table = pd.DataFrame(rows).set_index("Rank")

    print(f"\n{'='*65}")
    print(f"  LIVE PORTFOLIO  —  {last.date()}  —  Top-{n_top} {method.upper()}")
    print(f"{'='*65}")
    print(f"  Universe: {len(all_ranked)} ETFs  |  "
          f"Eligible: {int(elig_row.sum())} (price > 200DMA)  |  "
          f"Selected: {len(selected)}")

    if skipped:
        print(f"  Note: {skipped} ranked in top-{n_top} by score but failed 200DMA filter")

    print(f"\n{'─'*65}")
    print(table.to_string())

    print(f"\n  Portfolio weights:")
    for tkr in selected:
        print(f"    {tkr:<6}  {weights[tkr]*100:>6.2f}%")
    print(f"    {'TOTAL':>6}  {weights.sum()*100:>6.2f}%")

    return table


# ─────────────────────────────────────────────────────────────────────────────
# Top-level runner
# ─────────────────────────────────────────────────────────────────────────────

def run_portfolio_pipeline(
    signals: dict,
    prices:  pd.DataFrame,
    returns: pd.DataFrame,
    ns:      list = [5, 7],
    method:  str  = "hrp",
) -> dict:
    results = run_n_comparison(signals, prices, method=method, ns=ns)

    print()
    current_portfolio(signals, prices, n_top=ns[0], method=method)

    return results
