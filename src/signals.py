"""
Momentum signal computation — all signals calculated at month-end.

Strategy:
  3M  momentum = price / price_63d_ago  - 1
  6M  momentum = price / price_126d_ago - 1
  12M momentum = price / price_252d_ago - 1
  Trend        = price / 200DMA         - 1
  Vol (60d)    = 60-day annualised volatility

Composite score = 0.2 * 3M + 0.3 * 6M + 0.5 * 12M

Eligibility = price > 200DMA  (trend > 0)

All daily signals are calculated first, then sampled at month-end to avoid
look-ahead bias — shift() and rolling() only use past observations.
"""

import numpy as np
import pandas as pd
import os
from pathlib import Path


# ---------------------------------------------------------------------------
# Daily signal computation
# ---------------------------------------------------------------------------

def compute_daily_signals(prices: pd.DataFrame, returns: pd.DataFrame) -> dict:
    """
    Compute all raw signals at daily frequency.
    Returns dict of DataFrames, each (Date x Ticker).
    """
    dma200 = prices.rolling(200, min_periods=180).mean()

    signals = {
        "mom_3m":  prices / prices.shift(63)  - 1,
        "mom_6m":  prices / prices.shift(126) - 1,
        "mom_12m": prices / prices.shift(252) - 1,
        "trend":   prices / dma200            - 1,
        "vol_60d": returns.rolling(60, min_periods=50).std() * np.sqrt(252),
        "dma200":  dma200,
    }
    return signals


def compute_score(daily: dict) -> pd.DataFrame:
    """Composite momentum score (raw, not cross-sectionally ranked)."""
    return (
        0.2 * daily["mom_3m"]
        + 0.3 * daily["mom_6m"]
        + 0.5 * daily["mom_12m"]
    )


def compute_eligibility(daily: dict) -> pd.DataFrame:
    """Boolean mask: True when price is above 200DMA (trend > 0)."""
    return daily["trend"] > 0


# ---------------------------------------------------------------------------
# Month-end snapshot
# ---------------------------------------------------------------------------

def to_month_end(df: pd.DataFrame) -> pd.DataFrame:
    """Sample a daily DataFrame at the last trading day of each month."""
    return df.resample("ME").last()


def compute_month_end_signals(prices: pd.DataFrame, returns: pd.DataFrame) -> dict:
    """
    Full signal pipeline -> month-end snapshots.
    Returns dict with keys:
      mom_3m, mom_6m, mom_12m, trend, vol_60d, score, eligible, dma200
    """
    daily = compute_daily_signals(prices, returns)
    daily["score"]    = compute_score(daily)
    daily["eligible"] = compute_eligibility(daily)

    return {k: to_month_end(v) for k, v in daily.items()}


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_signals(signals: dict, directory: str = "data/processed") -> None:
    Path(directory).mkdir(parents=True, exist_ok=True)
    for name, df in signals.items():
        path = os.path.join(directory, f"signal_{name}.parquet")
        df.to_parquet(path)
    print(f"Saved {len(signals)} signal files -> {directory}/")


def load_signals(directory: str = "data/processed") -> dict:
    signals = {}
    for path in sorted(Path(directory).glob("signal_*.parquet")):
        key = path.stem.replace("signal_", "")
        signals[key] = pd.read_parquet(path)
    return signals


# ---------------------------------------------------------------------------
# Top-level pipeline
# ---------------------------------------------------------------------------

def run_signal_pipeline(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    proc_dir: str = "data/processed",
) -> dict:
    print("Computing momentum signals ...")
    signals = compute_month_end_signals(prices, returns)
    save_signals(signals, proc_dir)

    me = signals["score"]
    print(f"\nSignal matrix: {len(me)} month-ends x {me.shape[1]} tickers")
    print(f"Date range   : {me.index[0].date()} -> {me.index[-1].date()}")

    last = me.index[-1]
    score_last = signals["score"].loc[last].sort_values(ascending=False)
    elig_last  = signals["eligible"].loc[last]
    vol_last   = signals["vol_60d"].loc[last]

    print(f"\n{'='*60}")
    print(f"Latest month-end snapshot: {last.date()}")
    print(f"{'='*60}")
    print(f"{'Ticker':<8} {'Score':>8} {'3M%':>7} {'6M%':>7} {'12M%':>8} {'Vol%':>7} {'Eligible':>9}")
    print("-" * 60)
    for tkr in score_last.index:
        m3  = signals["mom_3m"].loc[last, tkr]
        m6  = signals["mom_6m"].loc[last, tkr]
        m12 = signals["mom_12m"].loc[last, tkr]
        v   = vol_last.get(tkr, float("nan"))
        e   = bool(elig_last.get(tkr, False))
        sc  = score_last[tkr]
        print(
            f"{tkr:<8} {sc:>8.4f} {m3*100:>6.1f}% {m6*100:>6.1f}% {m12*100:>7.1f}%"
            f" {v*100:>6.1f}%  {'YES' if e else 'NO':>8}"
        )

    eligible = elig_last[elig_last].index.tolist()
    print(f"\nEligible (price > 200DMA): {eligible}")
    return signals
