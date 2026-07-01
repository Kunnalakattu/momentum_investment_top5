"""
Data download, cleaning, and preprocessing for the ETF momentum universe.
"""

import os
import warnings
import yaml
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(config_path: str = "config/universe.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_tickers(cfg: dict) -> list[str]:
    tickers = []
    for group in cfg["universe"].values():
        tickers.extend(group)
    return sorted(set(tickers))


def _dl_tickers(tickers: list[str], cfg: dict) -> tuple[list[str], dict[str, str]]:
    """
    Apply exchange suffix (e.g. '.L' for LSE) to ticker symbols for yfinance.
    Returns (download_tickers, reverse_map) so results can be renamed back.
    """
    suffix = cfg.get("data", {}).get("exchange_suffix", "")
    if not suffix:
        return tickers, {}
    dl = [t + suffix for t in tickers]
    reverse = {t + suffix: t for t in tickers}   # "VUSA.L" -> "VUSA"
    return dl, reverse


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_prices(tickers: list[str], start: str, end: str | None = None) -> pd.DataFrame:
    """
    Download daily OHLCV from yfinance. Returns MultiIndex DataFrame
    (columns: Open/High/Low/Close/Volume × tickers).
    """
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,   # gives adjusted OHLCV directly in Close col
        progress=True,
        group_by="ticker",
        threads=True,
    )
    # yfinance with multiple tickers returns (Date, (Ticker, Field)) columns
    # Reorder to (Field, Ticker) MultiIndex
    if isinstance(raw.columns, pd.MultiIndex):
        raw = raw.swaplevel(axis=1).sort_index(axis=1)
    return raw


def download_risk_free_rate(ticker: str = "^IRX", start: str = "2005-01-03") -> pd.Series:
    """
    Download 13-week T-bill annualized rate (%) and convert to daily decimal rate.
    Returns a Series indexed by date.
    """
    rf = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    if rf.empty:
        raise RuntimeError(f"Failed to download risk-free rate from {ticker}")
    # ^IRX is in annualized % — convert to daily decimal
    rf_daily = rf["Close"] / 100 / 252
    rf_daily.name = "rf_daily"
    return rf_daily.squeeze()


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def clean_prices(raw: pd.DataFrame, price_field: str = "Close") -> pd.DataFrame:
    """
    Extract adjusted close prices, remove bad data, and forward-fill short gaps.
    Returns a clean (Date × Ticker) DataFrame.
    """
    if isinstance(raw.columns, pd.MultiIndex):
        adj = raw[price_field].copy()
    else:
        adj = raw[[price_field]].copy()

    # Drop rows that are 100% NaN (non-trading days already excluded by yfinance)
    adj = adj.dropna(how="all")

    # Zero or negative prices are bad data — replace with NaN
    adj = adj.where(adj > 0)

    # Flag tickers whose entire history is missing
    all_nan = adj.columns[adj.isna().all()]
    if len(all_nan):
        print(f"[warn] dropping tickers with no data: {list(all_nan)}")
        adj = adj.drop(columns=all_nan)

    # Forward-fill up to 5 consecutive missing days (e.g. holidays / halts)
    adj = adj.ffill(limit=5)

    # Drop any remaining columns with > 20% missing after ffill
    pct_missing = adj.isna().mean()
    bad = pct_missing[pct_missing > 0.20].index.tolist()
    if bad:
        print(f"[warn] dropping tickers with >20% missing data: {bad}")
        adj = adj.drop(columns=bad)

    return adj.sort_index()


def _corwin_schultz_spread(high: pd.Series, low: pd.Series) -> pd.Series:
    """
    Corwin & Schultz (2012) bid-ask spread estimator from daily H/L prices.
    Returns daily spread estimate as a fraction of price.
    """
    log_hl = np.log(high / low)
    beta = log_hl ** 2 + log_hl.shift(1) ** 2          # sum over 2 days
    gamma = np.log(pd.concat([high, high.shift(1)], axis=1).max(axis=1) /
                   pd.concat([low,  low.shift(1)],  axis=1).min(axis=1)) ** 2
    k = 3 - 2 * np.sqrt(2)
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
    spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    return spread.clip(lower=0)   # negative values are noise → 0


def estimate_bid_ask_spreads(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily Corwin-Schultz spread estimates for each ticker.
    Returns a (Date × Ticker) DataFrame of spread fractions.
    """
    if not isinstance(raw.columns, pd.MultiIndex):
        return pd.DataFrame()

    # After download_prices swaplevel, columns are (Field, Ticker)
    tickers = raw.columns.get_level_values(1).unique()
    try:
        highs = raw["High"]
        lows  = raw["Low"]
    except KeyError:
        return pd.DataFrame()

    spreads = {}
    for tkr in tickers:
        try:
            spreads[tkr] = _corwin_schultz_spread(highs[tkr], lows[tkr])
        except KeyError:
            pass

    return pd.DataFrame(spreads).sort_index()


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_raw(raw: pd.DataFrame, directory: str) -> None:
    Path(directory).mkdir(parents=True, exist_ok=True)
    path = os.path.join(directory, "ohlcv_raw.parquet")
    raw.to_parquet(path)
    print(f"Saved raw OHLCV → {path}")


def save_processed(
    prices: pd.DataFrame,
    spreads: pd.DataFrame,
    rf: pd.Series,
    directory: str,
) -> None:
    Path(directory).mkdir(parents=True, exist_ok=True)
    prices.to_parquet(os.path.join(directory, "prices.parquet"))
    spreads.to_parquet(os.path.join(directory, "bid_ask_spreads.parquet"))
    rf.to_frame().to_parquet(os.path.join(directory, "risk_free_rate.parquet"))
    print(f"Saved processed data → {directory}/")


def load_prices(directory: str = "data/processed") -> pd.DataFrame:
    return pd.read_parquet(os.path.join(directory, "prices.parquet"))


def load_spreads(directory: str = "data/processed") -> pd.DataFrame:
    return pd.read_parquet(os.path.join(directory, "bid_ask_spreads.parquet"))


def load_risk_free_rate(directory: str = "data/processed") -> pd.Series:
    df = pd.read_parquet(os.path.join(directory, "risk_free_rate.parquet"))
    s = df.iloc[:, 0]
    s.name = "rf_daily"
    return s


# ---------------------------------------------------------------------------
# Returns
# ---------------------------------------------------------------------------

def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple daily log returns."""
    return np.log(prices / prices.shift(1)).dropna(how="all")


# ---------------------------------------------------------------------------
# Incremental refresh (monthly use)
# ---------------------------------------------------------------------------

def refresh_data(
    proc_dir:    str = "data/processed",
    config_path: str = "config/universe.yaml",
) -> pd.DataFrame:
    """
    Download only the new trading days since the last saved date.
    Fast (~5-10 seconds) — suitable for running at the top of the
    rebalancing notebook every month.

    Returns the up-to-date prices DataFrame.
    Falls back to a full re-download if no saved data exists.
    """
    prices_path = os.path.join(proc_dir, "prices.parquet")
    rf_path     = os.path.join(proc_dir, "risk_free_rate.parquet")

    if not os.path.exists(prices_path):
        print("No existing data found — running full download.")
        result = run_data_pipeline(config_path)
        return result["prices"]

    existing = pd.read_parquet(prices_path)
    last_date = existing.index[-1]
    today     = pd.Timestamp.today().normalize()

    # Already current (allow 1 day lag for weekend / market close)
    if last_date >= today - pd.Timedelta(days=2):
        print(f"Data already up to date ({last_date.date()}).")
        return existing

    cfg     = load_config(config_path)
    tickers = get_tickers(cfg)
    dl_tickers, reverse_map = _dl_tickers(tickers, cfg)
    start_new = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    print(f"Fetching new price data from {start_new} to today...")

    new_raw = download_prices(dl_tickers, start=start_new)
    if new_raw.empty:
        print("No new data returned — market may be closed.")
        return existing

    # Strip exchange suffix
    if reverse_map and isinstance(new_raw.columns, pd.MultiIndex):
        new_raw.columns = pd.MultiIndex.from_tuples(
            [(reverse_map.get(t, t), f) for t, f in new_raw.columns]
        )
    elif reverse_map:
        new_raw.columns = [reverse_map.get(c, c) for c in new_raw.columns]

    new_prices = clean_prices(new_raw, price_field=cfg["data"]["price_col"])

    # Merge: drop any overlapping rows, keep new data for duplicates
    combined = pd.concat([existing, new_prices])
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    combined.to_parquet(prices_path)
    print(f"Prices updated: {last_date.date()} → {combined.index[-1].date()} "
          f"({len(new_prices)} new trading days)")

    # Refresh risk-free rate tail
    if os.path.exists(rf_path):
        rf_cfg    = cfg["data"]["risk_free_ticker"]
        existing_rf = pd.read_parquet(rf_path).iloc[:, 0]
        rf_last   = existing_rf.index[-1]
        if rf_last < today - pd.Timedelta(days=2):
            rf_start = (rf_last + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            new_rf   = download_risk_free_rate(ticker=rf_cfg, start=rf_start)
            combined_rf = pd.concat([existing_rf, new_rf])
            combined_rf = combined_rf[~combined_rf.index.duplicated(keep="last")].sort_index()
            # Align to trading days
            combined_rf = combined_rf.reindex(combined.index).ffill().bfill()
            combined_rf.to_frame().to_parquet(rf_path)
            print(f"Risk-free rate updated through {combined_rf.index[-1].date()}")

    return combined


# ---------------------------------------------------------------------------
# Top-level pipeline
# ---------------------------------------------------------------------------

def run_data_pipeline(config_path: str = "config/universe.yaml") -> dict:
    """
    Full pipeline: download → clean → estimate spreads → save.
    Returns dict with keys: prices, spreads, rf, returns.
    """
    cfg = load_config(config_path)
    tickers = get_tickers(cfg)
    dl_tickers, reverse_map = _dl_tickers(tickers, cfg)
    start = cfg["data"]["start_date"]
    raw_dir = cfg["data"]["raw_dir"]
    proc_dir = cfg["data"]["processed_dir"]
    rf_ticker = cfg["data"]["risk_free_ticker"]

    print(f"Downloading {len(dl_tickers)} tickers from {start} …")
    raw = download_prices(dl_tickers, start=start)

    # Strip exchange suffix from column names so downstream code uses clean names
    if reverse_map and isinstance(raw.columns, pd.MultiIndex):
        raw.columns = pd.MultiIndex.from_tuples(
            [(reverse_map.get(t, t), f) for t, f in raw.columns]
        )
    elif reverse_map:
        raw.columns = [reverse_map.get(c, c) for c in raw.columns]

    print("Estimating bid-ask spreads …")
    spreads = estimate_bid_ask_spreads(raw)

    print("Cleaning prices …")
    prices = clean_prices(raw, price_field=cfg["data"]["price_col"])

    print("Downloading risk-free rate …")
    rf = download_risk_free_rate(ticker=rf_ticker, start=start)

    # Align rf to trading days in price data
    rf = rf.reindex(prices.index).ffill().bfill()

    save_raw(raw, raw_dir)
    save_processed(prices, spreads, rf, proc_dir)

    returns = compute_returns(prices)
    print(f"\nDone. Price matrix: {prices.shape}, Returns: {returns.shape}")
    print(f"Date range: {prices.index[0].date()} → {prices.index[-1].date()}")
    print("\nMissing data summary (% of trading days):")
    print((prices.isna().mean() * 100).round(2).to_string())

    return {"prices": prices, "spreads": spreads, "rf": rf, "returns": returns}
