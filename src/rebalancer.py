"""
Rebalancing Engine — the final step before placing real orders.

Reads current holdings, computes HRP target weights for this month,
calculates exact share deltas, estimates costs, and outputs a
ready-to-execute trade sheet in execution order (sells before buys).

Usage:
    from src.rebalancer import run_rebalancer
    result = run_rebalancer(
        signals, prices, returns,
        holdings={"SPY": 10, "TLT": 8, "GLD": 5},
        cash=500.0,
        fractional_shares=False,   # True for IBKR / eToro
        spread_bps=2.0,
        commission_per_trade=0.0,  # £ per trade (IBKR Pro fixed = $0.35-$1)
        commission_pct=0.0,        # % of trade value
        min_trade_pct=0.005,       # skip trades < 0.5% of portfolio
    )
"""

import numpy as np
import pandas as pd

PERIODS = 12

# Estimated half-spread in bps for each ETF (based on typical bid/ask)
TYPICAL_SPREAD_BPS = {
    # US mega-cap tech / growth
    "AAPL": 0.1, "MSFT": 0.1, "NVDA": 0.1, "AMZN": 0.2, "META": 0.2,
    "GOOGL": 0.2, "AVGO": 0.2, "ORCL": 0.3, "CRM": 0.3, "ADBE": 0.3,
    # Semiconductors
    "AMD": 0.2, "TSM": 0.3, "QCOM": 0.3, "MU": 0.3, "TXN": 0.3,
    # Financials
    "JPM": 0.1, "BAC": 0.1, "GS": 0.2, "MS": 0.2, "BLK": 0.3,
    "V": 0.1, "MA": 0.1,
    # Healthcare
    "LLY": 0.2, "UNH": 0.2, "JNJ": 0.2, "ABBV": 0.2, "MRK": 0.2,
    # Consumer
    "COST": 0.2, "WMT": 0.2, "PG": 0.2, "KO": 0.2, "PEP": 0.2,
    # Industrials
    "CAT": 0.2, "GE": 0.2, "RTX": 0.3, "DE": 0.3, "HON": 0.3, "LIN": 0.3,
    # Energy
    "XOM": 0.2, "CVX": 0.2, "COP": 0.3, "SLB": 0.3, "EOG": 0.3,
    # Communication / media
    "NFLX": 0.3, "DIS": 0.3, "UBER": 0.3, "PLTR": 0.3, "SPOT": 0.5, "TMUS": 0.3,
    # Commodity-related equities
    "NEM": 0.3, "GOLD": 0.3, "AEM": 0.5, "FCX": 0.3, "SCCO": 0.5,
    "RIO": 0.5, "BHP": 0.5, "VALE": 0.5,
    "ADM": 0.3, "BG": 0.5, "MOS": 0.5, "NTR": 0.5,
    # Index / diversified
    "SPY": 0.1, "BRK-B": 0.2,
}


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Current prices
# ─────────────────────────────────────────────────────────────────────────────

def fetch_live_prices(tickers: list[str], fallback_prices: pd.DataFrame = None) -> dict[str, float]:
    """
    Download latest prices via yfinance.
    Falls back to last row of prices DataFrame if network unavailable.
    """
    import yfinance as yf

    tickers = [t for t in tickers if t]
    try:
        raw = yf.download(tickers, period="5d", auto_adjust=True, progress=False)["Close"]
        if isinstance(raw, pd.Series):
            raw = raw.to_frame(name=tickers[0])
        prices = raw.ffill().iloc[-1].to_dict()
        missing = [t for t in tickers if pd.isna(prices.get(t, float("nan")))]
        if missing and fallback_prices is not None:
            fb = fallback_prices.ffill().iloc[-1]
            for t in missing:
                if t in fb.index:
                    prices[t] = float(fb[t])
                    print(f"  [WARN] {t}: using fallback price ${fb[t]:.2f} (live fetch failed)")
        return {k: float(v) for k, v in prices.items() if not pd.isna(v)}
    except Exception as e:
        print(f"  [WARN] Live price fetch failed ({e}). Using last known prices.")
        if fallback_prices is not None:
            fb = fallback_prices.ffill().iloc[-1]
            return {t: float(fb[t]) for t in tickers if t in fb.index and not pd.isna(fb[t])}
        raise RuntimeError("Cannot fetch prices and no fallback provided.") from e


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Current portfolio state
# ─────────────────────────────────────────────────────────────────────────────

def compute_current_state(
    holdings: dict[str, float],
    prices: dict[str, float],
    cash: float = 0.0,
) -> dict:
    """
    Compute the current value and weight of every held position.

    holdings: {ticker: shares_held}  (0 shares = not held)
    prices:   {ticker: price_per_share}
    cash:     uninvested cash (same currency as prices)
    """
    positions = {}
    for tkr, shares in holdings.items():
        p = prices.get(tkr)
        if p is None or pd.isna(p):
            print(f"  [WARN] No price for {tkr} — excluded from current state")
            continue
        value = shares * float(p)
        positions[tkr] = {"shares": float(shares), "price": float(p), "value": value}

    invested = sum(v["value"] for v in positions.values())
    total    = invested + float(cash)

    for tkr in positions:
        positions[tkr]["weight"] = positions[tkr]["value"] / total if total > 0 else 0.0

    return {
        "positions": positions,
        "cash":      float(cash),
        "invested":  invested,
        "total":     total,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — HRP target weights for this month
# ─────────────────────────────────────────────────────────────────────────────

def get_target_weights(
    signals:  dict,
    prices_df: pd.DataFrame,
    n_top:    int = 5,
    proc_dir: str = "data/processed",
) -> pd.Series:
    """
    Run the signal → selection → HRP pipeline for the latest month-end
    and return the target weight vector.
    """
    from src.portfolio import build_weight_matrix

    me_returns = prices_df.resample("ME").last().pct_change().dropna()
    weights_df = build_weight_matrix(signals, me_returns, n_top=n_top, method="hrp")

    latest_wts = weights_df.iloc[-1]
    latest_wts = latest_wts[latest_wts > 1e-6]           # drop zero weights
    return latest_wts.sort_values(ascending=False)


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Target shares
# ─────────────────────────────────────────────────────────────────────────────

def compute_target_shares(
    target_weights: pd.Series,
    total_value:    float,
    prices:         dict[str, float],
    fractional_shares: bool = False,
) -> dict[str, dict]:
    """
    Convert weight fractions → target number of shares.

    Fractional mode: simple proportional allocation (exact weights).

    Whole-share mode (two phases):
      Phase 1 — floor each position to whole shares.
      Phase 2 — reinvest residual cash greedily: each iteration buys one
                 share of whichever ETF is most underweight relative to its
                 target, until no affordable ETF remains.
    This ensures the portfolio is as fully invested as possible even when
    individual share prices are large relative to the portfolio.
    """
    valid = {
        tkr: float(prices[tkr])
        for tkr, wt in target_weights.items()
        if prices.get(tkr) and prices[tkr] > 0
    }
    missing = [t for t in target_weights.index if t not in valid]
    for t in missing:
        print(f"  [WARN] No price for {t} — excluded from target")

    if fractional_shares:
        import math
        targets = {}
        for tkr, wt in target_weights.items():
            if tkr not in valid:
                continue
            p = valid[tkr]
            # Truncate (floor) to 2 decimal places — never overspend
            shares = math.floor(float(wt) * total_value / p * 100) / 100
            targets[tkr] = {
                "weight":        float(wt),
                "target_value":  float(wt) * total_value,
                "target_shares": shares,
                "actual_value":  shares * p,
            }
        return targets

    # ── Whole-share allocation ────────────────────────────────────────────────
    # Phase 1: floor
    share_counts: dict[str, int] = {}
    cash_left = total_value
    for tkr, wt in target_weights.items():
        if tkr not in valid:
            continue
        p = valid[tkr]
        n = int(float(wt) * total_value / p)
        share_counts[tkr] = n
        cash_left -= n * p

    # Phase 2: greedy reinvestment — buy 1 share of the most-underweight ETF
    # that we can still afford, repeat until no affordable position remains.
    affordable = [t for t in share_counts if valid[t] <= cash_left]
    while affordable:
        # Actual weight of each ETF so far
        invested = {t: share_counts[t] * valid[t] for t in share_counts}
        total_invested = sum(invested.values()) + cash_left  # constant = total_value

        # Pick the ETF with the largest gap: target_wt - actual_wt
        best = max(
            affordable,
            key=lambda t: float(target_weights[t]) - invested[t] / total_value
        )
        share_counts[best] += 1
        cash_left -= valid[best]
        affordable = [t for t in share_counts if valid[t] <= cash_left]

    # Warn if residual is large (means even the cheapest ETF is unaffordable)
    cheapest_price = min(valid[t] for t in share_counts) if share_counts else 0
    uninvested_pct = cash_left / total_value * 100
    if uninvested_pct > 5 and cash_left > cheapest_price:
        print(f"  [INFO] {uninvested_pct:.1f}% ({cash_left:.2f}) uninvested — "
              f"cheapest ETF is ${cheapest_price:.2f}. "
              f"Enable fractional_shares=True for full deployment.")

    targets = {}
    for tkr in share_counts:
        p   = valid[tkr]
        n   = share_counts[tkr]
        wt  = float(target_weights.get(tkr, 0))
        targets[tkr] = {
            "weight":        wt,
            "target_value":  wt * total_value,
            "target_shares": n,
            "actual_value":  n * p,
        }
    return targets


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Trade sheet
# ─────────────────────────────────────────────────────────────────────────────

def build_trade_sheet(
    current:              dict,
    target:               dict[str, dict],
    prices:               dict[str, float],
    spread_bps:           float = 2.0,
    commission_per_trade: float = 0.0,
    commission_pct:       float = 0.0,
    min_trade_pct:        float = 0.005,
) -> pd.DataFrame:
    """
    Build the full trade sheet.

    spread_bps           : one-way half-spread in bps (per ETF or global default)
    commission_per_trade : flat fee per order in £ (e.g. IBKR fixed = $0.35-$1)
    commission_pct       : % of trade value (e.g. Fidelity UK ETF = 0%)
    min_trade_pct        : skip any trade smaller than this fraction of total portfolio
    """
    total_value = current["total"]
    min_value   = min_trade_pct * total_value

    # All tickers that appear in either side
    all_tickers = sorted(
        set(current["positions"].keys()) | set(target.keys())
    )

    rows = []
    for tkr in all_tickers:
        curr_pos = current["positions"].get(tkr, {"shares": 0.0, "value": 0.0, "weight": 0.0, "price": prices.get(tkr, 0.0)})
        tgt_pos  = target.get(tkr,  {"weight": 0.0, "target_shares": 0, "actual_value": 0.0})

        curr_shares = float(curr_pos["shares"])
        tgt_shares  = float(tgt_pos["target_shares"])
        delta       = tgt_shares - curr_shares
        price       = float(prices.get(tkr, curr_pos.get("price", 0.0)))
        trade_value = abs(delta) * price

        # Determine action
        if abs(delta) == 0:
            action = "HOLD"
        elif trade_value < min_value:
            action      = "HOLD (drift < threshold)"
            delta       = 0.0
            trade_value = 0.0
        elif delta > 0:
            action = "BUY"
        elif tgt_shares == 0 and curr_shares > 0:
            action = "SELL ALL"
        else:
            action = "SELL"

        # Cost estimate
        sp_bps      = TYPICAL_SPREAD_BPS.get(tkr, spread_bps)
        spread_cost = trade_value * sp_bps / 10_000
        is_active   = action in ("BUY", "SELL", "SELL ALL")
        comm_flat   = commission_per_trade if is_active else 0.0
        comm_pct_v  = trade_value * commission_pct / 100.0 if is_active else 0.0
        total_cost  = spread_cost + comm_flat + comm_pct_v

        rows.append({
            "ETF":              tkr,
            "Action":           action,
            "Curr Shares":      curr_shares,
            "Target Shares":    tgt_shares,
            "Delta":            delta,
            "Price":            price,
            "Trade Value":      trade_value,
            "Spread Cost":      spread_cost,
            "Commission":       comm_flat + comm_pct_v,
            "Total Cost":       total_cost,
            "Curr Wt %":        curr_pos["weight"] * 100,
            "Target Wt %":      tgt_pos["weight"] * 100,
        })

    df = pd.DataFrame(rows)
    return df


def _execution_order(trade_df: pd.DataFrame) -> pd.DataFrame:
    """Sells first (free up cash), then buys, then holds."""
    sells = trade_df[trade_df["Action"].isin(["SELL", "SELL ALL"])].copy()
    buys  = trade_df[trade_df["Action"] == "BUY"].copy()
    holds = trade_df[~trade_df["Action"].isin(["SELL", "SELL ALL", "BUY"])].copy()
    return pd.concat([sells, buys, holds], ignore_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# Step 6 — Cash validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_cash(trade_df: pd.DataFrame, current_cash: float) -> dict:
    """
    Check whether you have enough cash to execute all buys
    (after receiving sell proceeds).
    """
    sell_proceeds = trade_df.loc[
        trade_df["Action"].isin(["SELL", "SELL ALL"]), "Trade Value"
    ].sum()
    buy_outlay = trade_df.loc[
        trade_df["Action"] == "BUY", "Trade Value"
    ].sum()
    total_cost = trade_df["Total Cost"].sum()

    available = current_cash + sell_proceeds
    net_cash  = available - buy_outlay - total_cost
    shortfall = max(0.0, buy_outlay + total_cost - available)

    return {
        "starting_cash":  current_cash,
        "sell_proceeds":  sell_proceeds,
        "buy_outlay":     buy_outlay,
        "total_costs":    total_cost,
        "net_cash":       net_cash,
        "shortfall":      shortfall,
        "ok":             shortfall < 0.01,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Formatted output
# ─────────────────────────────────────────────────────────────────────────────

def print_trade_sheet(
    trade_df: pd.DataFrame,
    current:  dict,
    cash_check: dict,
    latest_date: pd.Timestamp = None,
    currency: str = "$",
) -> None:
    """Print a formatted trade sheet ready to hand to your broker."""
    total = current["total"]
    from datetime import datetime
    today = datetime.now().strftime("%Y-%m-%d")
    sig_date = f"  Signal date : {latest_date.strftime('%Y-%m-%d')}" if latest_date else ""

    print(f"\n{'='*82}")
    print(f"  REBALANCING TRADE SHEET")
    print(f"  Generated   : {today}")
    print(sig_date)
    print(f"  Portfolio   : {currency}{total:>12,.2f}   Cash: {currency}{current['cash']:>10,.2f}")
    print(f"{'='*82}")

    seq = _execution_order(trade_df)
    active = seq[seq["Action"].isin(["BUY", "SELL", "SELL ALL"])]
    holds  = seq[~seq["Action"].isin(["BUY", "SELL", "SELL ALL"])]

    # Header row
    H = (f"  {'ETF':<6}  {'Action':<22}  {'Curr':>5}  {'->':>2}  {'Tgt':>5}  "
         f"{'Delta':>6}  {'Price':>8}  {'TradeVal':>10}  {'Cost':>7}  "
         f"{'CurrWt':>7}  {'TgtWt':>7}")
    print(H)
    print("  " + "─" * 80)

    for _, row in active.iterrows():
        sgn   = "+" if row["Delta"] > 0 else ""
        delta_str = f"{sgn}{int(row['Delta'])}"
        print(
            f"  {row['ETF']:<6}  {row['Action']:<22}  "
            f"{int(row['Curr Shares']):>5}  {'->':>2}  {int(row['Target Shares']):>5}  "
            f"{delta_str:>6}  {currency}{row['Price']:>7.2f}  "
            f"{currency}{row['Trade Value']:>9,.2f}  {currency}{row['Total Cost']:>6.2f}  "
            f"{row['Curr Wt %']:>6.1f}%  {row['Target Wt %']:>6.1f}%"
        )

    if len(holds) > 0:
        print("  " + "─" * 80)
        for _, row in holds.iterrows():
            print(
                f"  {row['ETF']:<6}  {row['Action']:<22}  "
                f"{int(row['Curr Shares']):>5}  {'':>2}  {int(row['Target Shares']):>5}  "
                f"{'':>6}  {currency}{row['Price']:>7.2f}  "
                f"{'':>10}  {'':>7}  "
                f"{row['Curr Wt %']:>6.1f}%  {row['Target Wt %']:>6.1f}%"
            )

    print("  " + "─" * 80)
    print(f"\n  CASH FLOW")
    print(f"  Starting cash      : {currency}{cash_check['starting_cash']:>10,.2f}")
    print(f"  + Sell proceeds    : {currency}{cash_check['sell_proceeds']:>10,.2f}")
    print(f"  - Buy outlay       : {currency}{cash_check['buy_outlay']:>10,.2f}")
    print(f"  - Transaction costs: {currency}{cash_check['total_costs']:>10,.2f}")
    print(f"  ─────────────────────────────────────")
    print(f"  = Net cash after   : {currency}{cash_check['net_cash']:>10,.2f}  (stays uninvested)")

    if not cash_check["ok"]:
        print(f"\n  ⚠  CASH SHORTFALL: {currency}{cash_check['shortfall']:,.2f}")
        print(f"  Scale down buys or add {currency}{cash_check['shortfall']:,.2f} cash first.")
    else:
        print(f"\n  ✓  Cash sufficient for all trades.")
    print(f"{'='*82}")


# ─────────────────────────────────────────────────────────────────────────────
# Top-level runner
# ─────────────────────────────────────────────────────────────────────────────

def run_rebalancer(
    signals:              dict,
    prices_df:            pd.DataFrame,
    returns_df:           pd.DataFrame,
    holdings:             dict[str, float],
    cash:                 float = 0.0,
    n_top:                int   = 5,
    fractional_shares:    bool  = False,
    spread_bps:           float = 2.0,
    commission_per_trade: float = 0.0,
    commission_pct:       float = 0.0,
    min_trade_pct:        float = 0.005,
    currency:             str   = "$",
    proc_dir:             str   = "data/processed",
) -> dict:
    """
    Full rebalancing pipeline.

    Parameters
    ----------
    holdings             : your current positions — {ticker: shares_held}
    cash                 : uninvested cash in your account
    n_top                : number of ETFs to hold (default 5, matches backtest)
    fractional_shares    : True if your broker supports fractional ETF units
    spread_bps           : fallback half-spread estimate (bps) for unlisted ETFs
    commission_per_trade : flat fee per order in local currency
    commission_pct       : percentage of trade value charged as commission
    min_trade_pct        : ignore trades smaller than this × portfolio value
    currency             : symbol for display ("$" or "£")

    Returns
    -------
    dict with keys: trade_df, target_weights, current, cash_check, live_prices
    """

    # 1. Target weights (run signal pipeline on latest data)
    print("\n  [1/5] Computing HRP target weights...")
    target_weights = get_target_weights(signals, prices_df, n_top=n_top, proc_dir=proc_dir)
    latest_date    = prices_df.resample("ME").last().index[-1]
    print(f"        Signal date: {latest_date.strftime('%Y-%m-%d')}")
    for tkr, wt in target_weights.items():
        print(f"        {tkr:<6}: {wt*100:.1f}%")

    # 2. Live prices for all relevant tickers
    all_tickers = sorted(
        set(list(holdings.keys())) | set(list(target_weights.index))
    )
    print(f"\n  [2/5] Fetching live prices for: {', '.join(all_tickers)}")
    live_prices = fetch_live_prices(all_tickers, fallback_prices=prices_df)
    for tkr in all_tickers:
        p = live_prices.get(tkr, float("nan"))
        print(f"        {tkr:<6}: {currency}{p:.2f}")

    # 3. Current portfolio state
    print(f"\n  [3/5] Current portfolio state...")
    current = compute_current_state(holdings, live_prices, cash)
    print(f"        Invested   : {currency}{current['invested']:>10,.2f}")
    print(f"        Cash       : {currency}{current['cash']:>10,.2f}")
    print(f"        Total      : {currency}{current['total']:>10,.2f}")

    # 4. Target shares
    print(f"\n  [4/5] Computing target shares (fractional={fractional_shares})...")
    target = compute_target_shares(
        target_weights, current["total"], live_prices, fractional_shares
    )

    # 5. Trade sheet
    print(f"\n  [5/5] Building trade sheet...")
    trade_df   = build_trade_sheet(
        current, target, live_prices,
        spread_bps=spread_bps,
        commission_per_trade=commission_per_trade,
        commission_pct=commission_pct,
        min_trade_pct=min_trade_pct,
    )
    cash_check = validate_cash(trade_df, current["cash"])

    print_trade_sheet(trade_df, current, cash_check, latest_date, currency)

    return {
        "trade_df":       trade_df,
        "target_weights": target_weights,
        "target_shares":  target,
        "current":        current,
        "cash_check":     cash_check,
        "live_prices":    live_prices,
        "latest_date":    latest_date,
    }
