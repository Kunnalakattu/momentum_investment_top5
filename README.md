# Momentum Investment System — Top 5 ETF Portfolio

A systematic, rules-based momentum strategy that selects the top 5 stocks from a 63-stock US equity universe each month, filtered by a 200-day moving average trend rule and weighted using Hierarchical Risk Parity (HRP). All stocks trade on NYSE/NASDAQ and are accessible globally.

## Strategy Summary

| Metric | Value |
|--------|-------|
| Universe | 63 US stocks across 10 sectors |
| Selection | Top 5 stocks by composite momentum score (3M / 6M / 12M) |
| Filter | Price > 200-day moving average |
| Weighting | Hierarchical Risk Parity (HRP) |
| Rebalancing | Monthly |
| Backtest CAGR | ~12% |
| Backtest Sharpe | ~1.08 |
| Max Drawdown | ~-13% |
| Governance | 7/8 PASS → GO |

## Universe (63 US stocks, NYSE/NASDAQ)

| Sector | Tickers |
|--------|---------|
| Technology | AAPL, MSFT, NVDA, AMZN, META, GOOGL, AVGO, ORCL, CRM, ADBE |
| Semiconductors | AMD, TSM, QCOM, MU, TXN |
| Financials | JPM, BAC, GS, MS, BLK, V, MA |
| Healthcare | LLY, UNH, JNJ, ABBV, MRK |
| Consumer | COST, WMT, PG, KO, PEP |
| Industrials | CAT, GE, RTX, DE, HON, LIN |
| Energy | XOM, CVX, COP, SLB, EOG |
| Communication | NFLX, DIS, UBER, PLTR, SPOT, TMUS |
| Commodities | NEM, GOLD, AEM, FCX, SCCO, RIO, BHP, VALE, ADM, BG, MOS, NTR |
| Index/Other | SPY (benchmark), BRK-B |

## Project Structure

```
momentum_investment_top5/
├── main.py                    # Full research pipeline (run once)
├── requirements.txt
├── config/
│   └── universe.yaml          # ETF universe and parameters
├── src/
│   ├── data.py                # Download, clean, refresh data
│   ├── signals.py             # Momentum signals + 200DMA filter
│   ├── portfolio.py           # Asset selection + HRP weighting
│   ├── backtest.py            # Strategy backtesting
│   ├── robustness.py          # Parameter sensitivity
│   ├── stats_validation.py    # IC, bootstrap, hypothesis tests
│   ├── walk_forward.py        # Out-of-sample walk-forward
│   ├── regime_analysis.py     # Performance across market regimes
│   ├── risk_attribution.py    # Risk decomposition
│   ├── factor_attribution.py  # Fama-French alpha / factor loadings
│   ├── stress_testing.py      # Historical crisis analysis
│   ├── capacity_analysis.py   # AUM scalability (£10k → £100M)
│   ├── cost_sensitivity.py    # Net returns at 0–100bp costs
│   ├── monitoring.py          # Monthly performance report
│   ├── governance.py          # GO/NO-GO model approval checklist
│   ├── rebalancer.py          # Trade sheet engine
│   └── health.py              # Strategy health dashboard (traffic lights)
└── notebooks/
    ├── 01_data_download.ipynb
    ├── 02_signal_research.ipynb
    ├── 03_hypothesis_tests.ipynb
    ├── 04_backtest.ipynb
    ├── 05_live_portfolio.ipynb
    ├── 06_robustness.ipynb
    ├── 07_stats_validation.ipynb
    ├── 08_walk_forward.ipynb
    ├── 09_regime_analysis.ipynb
    ├── 10_risk_attribution.ipynb
    ├── 11_factor_attribution.ipynb
    ├── 12_stress_testing.ipynb
    ├── 13_capacity_analysis.ipynb
    ├── 14_cost_sensitivity.ipynb
    ├── 15_monitoring.ipynb
    ├── 16_research_ideas.ipynb
    ├── 17_governance.ipynb
    ├── 18_rebalancing.ipynb   ← run monthly (trade sheet)
    └── 19_health_dashboard.ipynb  ← run monthly (health check)
```

## Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the full research pipeline (once)

```bash
python main.py
```

This downloads data, runs all analysis phases, and saves results to `data/processed/`.

### 3. Monthly workflow (20–30 min)

Open these two notebooks each month-end, in order:

```
notebooks/19_health_dashboard.ipynb   # Check all 6 health metrics
notebooks/18_rebalancing.ipynb        # Get the trade sheet
```

Both notebooks automatically refresh data and recompute signals — no need to re-run `main.py`.

### 4. Annual review

Re-run `notebooks/17_governance.ipynb` to get a fresh GO/NO-GO verdict using the full year of new data.

## Monthly Rebalancing Workflow

1. Open `notebooks/18_rebalancing.ipynb`
2. Edit **Cell 2** — enter your current share holdings and cash balance
3. Edit **Cell 3** — set your broker settings (fractional shares, commission)
4. Run all cells → receive an exact trade sheet (sells first, then buys)

## Health Dashboard Thresholds

| Metric | Green | Amber | Red |
|--------|-------|-------|-----|
| Max Drawdown | Better than historical worst | Within 10% of worst | New all-time low |
| Rolling Sharpe (36m) | > 0.7 | 0.5–0.7 | < 0.5 |
| Info Coefficient (12m) | > 0.02 | 0–0.02 | Negative |
| Monthly Turnover | ≤ 1.5× avg | 1.5–2× avg | > 2× avg |
| Data Quality | 0 missing days | 1–4 missing | ≥ 5 or corrupted |
| ETF Availability | All present | 1 issue | Delisted/illiquid |

Thresholds are fixed before investing and never adjusted retroactively.

## Governance Checklist Results

| Test | Result |
|------|--------|
| Walk-Forward Validation | ✓ PASS |
| Bootstrap Confidence Intervals | ✓ PASS |
| Parameter Sensitivity | ✓ PASS |
| Cost Sensitivity | ✓ PASS |
| Regime Analysis | ✓ PASS |
| Factor Attribution | ✓ PASS |
| Capacity Analysis (£100k) | ✓ PASS |
| Information Coefficient | ✓ PASS |

**Final verdict: GO**

## Risk Warnings

- Past backtest performance does not guarantee future results.
- This system is for informational purposes. Always do your own due diligence.
- Stocks trade in USD on NYSE/NASDAQ. Non-US investors carry currency risk.
- The strategy can underperform the S&P 500 during narrow-breadth bull markets (e.g. 2023–2024 mega-cap tech dominance).
- At small portfolio sizes, high share prices (e.g. COST ~$900, NVR) mean whole-share rounding can leave significant cash uninvested — enable fractional shares if your broker supports it.
- Newer stocks (PLTR, UBER, SPOT) may have limited backtest history and will be excluded from signals until they have sufficient data.
