from src.data import run_data_pipeline, load_prices, compute_returns
from src.signals import run_signal_pipeline, load_signals
from src.portfolio import run_portfolio_pipeline
from src.backtest import run_backtest_pipeline
from src.robustness import run_robustness_analysis
from src.stats_validation import run_stats_validation
from src.walk_forward import run_walk_forward_pipeline
from src.regime_analysis import run_regime_analysis
from src.risk_attribution import run_risk_attribution
from src.factor_attribution import run_factor_attribution
from src.stress_testing import run_stress_testing
from src.capacity_analysis import run_capacity_analysis
from src.cost_sensitivity import run_cost_sensitivity_pipeline
from src.monitoring import run_monitoring
from src.governance import run_governance
from src.health import run_health_check

if __name__ == "__main__":
    run_data_pipeline()

    prices  = load_prices()
    returns = compute_returns(prices)
    signals = run_signal_pipeline(prices, returns)

    run_portfolio_pipeline(signals, prices, returns, ns=[5, 7])
    run_backtest_pipeline(signals, prices, returns)
    run_robustness_analysis(prices, returns)
    run_stats_validation(signals, prices, returns)
    run_walk_forward_pipeline(prices, returns)
    run_regime_analysis(prices, returns)
    run_risk_attribution(signals, prices, returns)
    run_factor_attribution(prices, returns)
    run_stress_testing(prices, returns)
    run_capacity_analysis(prices, returns)
    run_cost_sensitivity_pipeline(prices, returns)
    run_monitoring(signals, prices, returns)
    run_governance(signals, prices, returns)
    run_health_check(signals, prices, returns)
