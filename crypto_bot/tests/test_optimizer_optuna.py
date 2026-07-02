import pandas as pd
import numpy as np
from crypto_bot.backtest.optimizer import StrategyOptimizer
from crypto_bot.strategies.ema_strategy import EMACrossoverStrategy

def generate_complex_data(n=2000):
    np.random.seed(42)
    t = np.linspace(0, 20, n)
    # Birden fazla dalga ve trend bileşeni ekleyerek daha gerçekçi bir veri oluştur
    close = 100 + 15 * np.sin(t) + 5 * np.sin(2*t) + 3 * t + np.random.normal(0, 0.8, n)
    data = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="min", tz="UTC"),
        "open": close - 0.1,
        "high": close + 0.5,
        "low": close - 0.5,
        "close": close,
        "volume": np.random.randint(100, 1000, n),
    })
    return data

def test_optuna_optimization():
    data = generate_complex_data(3000)
    # Sharpe Ratio üzerinden optimize et
    optimizer = StrategyOptimizer(EMACrossoverStrategy, data, metric="sharpe_ratio")
    
    # Optuna parametre aralıkları: (min, max, type)
    param_ranges = {
        "fast_period": (2, 30, "int"),
        "slow_period": (31, 100, "int"),
        "primary_timeframe": (["1m"], None, "categorical")
    }
    
    print("\n--- Testing Optuna Search (Sharpe Ratio) ---")
    result = optimizer.run_optuna_search(param_ranges, n_trials=30)
    print(f"Best Params: {result['best_params']}")
    print(f"Best Sharpe Ratio: {result['best_metric']:.4f}")

    print("\n--- Testing Walk-Forward Analysis with Optuna ---")
    wf_results = optimizer.walk_forward_analysis(param_ranges, n_folds=2, n_trials=20)
    for res in wf_results:
        print(f"Fold {res['fold']}: Train Sharpe: {res['train_metric']:.4f}, Test Sharpe: {res['test_metric']:.4f}, Drawdown: {res['test_drawdown']:.4f}")

if __name__ == "__main__":
    test_optuna_optimization()
