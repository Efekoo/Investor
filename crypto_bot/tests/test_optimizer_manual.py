import pandas as pd
import numpy as np
from crypto_bot.backtest.optimizer import StrategyOptimizer
from crypto_bot.strategies.ema_strategy import EMACrossoverStrategy

def generate_trending_data(n=1000):
    """EMA stratejisini test etmek için trend içeren veri oluşturur."""
    np.random.seed(42)
    t = np.linspace(0, 10, n)
    # Sinus dalgası üzerine trend ve gürültü ekle
    close = 100 + 10 * np.sin(t) + 2 * t + np.random.normal(0, 0.5, n)
    data = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="min", tz="UTC"),
        "open": close - 0.1,
        "high": close + 0.5,
        "low": close - 0.5,
        "close": close,
        "volume": np.random.randint(100, 1000, n),
    })
    return data

def test_optimization():
    data = generate_trending_data(2000)
    optimizer = StrategyOptimizer(EMACrossoverStrategy, data, metric="final_equity")
    
    # Denenecek parametreler
    param_grid = {
        "fast_period": [5, 10, 15],
        "slow_period": [20, 30, 40],
        "primary_timeframe": ["1m"]
    }
    
    print("\n--- Testing Grid Search ---")
    result = optimizer.run_grid_search(param_grid)
    print(f"Best Params: {result['best_params']}")
    print(f"Best Metric: {result['best_metric']:.2f}")

    print("\n--- Testing Walk-Forward Analysis ---")
    wf_results = optimizer.walk_forward_analysis(param_grid, n_folds=2)
    for res in wf_results:
        print(f"Fold {res['fold']}: Train Metric: {res['train_metric']:.2f}, Test Metric: {res['test_metric']:.2f}, Params: {res['best_params']}")

if __name__ == "__main__":
    test_optimization()
