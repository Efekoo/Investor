import pandas as pd

from crypto_bot.backtest.engine import BacktestEngine
from crypto_bot.strategies.ema_strategy import EMACrossoverStrategy


def test_backtest_runs_and_returns_metrics():
    data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=300, freq="min", tz="UTC"),
            "open": [100 + i * 0.1 for i in range(300)],
            "high": [100 + i * 0.1 + 1 for i in range(300)],
            "low": [100 + i * 0.1 - 1 for i in range(300)],
            "close": [100 + i * 0.1 for i in range(300)],
            "volume": [1_000 for _ in range(300)],
        }
    )
    strategy = EMACrossoverStrategy({"fast_period": 12, "slow_period": 26, "primary_timeframe": "1m"})
    engine = BacktestEngine(
        strategy,
        initial_balance=10_000,
        maker_fee_pct=0.0002,
        taker_fee_pct=0.0006,
        slippage_pct=0.001,
        execution_delay_candles=1,
    )
    result = engine.run(data)
    assert "max_drawdown" in result.metrics
    assert "win_rate" in result.metrics
    assert "profit_factor" in result.metrics
