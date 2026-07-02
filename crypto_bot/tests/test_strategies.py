import pandas as pd

from crypto_bot.strategies.ema_rsi_volume_strategy import EmaRsiVolumeStrategy
from crypto_bot.strategies.ema_strategy import EMACrossoverStrategy
from crypto_bot.strategies.rsi_strategy import RSIStrategy


def test_ema_generates_signal():
    data = pd.DataFrame(
        {
            "close": [100 + i for i in range(60)],
            "timestamp": pd.date_range("2024-01-01", periods=60, freq="min", tz="UTC"),
        }
    )
    strategy = EMACrossoverStrategy({"fast_period": 5, "slow_period": 20, "primary_timeframe": "1m"})
    signal = strategy.generate_signal(data)
    assert signal in {"BUY", "SELL", "HOLD"}


def test_rsi_generates_signal():
    prices = [100] * 10 + [90] * 10 + [110] * 20
    data = pd.DataFrame(
        {
            "close": prices,
            "timestamp": pd.date_range("2024-01-01", periods=len(prices), freq="min", tz="UTC"),
        }
    )
    strategy = RSIStrategy({"period": 14, "oversold": 30, "overbought": 70, "primary_timeframe": "1m"})
    signal = strategy.generate_signal(data)
    assert signal in {"BUY", "SELL", "HOLD"}


def test_ema_rsi_volume_generates_decision():
    data = pd.DataFrame(
        {
            "close": [100 + (i * 0.2) for i in range(120)],
            "volume": [1000 + (i % 20) * 10 for i in range(120)],
            "timestamp": pd.date_range("2024-01-01", periods=120, freq="min", tz="UTC"),
        }
    )
    strategy = EmaRsiVolumeStrategy(
        {
            "primary_timeframe": "1m",
            "confirm_timeframe": "5m",
            "ema_fast": 12,
            "ema_slow": 26,
            "rsi_period": 14,
            "rsi_lower": 35,
            "rsi_upper": 70,
            "volume_window": 20,
            "min_volume_ratio": 0.8,
        }
    )
    decision = strategy.generate_decision({"1m": data, "5m": data.iloc[::5].reset_index(drop=True)})
    assert decision.signal in {"BUY", "SELL", "HOLD"}
    assert isinstance(decision.reason, str)
