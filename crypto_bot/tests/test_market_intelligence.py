import pandas as pd

from crypto_bot.core.performance import PerformanceConfig, StrategyPerformanceTracker
from crypto_bot.core.regime import RegimeDetector
from crypto_bot.strategies.ema_strategy import EMACrossoverStrategy
from crypto_bot.strategies.rsi_strategy import RSIStrategy


def test_regime_detector_returns_expected_labels():
    detector = RegimeDetector(
        atr_period=14,
        adx_period=14,
        trend_adx_threshold=20,
        high_volatility_threshold=0.03,
        low_volatility_threshold=0.001,
    )
    data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=120, freq="min", tz="UTC"),
            "open": [100 + i * 0.2 for i in range(120)],
            "high": [100 + i * 0.2 + 1 for i in range(120)],
            "low": [100 + i * 0.2 - 1 for i in range(120)],
            "close": [100 + i * 0.2 for i in range(120)],
            "volume": [1000 for _ in range(120)],
        }
    )
    state = detector.detect(data)
    assert state.regime in {"TREND", "RANGE", "VOLATILE"}


def test_strategy_supported_regimes_defined():
    ema = EMACrossoverStrategy({"fast_period": 12, "slow_period": 26, "primary_timeframe": "1m"})
    rsi = RSIStrategy({"period": 14, "oversold": 30, "overbought": 70, "primary_timeframe": "1m"})
    assert "TREND" in ema.supported_regimes
    assert "RANGE" in rsi.supported_regimes


def test_performance_tracker_auto_disables_underperformer():
    tracker = StrategyPerformanceTracker(
        PerformanceConfig(
            min_trades_for_eval=3,
            min_win_rate=0.5,
            max_drawdown=0.2,
            min_pnl=0.0,
            drawdown_risk_multiplier=0.7,
            high_volatility_risk_multiplier=0.6,
            stable_profit_risk_multiplier=1.05,
        )
    )
    tracker.record_trade("ema", -10)
    tracker.record_trade("ema", -5)
    tracker.record_trade("ema", -3)
    disabled, _ = tracker.is_disabled("ema")
    assert disabled
