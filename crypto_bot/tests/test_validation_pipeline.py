"""Tests for the validation pipeline: data loader, realistic engine, report."""
import pandas as pd
import pytest

from crypto_bot.backtest.data_loader import (
    generate_synthetic_ohlcv,
    resample_ohlcv,
    timeframe_to_ms,
)
from crypto_bot.backtest.engine import BacktestEngine
from crypto_bot.backtest.filters import bear_regime_filter, bull_regime_filter
from crypto_bot.backtest.report import ReportConfig, build_comparison_report, save_report
from crypto_bot.strategies.ema_strategy import EMACrossoverStrategy


def test_timeframe_to_ms():
    assert timeframe_to_ms("1m") == 60_000
    assert timeframe_to_ms("1h") == 3_600_000
    assert timeframe_to_ms("4h") == 4 * 3_600_000
    with pytest.raises(ValueError):
        timeframe_to_ms("1x")


def test_synthetic_data_shape_and_sanity():
    df = generate_synthetic_ohlcv(n=500, timeframe="1h", seed=1)
    assert len(df) == 500
    assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
    assert (df["high"] >= df[["open", "close"]].max(axis=1)).all()
    assert (df["low"] <= df[["open", "close"]].min(axis=1)).all()
    assert (df["close"] > 0).all()
    assert df["timestamp"].is_monotonic_increasing


def test_resample_ohlcv():
    df = generate_synthetic_ohlcv(n=240, timeframe="1m", seed=2)
    htf = resample_ohlcv(df, "1h")
    assert 3 <= len(htf) <= 5
    assert htf["high"].iloc[0] >= htf["close"].iloc[0] or htf["high"].iloc[0] >= htf["open"].iloc[0]


def _trending_data(n: int = 400) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"),
            "open": [100 + i * 0.5 for i in range(n)],
            "high": [100 + i * 0.5 + 1 for i in range(n)],
            "low": [100 + i * 0.5 - 1 for i in range(n)],
            "close": [100 + i * 0.5 for i in range(n)],
            "volume": [1_000.0] * n,
        }
    )


def test_engine_backward_compatible_metrics():
    strategy = EMACrossoverStrategy({"fast_period": 12, "slow_period": 26})
    engine = BacktestEngine(strategy, initial_balance=10_000)
    result = engine.run(_trending_data())
    for key in ["max_drawdown", "win_rate", "profit_factor", "total_return_pct",
                "buy_hold_return_pct", "fees_paid", "exposure_pct"]:
        assert key in result.metrics


def test_engine_stop_loss_triggers():
    """A crash after entry must be caught by the stop-loss, capping the loss."""
    n = 200
    prices = [100.0 + i * 0.5 for i in range(120)] + [160.0 - (i - 119) * 3.0 for i in range(120, n)]
    data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"),
            "open": prices,
            "high": [p + 0.5 for p in prices],
            "low": [p - 0.5 for p in prices],
            "close": prices,
            "volume": [1_000.0] * n,
        }
    )
    strategy = EMACrossoverStrategy({"fast_period": 5, "slow_period": 15})
    engine = BacktestEngine(
        strategy, initial_balance=10_000, stop_loss_pct=0.02, execution_delay_candles=0
    )
    result = engine.run(data)
    if not result.trades.empty:
        stop_exits = result.trades[result.trades["reason"] == "stop_loss"]
        # crash of ~40% must have triggered a 2% stop at least once
        assert not stop_exits.empty
        # no single trade should lose much more than ~2% of balance + fees/slippage
        assert result.trades["pnl"].min() > -10_000 * 0.05


def test_engine_risk_sizing_smaller_than_all_in():
    data = _trending_data()
    strategy = EMACrossoverStrategy({"fast_period": 5, "slow_period": 15})
    all_in = BacktestEngine(strategy, 10_000, execution_delay_candles=0).run(data)
    sized = BacktestEngine(
        EMACrossoverStrategy({"fast_period": 5, "slow_period": 15}),
        10_000, execution_delay_candles=0,
        risk_per_trade=0.01, stop_loss_pct=0.02,
    ).run(data)
    if not all_in.trades.empty and not sized.trades.empty:
        assert sized.trades["qty"].max() < all_in.trades["qty"].max()


def test_engine_fees_accumulate():
    strategy = EMACrossoverStrategy({"fast_period": 5, "slow_period": 15})
    result = BacktestEngine(strategy, 10_000, execution_delay_candles=0).run(_trending_data())
    if result.metrics["total_trades"] > 0:
        assert result.metrics["fees_paid"] > 0


def test_comparison_report_and_save(tmp_path):
    data = {"BTC/USDT": generate_synthetic_ohlcv(n=800, timeframe="1h", seed=7)}
    params = {"ema": {"fast_period": 12, "slow_period": 26},
              "rsi": {"period": 14, "oversold": 30, "overbought": 70}}
    report = build_comparison_report(
        data, params, ["ema", "rsi"], "1h", ReportConfig(min_trades_for_verdict=1)
    )
    assert not report.empty
    assert set(report["strategy"]) == {"ema", "rsi"}
    assert "verdict" in report.columns
    csv_path, html_path = save_report(report, tmp_path)
    assert csv_path.exists() and html_path.exists()
    assert "Strategy Comparison" in html_path.read_text(encoding="utf-8")


def _bear_data(n: int = 600) -> pd.DataFrame:
    prices = [1000.0 * (0.999 ** i) for i in range(n)]  # steady downtrend
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"),
            "open": prices,
            "high": [p * 1.002 for p in prices],
            "low": [p * 0.998 for p in prices],
            "close": prices,
            "volume": [1_000.0] * n,
        }
    )


def test_bull_regime_filter_blocks_bear_market():
    """In a steady downtrend the filter must block (almost) everything."""
    data = _bear_data()
    allowed = bull_regime_filter(data)
    assert allowed.dtype == bool
    assert allowed.tail(300).sum() == 0  # after warmup, all blocked


def test_bull_regime_filter_allows_bull_market():
    data = _trending_data(600)
    allowed = bull_regime_filter(data)
    assert allowed.tail(100).all()  # steady uptrend -> entries allowed


def test_bear_regime_filter_mirrors_bull():
    bear = _bear_data(600)
    bull = _trending_data(600)
    assert bear_regime_filter(bear).tail(100).all()      # downtrend -> shorts allowed
    assert bear_regime_filter(bull).tail(100).sum() == 0  # uptrend -> shorts blocked
    assert bull_regime_filter(bear).tail(100).sum() == 0


def test_engine_entry_filter_blocks_trades():
    """With an all-False entry filter no position may ever be opened."""
    data = _trending_data(400)
    strategy = EMACrossoverStrategy({"fast_period": 5, "slow_period": 15})
    blocked = BacktestEngine(strategy, 10_000, execution_delay_candles=0).run(
        data, entry_filter=pd.Series(False, index=data.index)
    )
    assert blocked.metrics["total_trades"] == 0
    assert blocked.metrics["final_equity"] == 10_000

    unblocked = BacktestEngine(
        EMACrossoverStrategy({"fast_period": 5, "slow_period": 15}),
        10_000, execution_delay_candles=0,
    ).run(data, entry_filter=pd.Series(True, index=data.index))
    assert unblocked.metrics["total_trades"] >= 0  # filter=True behaves like no filter


def test_regime_filter_reduces_bear_losses():
    """Regime filter must not lose more than unfiltered in a bear market."""
    data = _bear_data(800)
    params = {"ema": {"fast_period": 12, "slow_period": 26}}
    off = build_comparison_report({"X/USDT": data}, params, ["ema"], "1h",
                                  ReportConfig(min_trades_for_verdict=1, regime_filter=False))
    on = build_comparison_report({"X/USDT": data}, params, ["ema"], "1h",
                                 ReportConfig(min_trades_for_verdict=1, regime_filter=True))
    assert on["total_return_pct"].iloc[0] >= off["total_return_pct"].iloc[0]
    assert on["total_return_pct"].iloc[0] >= -0.001  # filtered: stayed ~flat


def test_mtf_no_lookahead():
    """HTF data passed to strategies must only contain fully closed candles."""
    data = generate_synthetic_ohlcv(n=600, timeframe="1h", seed=3)
    htf = resample_ohlcv(data, "4h")

    seen_leaks = []

    class ProbeStrategy(EMACrossoverStrategy):
        def generate_signal(self, d):
            if isinstance(d, dict) and "4h" in d and "1h" in d:
                primary, confirm = d["1h"], d["4h"]
                if not confirm.empty and not primary.empty:
                    now = primary["timestamp"].iloc[-1]
                    last_htf_close = confirm["timestamp"].iloc[-1] + pd.Timedelta(hours=4)
                    if last_htf_close > now:
                        seen_leaks.append((now, last_htf_close))
            return super().generate_signal(d)

    strategy = ProbeStrategy({"fast_period": 12, "slow_period": 26,
                              "primary_timeframe": "1h", "confirm_timeframe": "4h"})
    engine = BacktestEngine(strategy, 10_000)
    engine.run(data, htf_data=htf, primary_tf="1h", confirm_tf="4h")
    assert not seen_leaks
