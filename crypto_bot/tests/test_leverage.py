import logging

import pandas as pd
import pytest

from crypto_bot.backtest.engine import BacktestEngine
from crypto_bot.core.exchange import ExchangeClient, ExchangeConfig
from crypto_bot.core.execution import ExecutionConfig, ExecutionEngine
from crypto_bot.core.leverage import (
    LeverageConfig,
    clamp_stop_to_liquidation,
    futures_symbol,
    is_liquidated,
    liquidation_price,
    spot_symbol,
)
from crypto_bot.core.portfolio import Portfolio, Position
from crypto_bot.core.risk import RiskConfig, RiskManager
from crypto_bot.main import TradingBot

logger = logging.getLogger("test")


def test_config_clamps_leverage_to_max():
    cfg = LeverageConfig.from_settings({"leverage": {"enabled": True, "leverage": 50, "max_leverage": 5}})
    assert cfg.leverage == 5
    assert cfg.effective_leverage == 5
    disabled = LeverageConfig.from_settings({"leverage": {"enabled": False, "leverage": 3}})
    assert disabled.effective_leverage == 1.0
    with pytest.raises(ValueError):
        LeverageConfig.from_settings({"leverage": {"margin_mode": "portfolio"}})


def test_liquidation_price_long_and_short():
    # 10x, mmr %0.5 → fiyat %9.5 ters giderse likidasyon
    assert liquidation_price("BUY", 100.0, 10, 0.005) == pytest.approx(90.5)
    assert liquidation_price("SELL", 100.0, 10, 0.005) == pytest.approx(109.5)
    assert liquidation_price("BUY", 100.0, 1, 0.005) == 0.0
    assert is_liquidated("BUY", 90.0, 90.5)
    assert not is_liquidated("BUY", 91.0, 90.5)
    assert is_liquidated("SELL", 110.0, 109.5)
    assert not is_liquidated("BUY", 1.0, 0.0)


def test_stop_is_pulled_inside_liquidation():
    # 10x → likidasyon mesafesi %9.5, buffer 0.5 → SL en fazla %4.75 uzakta
    assert clamp_stop_to_liquidation("BUY", 100.0, 80.0, 10, 0.005, 0.5) == pytest.approx(95.25)
    assert clamp_stop_to_liquidation("SELL", 100.0, 120.0, 10, 0.005, 0.5) == pytest.approx(104.75)
    # Zaten yeterince yakın olan stop'a dokunulmaz
    assert clamp_stop_to_liquidation("BUY", 100.0, 98.0, 10, 0.005, 0.5) == 98.0


def test_symbol_mapping():
    assert futures_symbol("BTC/USDT", "USDT") == "BTC/USDT:USDT"
    assert futures_symbol("BTC/USDT:USDT", "USDT") == "BTC/USDT:USDT"
    assert spot_symbol("BTC/USDT:USDT") == "BTC/USDT"


def test_exchange_maps_symbols_only_in_futures_mode():
    spot = ExchangeClient(ExchangeConfig(name="binance", api_key="", api_secret=""), logger)
    fut = ExchangeClient(
        ExchangeConfig(name="binance", api_key="", api_secret="", market_type="swap"), logger
    )
    assert spot._sym("BTC/USDT") == "BTC/USDT"
    assert fut._sym("BTC/USDT") == "BTC/USDT:USDT"
    assert fut.client.options["defaultType"] == "swap"


def test_position_size_uses_leverage_for_caps():
    risk = RiskManager(
        RiskConfig(
            risk_per_trade=0.01, stop_loss_pct=0.01, take_profit_pct=0.02,
            max_open_trades=5, max_daily_loss_pct=0.05, max_trade_size_quote=2000,
        ),
        logger,
    )
    # Çok dar stop → risk tabanlı boyut çok büyük, tavan belirleyici olur
    spot_qty = risk.calculate_position_size(10_000, 100.0, 99.9)
    lev_qty = risk.calculate_position_size(10_000, 100.0, 99.9, leverage=3)
    assert spot_qty == pytest.approx(20.0)   # 2000 notional
    assert lev_qty == pytest.approx(60.0)    # 2000 teminat × 3x
    # Geniş stop → risk tabanlı boyut, kaldıraçtan bağımsız
    assert risk.calculate_position_size(10_000, 100.0, 90.0, leverage=3) == pytest.approx(10.0)


def test_partial_close_releases_margin_proportionally():
    pf = Portfolio()
    pf.open_position(Position("BTC/USDT", "BUY", 100.0, 10.0, 0.0, 95.0, 110.0, 100.0,
                              leverage=5, margin=200.0, liquidation_price=80.5))
    trade = pf.partial_close_position("BTC/USDT", 4.0, 105.0)
    assert trade["margin_released"] == pytest.approx(80.0)
    assert pf.positions["BTC/USDT"].margin == pytest.approx(120.0)
    trade = pf.close_position("BTC/USDT", 105.0)
    assert trade["margin_released"] == pytest.approx(120.0)
    assert pf.margin_used() == 0.0


def test_portfolio_snapshot_roundtrip_keeps_leverage():
    pf = Portfolio()
    pf.open_position(Position("ETH/USDT", "SELL", 2000.0, 1.0, 0.0, 2050.0, 1900.0, 2000.0,
                              leverage=3, margin=666.67, liquidation_price=2656.67))
    restored = Portfolio()
    restored.restore(pf.snapshot())
    pos = restored.positions["ETH/USDT"]
    assert (pos.leverage, pos.margin, pos.liquidation_price) == (3, 666.67, 2656.67)


def _paper_bot(cash: float) -> TradingBot:
    bot = TradingBot.__new__(TradingBot)
    bot.mode = "paper"
    bot.paper_cash = cash
    bot.portfolio = Portfolio()
    bot._last_prices = {}
    return bot


@pytest.mark.parametrize("side,exit_price,expected_pnl", [("BUY", 110.0, 100.0), ("SELL", 110.0, -100.0)])
def test_paper_margin_accounting_open_mark_close(side, exit_price, expected_pnl):
    # 10 adet @100, 5x → 200 teminat kilitlenir
    bot = _paper_bot(10_000.0 - 200.0)
    bot.portfolio.open_position(Position("BTC/USDT", side, 100.0, 10.0, 0.0, 0.0, 0.0, 100.0,
                                         leverage=5, margin=200.0))
    bot._last_prices["BTC/USDT"] = exit_price
    assert bot._get_total_equity() == pytest.approx(10_000.0 + expected_pnl)

    trade = bot.portfolio.close_position("BTC/USDT", exit_price)
    bot._apply_paper_close_cash(trade)
    assert bot.paper_cash == pytest.approx(10_000.0 + expected_pnl)


def test_paper_liquidation_wipes_margin():
    bot = _paper_bot(10_000.0 - 200.0)
    bot.settings = {}
    bot.notifier = type("N", (), {"send": lambda self, msg: None})()
    bot.logger = logger
    bot.risk = type("R", (), {"update_protective_levels": lambda self, p, c: (p.stop_loss, p.take_profit)})()
    bot._register_trade_result = lambda name, pnl: None
    liq = liquidation_price("BUY", 100.0, 5, 0.005)
    bot.portfolio.open_position(Position("BTC/USDT", "BUY", 100.0, 10.0, 1.0, 95.0, 120.0, 100.0,
                                         leverage=5, margin=200.0, liquidation_price=liq))
    # Fiyat SL'yi atlayıp likidasyonun altına boşluk yaptı
    assert bot._check_protective_levels("BTC/USDT", 75.0) is True
    assert not bot.portfolio.is_open("BTC/USDT")
    last = bot.portfolio.trade_history[-1]
    assert last["liquidated"] and last["pnl"] == pytest.approx(-201.0)
    assert bot.paper_cash == pytest.approx(9_800.0)  # teminat geri dönmez


class _FakeExchange:
    def __init__(self):
        self.calls = []

    def get_fee_rates(self, symbol):
        return 0.0002, 0.0005

    def validate_order(self, symbol, price, amount):
        return price, amount

    def estimate_fill_price(self, symbol, side, amount, limit=50):
        return 100.0

    def create_market_order_with_params(self, symbol, side, amount, params=None):
        self.calls.append(params or {})
        return {"id": "1"}

    def fetch_order(self, order_id, symbol):
        return {"id": order_id, "status": "closed", "filled": 1.0, "average": 100.0}


class _FakeDB:
    from contextlib import contextmanager

    @contextmanager
    def session_scope(self):
        yield type("S", (), {"add": lambda self, x: None})()


def _live_engine(futures: bool) -> tuple[ExecutionEngine, _FakeExchange]:
    ex = _FakeExchange()
    cfg = ExecutionConfig(
        mode="live", order_type="market", max_retries=1, retry_delay_seconds=0, poll_interval_seconds=0,
        max_slippage_pct=0.0, slippage_model="static", slippage_volatility_factor=0, slippage_volume_factor=0,
        use_order_book=False, order_book_depth=5, simulate_order_delay_seconds=0, partial_fill_min_ratio=1.0,
        order_fill_timeout_seconds=1, max_execution_drift_pct=0.01, futures=futures,
    )
    return ExecutionEngine(ex, _FakeDB(), logger, cfg), ex


def test_close_orders_are_reduce_only_on_futures():
    engine, ex = _live_engine(futures=True)
    engine.place_order("BTC/USDT", "sell", 1.0, 100.0, reduce_only=True)
    engine.place_order("BTC/USDT", "buy", 1.0, 100.0)
    assert ex.calls[0].get("reduceOnly") is True
    assert "reduceOnly" not in ex.calls[1]

    spot_engine, spot_ex = _live_engine(futures=False)
    spot_engine.place_order("BTC/USDT", "sell", 1.0, 100.0, reduce_only=True)
    assert "reduceOnly" not in spot_ex.calls[0]


def _crash_data() -> pd.DataFrame:
    n = 200
    close = [100.0] * 120 + [100.0 - (i + 1) * 2.0 for i in range(80)]
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"),
        "open": close, "high": [c + 0.5 for c in close], "low": [c - 0.5 for c in close],
        "close": close, "volume": [1_000.0] * n,
    })


class _BuyOnce:
    params: dict = {}

    def __init__(self):
        self.fired = False

    def generate_signal(self, window):
        if not self.fired and len(window) >= 100:
            self.fired = True
            return "BUY"
        return "HOLD"


def test_backtest_leverage_1_matches_spot():
    data = _crash_data()
    spot = BacktestEngine(_BuyOnce(), 10_000, execution_delay_candles=0, stop_loss_pct=0.05).run(data)
    lev1 = BacktestEngine(_BuyOnce(), 10_000, execution_delay_candles=0, stop_loss_pct=0.05, leverage=1).run(data)
    assert lev1.metrics["final_equity"] == pytest.approx(spot.metrics["final_equity"])
    assert lev1.metrics["liquidations"] == 0


def test_backtest_liquidates_without_stop():
    result = BacktestEngine(_BuyOnce(), 10_000, execution_delay_candles=0, leverage=10).run(_crash_data())
    assert result.metrics["liquidations"] == 1
    # All-in 10x: teminatın tamamı + giriş ücreti kaybedilir, hesap sıfırlanmaz
    assert 0 < result.metrics["final_equity"] < 10_000


def test_backtest_stop_is_clamped_before_liquidation():
    # %20 stop 10x'te likidasyonun ötesinde → motor stop'u içeri çeker, likidasyon olmaz
    result = BacktestEngine(
        _BuyOnce(), 10_000, execution_delay_candles=0, stop_loss_pct=0.20, risk_per_trade=0.02, leverage=10,
    ).run(_crash_data())
    assert result.metrics["liquidations"] == 0
    assert result.trades.iloc[0]["reason"] == "stop_loss"
