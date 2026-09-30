import logging

import pandas as pd
import pytest
import yaml

from crypto_bot.backtest import replay as rp
from crypto_bot.backtest.data_loader import generate_synthetic_ohlcv
from crypto_bot.core.execution import ExecutionConfig, ExecutionEngine
from crypto_bot.core.portfolio import Portfolio, Position
from crypto_bot.main import TradingBot

SETTINGS = yaml.safe_load(open(rp.DEFAULT_SETTINGS, encoding="utf-8"))


def test_retime_settings_moves_strategies_and_confirm_timeframes():
    s = rp.retime_settings(SETTINGS, "15m")
    assert s["trading"]["timeframe"] == "15m"
    for params in s["strategy"]["params"].values():
        assert params["primary_timeframe"] == "15m"
        if "confirm_timeframe" in params:
            assert rp.timeframe_to_ms(params["confirm_timeframe"]) > rp.timeframe_to_ms("15m")
    # higher_timeframe ana zaman diliminden büyük kalmalı
    assert rp.timeframe_to_ms(rp.retime_settings(SETTINGS, "1h")["trading"]["higher_timeframe"]) > rp.timeframe_to_ms("1h")
    assert SETTINGS["trading"]["timeframe"] == "1m"  # orijinal değişmez


def _synthetic_data(days: int = 12) -> tuple[dict, pd.Timestamp]:
    settings = rp.retime_settings(SETTINGS, "15m")
    tfs = {"15m", settings["trading"]["higher_timeframe"]}
    for p in settings["strategy"]["params"].values():
        tfs.update(x for x in (p.get("primary_timeframe"), p.get("confirm_timeframe")) if x)
    data = {}
    for i, sym in enumerate(["BTC/USDT", "ETH/USDT"]):
        df = generate_synthetic_ohlcv(n=4 * 24 * days, timeframe="15m", volatility=0.004, seed=i + 3)
        data[sym] = rp.prepare_symbol_data(df, sorted(tfs, key=rp.timeframe_to_ms), "15m")
    start = df["timestamp"].iloc[0] + pd.Timedelta(days=days - 3)
    return data, start


def test_replay_runs_bot_pipeline_deterministically():
    data, start = _synthetic_data()
    settings = rp.retime_settings(SETTINGS, "15m")
    a = rp.run_replay(settings, data, start, leverage=3, seed=1)
    b = rp.run_replay(settings, data, start, leverage=3, seed=1)
    assert a.metrics["final_equity"] == pytest.approx(b.metrics["final_equity"])
    assert a.metrics["days"] == pytest.approx(3.0, abs=0.1)
    assert a.metrics["errors"] == 0
    # Simülasyon saati: işlem zamanları gerçek saat değil, veri zamanıdır
    if not a.trades.empty:
        opened = pd.to_datetime(a.trades["opened_at"], utc=True)
        assert opened.min() >= start
    # Sonda açık pozisyon kalmaz; varlık = nakit
    assert a.equity["equity"].iloc[-1] == pytest.approx(a.metrics["final_equity"])
    assert logging.getLogger("crypto_bot").level != logging.CRITICAL  # log seviyesi geri yüklendi


def test_replay_charges_funding_on_open_positions():
    data, start = _synthetic_data()
    for sd in data.values():
        idx = pd.date_range(start - pd.Timedelta(days=10), periods=200, freq="8h", tz="UTC")
        sd.set_funding(pd.Series(0.01, index=idx))  # uç değer: %1 / 8 saat
    res = rp.run_replay(rp.retime_settings(SETTINGS, "15m"), data, start, leverage=3, seed=1)
    if res.metrics["entries"]:
        assert res.metrics["funding"] != 0.0


def _paper_bot() -> TradingBot:
    bot = TradingBot.__new__(TradingBot)
    bot.mode = "paper"
    bot.paper_cash = 10_000.0
    bot.portfolio = Portfolio()
    bot._last_prices = {}
    bot.settings = {"risk": {"max_slippage_pct": 0.001}, "execution": {"fee_pct": 0.001}}
    bot.exchange = rp.ReplayExchange(maker_fee=0.0002, taker_fee=0.0005, spread_pct=0.0, futures=False)
    bot.notifier = rp._SilentNotifier()
    bot.logger = logging.getLogger("test")
    bot.risk = type("R", (), {"update_protective_levels": lambda self, p, c: (p.stop_loss, p.take_profit)})()
    bot._register_trade_result = lambda name, pnl: None
    return bot


def test_paper_stop_loss_charges_taker_fee_and_slippage():
    bot = _paper_bot()
    bot.paper_cash -= 1_000.0
    bot.portfolio.open_position(Position("BTC/USDT", "BUY", 100.0, 10.0, 0.0, 95.0, 120.0, 100.0))
    assert bot._check_protective_levels("BTC/USDT", 94.0) is True
    trade = bot.portfolio.trade_history[-1]
    fill = 94.0 * (1 - 0.001)
    assert trade["exit_price"] == pytest.approx(fill)
    assert trade["fee_paid"] == pytest.approx(10 * fill * 0.0005)
    assert bot.paper_cash == pytest.approx(9_000 + 10 * fill - 10 * fill * 0.0005)


def test_paper_take_profit_fills_at_limit_price_with_maker_fee():
    bot = _paper_bot()
    bot.portfolio.open_position(Position("BTC/USDT", "SELL", 100.0, 10.0, 0.0, 105.0, 90.0, 100.0))
    bot._check_protective_levels("BTC/USDT", 85.0)  # fiyat TP'nin ötesine sıçradı
    trade = bot.portfolio.trade_history[-1]
    assert trade["exit_price"] == 90.0  # limit kendi fiyatından dolar
    assert trade["fee_paid"] == pytest.approx(10 * 90 * 0.0002)
    assert trade["pnl"] == pytest.approx(100 - 10 * 90 * 0.0002)


def test_paper_partial_take_profit_uses_target_price():
    bot = _paper_bot()
    targets = [{"price": 102.0, "close_pct": 0.5, "hit": False}, {"price": 104.0, "close_pct": 1.0, "hit": False}]
    bot.portfolio.open_position(Position("ETH/USDT", "BUY", 100.0, 10.0, 0.0, 95.0, 110.0, 100.0, partial_tp_targets=targets))
    bot._check_protective_levels("ETH/USDT", 103.0)
    trade = bot.portfolio.trade_history[-1]
    assert trade["exit_price"] == 102.0 and trade["qty"] == pytest.approx(5.0)
    assert bot.portfolio.positions["ETH/USDT"].qty == pytest.approx(5.0)


def test_slippage_uses_quote_volume_so_btc_orders_are_not_aborted():
    engine = ExecutionEngine(
        rp.ReplayExchange(0.0002, 0.0005, 0.0, futures=False), None, logging.getLogger("test"),
        ExecutionConfig(
            mode="paper", order_type="market", max_retries=1, retry_delay_seconds=0, poll_interval_seconds=0,
            max_slippage_pct=0.001, slippage_model="dynamic", slippage_volatility_factor=6.0,
            slippage_volume_factor=300.0, use_order_book=False, order_book_depth=5,
            simulate_order_delay_seconds=0, partial_fill_min_ratio=1.0, order_fill_timeout_seconds=1,
            max_execution_drift_pct=0.003,
        ),
    )
    engine._save_order = lambda *a, **k: None
    # BTC 1m: ~37 BTC hacim ≈ 3M$ → eskiden kayma %0,5'e dayanıp işlem iptal ediliyordu
    order = engine.place_order("BTC/USDT", "buy", 0.01, 84_000.0,
                               market_context={"volatility": 0.0005, "volume": 37.0, "quote_volume": 37.0 * 84_000})
    assert order["status"] == "FILLED"
    assert order["execution_details"]["drift_pct"] < 0.0015
