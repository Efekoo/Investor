from crypto_bot.core.portfolio import Portfolio, Position
from crypto_bot.core.risk import RiskConfig, RiskManager


def build_risk_manager() -> RiskManager:
    return RiskManager(
        RiskConfig(
            risk_per_trade=0.01,
            stop_loss_pct=0.01,
            take_profit_pct=0.02,
            max_open_trades=3,
            max_daily_loss_pct=0.05,
            trailing_stop_pct=0.008,
            break_even_trigger_pct=0.01,
            max_risk_per_symbol_pct=0.25,
            volatility_position_scale=10.0,
            max_trade_size_quote=2500,
            max_consecutive_losses=3,
            volatility_spike_threshold=0.03,
            slippage_breaker_threshold=0.01,
        ),
        logger=None,
    )


def test_position_sizing_positive():
    rm = build_risk_manager()
    size = rm.calculate_position_size(balance=10_000, entry_price=100, stop_price=99, volatility=0.01)
    assert size > 0


def test_trade_validation_respects_open_positions():
    rm = build_risk_manager()
    portfolio = Portfolio()
    assert rm.validate_trade(portfolio, "BTC/USDT", 1000, 1000, 100)[0]


def test_trailing_stop_moves_up_for_long():
    rm = build_risk_manager()
    pos = Position(
        symbol="BTC/USDT",
        side="BUY",
        entry_price=100,
        qty=1,
        entry_fee=0.0,
        stop_loss=95,
        take_profit=120,
        peak_price=100,
    )
    stop, _ = rm.update_protective_levels(pos, 110)
    assert stop >= 100
