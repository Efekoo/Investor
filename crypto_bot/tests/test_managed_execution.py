import pytest
from unittest.mock import MagicMock
from crypto_bot.core.execution import ExecutionEngine, ExecutionConfig

class MockDB:
    def session_scope(self):
        return MagicMock()

@pytest.fixture
def execution_engine():
    exchange = MagicMock()
    # Mock fee rates
    exchange.get_fee_rates.return_value = (0.001, 0.001)
    # Mock validation
    exchange.validate_order.side_effect = lambda s, p, a: (p, a)
    # Paper mode ayarları
    config = ExecutionConfig(
        mode="paper",
        order_type="market",
        max_retries=3,
        retry_delay_seconds=0.1,
        poll_interval_seconds=0.1,
        max_slippage_pct=0.01,
        slippage_model="static",
        slippage_volatility_factor=1.0,
        slippage_volume_factor=1.0,
        use_order_book=False,
        order_book_depth=10,
        simulate_order_delay_seconds=0.0,
        partial_fill_min_ratio=1.0, # Full fill for simplicity
        order_fill_timeout_seconds=5.0,
        max_execution_drift_pct=0.1
    )
    return ExecutionEngine(exchange, MockDB(), MagicMock(), config)

def test_place_managed_trade_creates_tp_orders(execution_engine):
    symbol = "BTC/USDT"
    side = "buy"
    amount = 1.0
    reference_price = 50000.0
    tp_targets = [
        {"price": 51000.0, "close_pct": 0.5},
        {"price": 52000.0, "close_pct": 0.5}
    ]
    stop_loss = 49000.0
    
    result = execution_engine.place_managed_trade(
        symbol, side, amount, reference_price, tp_targets, stop_loss
    )
    
    assert result["status"] == "FILLED"
    assert "exit_plans" in result
    assert len(result["exit_plans"]["tp_orders"]) == 2
    
    tp1 = result["exit_plans"]["tp_orders"][0]
    tp2 = result["exit_plans"]["tp_orders"][1]
    
    assert tp1["price"] == 51000.0
    assert tp1["amount"] == 0.5
    assert tp2["price"] == 52000.0
    assert tp2["amount"] == 0.5
