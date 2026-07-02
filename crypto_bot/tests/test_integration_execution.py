from crypto_bot.core.execution import ExecutionConfig, ExecutionEngine
from crypto_bot.database.db import Database
from crypto_bot.database.models import Order


class DummyExchange:
    def get_fee_rates(self, _symbol):
        return 0.0002, 0.0006

    def validate_order(self, _symbol, price, amount):
        return price, amount

    def estimate_fill_price(self, _symbol, _side, _amount, limit=50):
        _ = limit
        return 50_000.0


class DummyLogger:
    def info(self, *_args, **_kwargs):
        return None

    def error(self, *_args, **_kwargs):
        return None

    def warning(self, *_args, **_kwargs):
        return None


def test_paper_execution_persists_order(tmp_path):
    db_path = tmp_path / "integration.db"
    db = Database(f"sqlite:///{db_path}")
    db.create_tables()
    engine = ExecutionEngine(
        exchange=DummyExchange(),
        db=db,
        logger=DummyLogger(),
        config=ExecutionConfig(
            mode="paper",
            order_type="market",
            max_retries=2,
            retry_delay_seconds=1,
            poll_interval_seconds=0.01,
            max_slippage_pct=0.001,
            slippage_model="dynamic",
            slippage_volatility_factor=2.0,
            slippage_volume_factor=50.0,
            use_order_book=True,
            order_book_depth=20,
            simulate_order_delay_seconds=0.0,
            partial_fill_min_ratio=0.8,
            order_fill_timeout_seconds=3,
            max_execution_drift_pct=0.01,
        ),
    )

    order = engine.place_order("BTC/USDT", "buy", 0.1, 50_000.0, market_context={"volatility": 0.01, "volume": 300})
    assert order["status"] == "FILLED"
    assert order["fee_paid"] > 0

    with db.session_scope() as session:
        saved = session.query(Order).all()
        assert len(saved) == 1
        assert saved[0].symbol == "BTC/USDT"
