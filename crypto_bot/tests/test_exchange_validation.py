from crypto_bot.core.exchange import ExchangeClient, ExchangeConfig, OrderValidationError


class DummyLogger:
    def warning(self, *_args, **_kwargs):
        return None


def test_validate_order_applies_precision_and_limits():
    client = ExchangeClient(
        ExchangeConfig(name="binance", api_key="", api_secret=""),
        logger=DummyLogger(),
    )
    client._markets_loaded = True
    client._market_cache = {
        "BTC/USDT": {
            "precision": {"price": 2, "amount": 4},
            "limits": {
                "amount": {"min": 0.001},
                "price": {"min": 1, "max": 1_000_000},
                "cost": {"min": 10},
            },
            "info": {"stepSize": "0.0001", "tickSize": "0.01"},
            "maker": 0.0002,
            "taker": 0.0006,
        }
    }

    price, amount = client.validate_order("BTC/USDT", price=65000.1234, amount=0.005678)
    assert price == 65000.12
    assert amount == 0.0056


def test_validate_order_rejects_small_notional():
    client = ExchangeClient(
        ExchangeConfig(name="binance", api_key="", api_secret=""),
        logger=DummyLogger(),
    )
    client._markets_loaded = True
    client._market_cache = {
        "BTC/USDT": {
            "precision": {"price": 2, "amount": 4},
            "limits": {"amount": {"min": 0.001}, "price": {"min": 1, "max": 1_000_000}, "cost": {"min": 10}},
            "info": {"stepSize": "0.0001", "tickSize": "0.01"},
            "maker": 0.0002,
            "taker": 0.0006,
        }
    }
    try:
        client.validate_order("BTC/USDT", price=100, amount=0.001)
        assert False, "Expected notional validation error"
    except OrderValidationError:
        assert True
