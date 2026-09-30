from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN
from statistics import mean
from typing import Any, AsyncGenerator, Callable, Optional

import ccxt

from crypto_bot.core.leverage import futures_symbol, spot_symbol


class ExchangeError(Exception):
    pass


class OrderValidationError(Exception):
    pass


@dataclass
class ExchangeConfig:
    name: str
    api_key: str
    api_secret: str
    mode: str = "paper"  # Yeni: paper | live
    password: str | None = None
    enable_rate_limit: bool = True
    timeout_ms: int = 20000
    max_retries: int = 3
    retry_delay_seconds: float = 1.5
    market_type: str = "spot"  # spot | swap (perpetual futures)
    settle_currency: str = "USDT"


class ExchangeRulesAdapter:
    def __init__(self, exchange_name: str) -> None:
        self.exchange_name = exchange_name.lower()

    def parse(self, market: dict[str, Any]) -> dict[str, Any]:
        info = market.get("info", {}) or {}
        limits = market.get("limits", {}) or {}
        precision = market.get("precision", {}) or {}
        rules = {
            "min_qty": float((limits.get("amount", {}) or {}).get("min") or 0.0),
            "min_notional": float((limits.get("cost", {}) or {}).get("min") or 0.0),
            "step_size": 0.0,
            "tick_size": 0.0,
            "qty_precision": precision.get("amount"),
            "price_precision": precision.get("price"),
        }

        if self.exchange_name == "binance":
            parsed = self._parse_binance(info)
        elif self.exchange_name == "bybit":
            parsed = self._parse_bybit(info)
        elif self.exchange_name == "okx":
            parsed = self._parse_okx(info)
        else:
            parsed = {"step_size": float(info.get("stepSize") or 0.0), "tick_size": float(info.get("tickSize") or 0.0)}
        for key, value in parsed.items():
            if isinstance(value, (int, float)) and value <= 0 and isinstance(rules.get(key), (int, float)) and rules.get(key, 0) > 0:
                continue
            rules[key] = value
        return rules

    @staticmethod
    def _parse_binance(info: dict[str, Any]) -> dict[str, Any]:
        filters = info.get("filters", [])
        step_size = float(info.get("stepSize") or 0.0)
        tick_size = float(info.get("tickSize") or 0.0)
        min_qty = float(info.get("minQty") or 0.0)
        min_notional = float(info.get("minNotional") or 0.0)
        for flt in filters:
            ftype = flt.get("filterType")
            if ftype == "LOT_SIZE":
                if flt.get("stepSize"):
                    step_size = float(flt.get("stepSize"))
                if flt.get("minQty"):
                    min_qty = float(flt.get("minQty"))
            elif ftype == "MIN_NOTIONAL":
                if flt.get("minNotional"):
                    min_notional = float(flt.get("minNotional"))
            elif ftype == "PRICE_FILTER":
                if flt.get("tickSize"):
                    tick_size = float(flt.get("tickSize"))
        return {"step_size": step_size, "tick_size": tick_size, "min_qty": min_qty, "min_notional": min_notional}

    @staticmethod
    def _parse_bybit(info: dict[str, Any]) -> dict[str, Any]:
        lot_filter = info.get("lotSizeFilter", {}) or {}
        price_filter = info.get("priceFilter", {}) or {}
        return {
            "step_size": float(lot_filter.get("qtyStep") or 0.0),
            "min_qty": float(lot_filter.get("minOrderQty") or 0.0),
            "tick_size": float(price_filter.get("tickSize") or 0.0),
            "min_notional": float(lot_filter.get("minNotionalValue") or 0.0),
        }

    @staticmethod
    def _parse_okx(info: dict[str, Any]) -> dict[str, Any]:
        return {
            "step_size": float(info.get("lotSz") or 0.0),
            "tick_size": float(info.get("tickSz") or 0.0),
            "min_qty": float(info.get("minSz") or 0.0),
            "min_notional": float(info.get("minNotional") or 0.0),
        }


class ExchangeClient:
    def __init__(self, config: ExchangeConfig, logger: Any) -> None:
        self.config = config
        self.logger = logger
        exchange_cls = getattr(ccxt, config.name)
        self.client = exchange_cls(
            {
                "apiKey": config.api_key,
                "secret": config.api_secret,
                "password": config.password,
                "enableRateLimit": config.enable_rate_limit,
                "timeout": config.timeout_ms,
                "options": {"defaultType": config.market_type},
            }
        )
        self._markets_loaded = False
        self._market_cache: dict[str, dict[str, Any]] = {}
        self._rules_cache: dict[str, dict[str, Any]] = {}
        self.rules_adapter = ExchangeRulesAdapter(config.name)
        self.consecutive_failures = 0
        self.last_error: str = ""
        self.latency_ms_samples: list[float] = []

    @property
    def is_futures(self) -> bool:
        return self.config.market_type != "spot"

    def _sym(self, symbol: str) -> str:
        """Botun iç sembolünü ('BTC/USDT') borsa sembolüne çevirir."""
        return futures_symbol(symbol, self.config.settle_currency) if self.is_futures else symbol

    def configure_leverage(self, symbols: list[str], leverage: float, margin_mode: str) -> None:
        """Her sembol için borsada marj modunu ve kaldıracı ayarlar (yalnızca live)."""
        if not self.is_futures:
            return
        for symbol in symbols:
            ex_symbol = self._sym(symbol)
            try:
                self.client.set_margin_mode(margin_mode, ex_symbol)
            except Exception as exc:
                # Binance zaten aynı moddaysa "No need to change margin type" hatası verir
                if "no need to change" not in str(exc).lower():
                    raise ExchangeError(f"set_margin_mode failed for {ex_symbol}: {exc}") from exc
            try:
                self.client.set_leverage(int(leverage), ex_symbol)
            except Exception as exc:
                raise ExchangeError(f"set_leverage failed for {ex_symbol}: {exc}") from exc
            self.logger.info("Leverage configured: %s %sx %s", ex_symbol, int(leverage), margin_mode)

    def load_markets(self, force_reload: bool = False) -> dict[str, Any]:
        if not self._markets_loaded or force_reload:
            markets = self._retry(self.client.load_markets, force_reload)
            self._market_cache = markets
            self._rules_cache = {symbol: self.rules_adapter.parse(mkt) for symbol, mkt in markets.items()}
            self._markets_loaded = True
        return self._market_cache

    def get_market(self, symbol: str) -> dict[str, Any]:
        markets = self.load_markets()
        symbol = self._sym(symbol)
        if symbol not in markets:
            raise ExchangeError(f"Market metadata missing for symbol: {symbol}")
        return markets[symbol]

    def get_rules(self, symbol: str) -> dict[str, Any]:
        self.load_markets()
        symbol = self._sym(symbol)
        if symbol not in self._rules_cache:
            self._rules_cache[symbol] = self.rules_adapter.parse(self.get_market(symbol))
        return self._rules_cache[symbol]

    @staticmethod
    def _round_by_step(value: float, step: float) -> float:
        if step <= 0:
            return value
        return float((Decimal(str(value)) / Decimal(str(step))).to_integral_value(rounding=ROUND_DOWN) * Decimal(str(step)))

    @staticmethod
    def _round_by_precision(value: float, precision: int | None) -> float:
        if precision is None or precision < 0:
            return value
        factor = 10 ** precision
        return math.floor(value * factor) / factor

    def get_fee_rates(self, symbol: str) -> tuple[float, float]:
        market = self.get_market(symbol)
        maker = float(market.get("maker", 0.001))
        taker = float(market.get("taker", maker if maker > 0 else 0.001))
        return maker, taker

    def validate_order(self, symbol: str, price: float, amount: float) -> tuple[float, float]:
        rules = self.get_rules(symbol)
        rounded_amount = self._round_by_step(amount, float(rules.get("step_size", 0.0)))
        rounded_price = self._round_by_step(price, float(rules.get("tick_size", 0.0)))
        rounded_amount = self._round_by_precision(rounded_amount, rules.get("qty_precision"))
        rounded_price = self._round_by_precision(rounded_price, rules.get("price_precision"))

        min_qty = float(rules.get("min_qty", 0.0))
        min_notional = float(rules.get("min_notional", 0.0))
        if rounded_amount <= 0 or rounded_price <= 0:
            raise OrderValidationError("Rounded order values are non-positive")
        if min_qty > 0 and rounded_amount < min_qty:
            raise OrderValidationError(f"Amount {rounded_amount} below minQty {min_qty} for {symbol}")
        if min_notional > 0 and rounded_amount * rounded_price < min_notional:
            raise OrderValidationError(
                f"Notional {rounded_amount * rounded_price:.8f} below minNotional {min_notional} for {symbol}"
            )
        return rounded_price, rounded_amount

    def _retry(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        last_exc: Exception | None = None
        for attempt in range(1, self.config.max_retries + 1):
            started = time.monotonic()
            try:
                result = fn(*args, **kwargs)
                latency_ms = (time.monotonic() - started) * 1000
                self.latency_ms_samples.append(latency_ms)
                self.latency_ms_samples = self.latency_ms_samples[-200:]
                self.consecutive_failures = 0
                self.last_error = ""
                return result
            except (ccxt.NetworkError, ccxt.ExchangeError, ccxt.RequestTimeout) as exc:
                last_exc = exc
                self.consecutive_failures += 1
                self.last_error = str(exc)
                self.logger.warning("Exchange call failed (attempt %s/%s): %s", attempt, self.config.max_retries, exc)
                if attempt < self.config.max_retries:
                    time.sleep(self.config.retry_delay_seconds * attempt)
        raise ExchangeError(f"Exchange request failed after retries: {last_exc}")

    def should_pause_trading(self, max_consecutive_failures: int, timeout_spike_ms: float) -> bool:
        avg_latency = mean(self.latency_ms_samples[-20:]) if self.latency_ms_samples else 0.0
        return self.consecutive_failures >= max_consecutive_failures or avg_latency >= timeout_spike_ms

    def get_health_snapshot(self) -> dict[str, Any]:
        avg_latency = mean(self.latency_ms_samples[-20:]) if self.latency_ms_samples else 0.0
        return {
            "consecutive_failures": self.consecutive_failures,
            "avg_latency_ms": avg_latency,
            "last_error": self.last_error,
        }

    def get_balance(self) -> dict[str, Any]:
        if self.config.mode == "paper" and (not self.config.api_key or not self.config.api_secret):
            return {"total": {}}
        return self._retry(self.client.fetch_balance)

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 500) -> list[list[float]]:
        # Herkese açık veriler çekilebilir
        return self._retry(self.client.fetch_ohlcv, self._sym(symbol), timeframe=timeframe, limit=limit)

    def get_open_orders(self, symbol: Optional[str] = None) -> list[dict[str, Any]]:
        if self.config.mode == "paper" and (not self.config.api_key or not self.config.api_secret):
            return []
        return self._retry(self.client.fetch_open_orders, self._sym(symbol) if symbol else None)

    def fetch_positions(self, symbols: list[str]) -> dict[str, float]:
        if self.config.mode == "paper" and (not self.config.api_key or not self.config.api_secret):
            return {s: 0.0 for s in symbols}
            
        if hasattr(self.client, "fetch_positions"):
            try:
                rows = self._retry(self.client.fetch_positions, [self._sym(s) for s in symbols])
                out = {s: 0.0 for s in symbols}
                for row in rows:
                    sym = spot_symbol(str(row.get("symbol", "")))
                    contracts = abs(float(row.get("contracts") or row.get("positionAmt") or row.get("size") or 0))
                    # Short pozisyonlar negatif miktar olarak döner
                    out[sym] = -contracts if str(row.get("side", "")).lower() == "short" else contracts
                return out
            except Exception:
                pass

        balance = self.get_balance().get("total", {})
        out = {}
        for symbol in symbols:
            base = symbol.split("/")[0]
            out[symbol] = float(balance.get(base, 0.0))
        return out

    def fetch_server_time_ms(self) -> int | None:
        if hasattr(self.client, "fetch_time"):
            try:
                return int(self._retry(self.client.fetch_time))
            except Exception:
                return None
        return None

    def fetch_ticker(self, symbol: str) -> dict[str, Any]:
        return self._retry(self.client.fetch_ticker, self._sym(symbol))

    def fetch_order_book(self, symbol: str, limit: int = 50) -> dict[str, Any]:
        return self._retry(self.client.fetch_order_book, self._sym(symbol), limit)

    def fetch_order(self, order_id: str, symbol: str) -> dict[str, Any]:
        return self._retry(self.client.fetch_order, order_id, self._sym(symbol))

    def cancel_order(self, order_id: str, symbol: str) -> dict[str, Any]:
        return self._retry(self.client.cancel_order, order_id, self._sym(symbol))

    def create_market_order_with_params(
        self, symbol: str, side: str, amount: float, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        return self._retry(self.client.create_order, self._sym(symbol), "market", side, amount, None, params or {})

    def create_limit_order_with_params(
        self, symbol: str, side: str, amount: float, price: float, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        return self._retry(self.client.create_order, self._sym(symbol), "limit", side, amount, price, params or {})

    def create_limit_order(self, symbol: str, side: str, amount: float, price: float) -> dict[str, Any]:
        return self.create_limit_order_with_params(symbol, side, amount, price)

    def create_stop_loss_order(
        self, symbol: str, side: str, amount: float, stop_price: float, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Borsada koruyucu stop-loss emri oluşturur. ccxt'nin birleşik
        stop emri parametrelerini (stopLossPrice / triggerPrice) kullanır;
        borsalar arasında uyumluluk için ikisini de gönderir.
        """
        merged: dict[str, Any] = {"stopLossPrice": stop_price, "triggerPrice": stop_price, "reduceOnly": True}
        if params:
            merged.update(params)
        return self._retry(self.client.create_order, self._sym(symbol), "market", side, amount, None, merged)

    def estimate_fill_price(self, symbol: str, side: str, amount: float, limit: int = 50) -> float:
        order_book = self.fetch_order_book(symbol, limit=limit)
        levels = order_book.get("asks", []) if side.lower() == "buy" else order_book.get("bids", [])
        if not levels:
            ticker = self.fetch_ticker(symbol)
            return float(ticker.get("last") or ticker.get("close") or 0.0)

        remaining = amount
        cost = 0.0
        for level_price, level_qty in levels:
            take = min(remaining, float(level_qty))
            cost += take * float(level_price)
            remaining -= take
            if remaining <= 0:
                break
        if remaining > 0:
            cost += remaining * float(levels[-1][0])
        return cost / amount if amount > 0 else float(levels[0][0])

    async def stream_ohlcv(self, symbol: str, timeframe: str) -> AsyncGenerator[list[float], None]:
        ccxtpro = None
        try:
            import ccxt.pro as ccxtpro  # type: ignore
        except Exception:
            ccxtpro = None

        if ccxtpro and hasattr(ccxtpro, self.config.name):
            pro_cls = getattr(ccxtpro, self.config.name)
            ws = pro_cls(
                {
                    "apiKey": self.config.api_key,
                    "secret": self.config.api_secret,
                    "password": self.config.password,
                    "enableRateLimit": self.config.enable_rate_limit,
                    "options": {"defaultType": self.config.market_type},
                }
            )
            try:
                while True:
                    candles = await ws.watch_ohlcv(self._sym(symbol), timeframe)
                    if candles:
                        yield candles[-1]
            finally:
                await ws.close()
        else:
            self.logger.info("ccxt.pro unavailable, fallback to polling for %s", symbol)
            while True:
                candles = self.fetch_ohlcv(symbol, timeframe, limit=2)
                if candles:
                    yield candles[-1]
                await asyncio.sleep(1)

    async def stream_private_updates(self, symbols: list[str]) -> AsyncGenerator[dict[str, Any], None]:
        ccxtpro = None
        try:
            import ccxt.pro as ccxtpro  # type: ignore
        except Exception:
            ccxtpro = None

        if ccxtpro and hasattr(ccxtpro, self.config.name):
            pro_cls = getattr(ccxtpro, self.config.name)
            ws = pro_cls(
                {
                    "apiKey": self.config.api_key,
                    "secret": self.config.api_secret,
                    "password": self.config.password,
                    "enableRateLimit": self.config.enable_rate_limit,
                    "options": {"defaultType": self.config.market_type},
                }
            )
            try:
                while True:
                    if hasattr(ws, "watch_orders"):
                        orders = await ws.watch_orders()
                        for order in orders or []:
                            yield {"type": "order", "data": order}
                    if hasattr(ws, "watch_positions"):
                        positions = await ws.watch_positions([self._sym(s) for s in symbols])
                        for position in positions or []:
                            yield {"type": "position", "data": position}
                    await asyncio.sleep(0)
            finally:
                await ws.close()
        else:
            while True:
                for symbol in symbols:
                    for order in self.get_open_orders(symbol):
                        yield {"type": "order", "data": order}
                for symbol, qty in self.fetch_positions(symbols).items():
                    yield {"type": "position", "data": {"symbol": symbol, "contracts": qty}}
                await asyncio.sleep(2)
