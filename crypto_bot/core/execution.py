from __future__ import annotations

import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from crypto_bot.core.exchange import OrderValidationError
from crypto_bot.database.models import Order


@dataclass
class ExecutionConfig:
    mode: str
    order_type: str
    max_retries: int
    retry_delay_seconds: float
    poll_interval_seconds: float
    max_slippage_pct: float
    slippage_model: str
    slippage_volatility_factor: float
    slippage_volume_factor: float
    use_order_book: bool
    order_book_depth: int
    simulate_order_delay_seconds: float
    partial_fill_min_ratio: float
    order_fill_timeout_seconds: float
    max_execution_drift_pct: float
    use_limit_for_tp: bool = True
    # Limit entry optimizasyonu
    use_post_only_entry: bool = False   # True → giriş emirleri post-only limit olarak gönderilir
    limit_entry_offset_pct: float = 0.0002  # Fiyatın ne kadar içinde limit koy (doldurulabilirlik için)
    limit_entry_timeout_seconds: float = 15.0  # Bu sürede dolmazsa market'a düş


class ExecutionEngine:
    def __init__(self, exchange: Any, db: Any, logger: Any, config: ExecutionConfig) -> None:
        self.exchange = exchange
        self.db = db
        self.logger = logger
        self.config = config

    @staticmethod
    def _now() -> datetime:
        return datetime.now(timezone.utc)

    @staticmethod
    def _normalize_status(status: str) -> str:
        s = (status or "").lower()
        if s in {"new", "open"}:
            return "NEW"
        if s in {"partially_filled", "partial", "partially-filled"}:
            return "PARTIALLY_FILLED"
        if s in {"closed", "filled"}:
            return "FILLED"
        if s in {"canceled", "cancelled"}:
            return "CANCELED"
        return "NEW"

    def _dynamic_slippage_pct(self, volatility: float, volume: float) -> float:
        if self.config.slippage_model != "dynamic":
            return self.config.max_slippage_pct
        volume_penalty = self.config.slippage_volume_factor / max(volume, 1.0)
        dynamic = self.config.max_slippage_pct * (
            1.0 + (volatility * self.config.slippage_volatility_factor) + volume_penalty
        )
        return max(self.config.max_slippage_pct * 0.2, min(dynamic, self.config.max_slippage_pct * 5))

    def _apply_slippage(self, side: str, reference_price: float, volatility: float, volume: float) -> float:
        slip = self._dynamic_slippage_pct(volatility, volume)
        if side.upper() == "BUY":
            return reference_price * (1 + slip)
        return reference_price * (1 - slip)

    def _estimate_reference_price(self, symbol: str, side: str, amount: float, fallback: float) -> float:
        if not self.config.use_order_book:
            return fallback
        try:
            return self.exchange.estimate_fill_price(symbol, side, amount, limit=self.config.order_book_depth)
        except Exception as exc:
            self.logger.warning("Order book estimate failed for %s: %s", symbol, exc)
            return fallback

    @staticmethod
    def _calculate_fee(notional: float, fee_rate: float) -> float:
        return max(0.0, notional * fee_rate)

    def _save_order(
        self,
        order: dict[str, Any],
        symbol: str,
        side: str,
        order_type: str,
        amount: float,
        price: float,
    ) -> None:
        with self.db.session_scope() as session:
            session.add(
                Order(
                    exchange_order_id=str(order.get("id", "paper-order")),
                    symbol=symbol,
                    side=side,
                    order_type=order_type,
                    amount=amount,
                    price=price,
                    filled=float(order.get("filled", amount)),
                    status=str(order.get("status", "NEW")),
                )
            )

    def place_managed_trade(
        self,
        symbol: str,
        side: str,
        amount: float,
        reference_price: float,
        tp_targets: list[dict[str, float]],
        stop_loss_price: float,
        market_context: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """
        Giriş emrini gönderir ve ardından kademeli kar al emirlerini dizer.
        """
        self.logger.info(f"Placing managed {side} trade for {symbol} at {reference_price}")
        
        # 1. Giriş Emri (Entry Order)
        entry_order = self.place_order(symbol, side, amount, reference_price, market_context)
        
        if entry_order["status"] != "FILLED":
            self.logger.warning(f"Entry order for {symbol} was not fully filled. Status: {entry_order['status']}")
            return entry_order

        # 2. Kar Al Emirleri (Exit Orders - Partial TP)
        exit_side = "sell" if side.lower() == "buy" else "buy"
        filled_qty = float(entry_order["filled"])
        tp_orders = []

        # 2a. Koruyucu Stop-Loss (live modda borsaya gerçek emir olarak gönderilir)
        stop_loss_order: dict[str, Any] | None = None
        if self.config.mode != "paper" and stop_loss_price and stop_loss_price > 0:
            try:
                v_sl_price, v_sl_amount = self.exchange.validate_order(symbol, stop_loss_price, filled_qty)
                stop_loss_order = self.exchange.create_stop_loss_order(
                    symbol, exit_side, v_sl_amount, v_sl_price
                )
                self.logger.info(
                    "Protective stop-loss placed for %s: %s %s @ %s",
                    symbol, exit_side, v_sl_amount, v_sl_price,
                )
            except Exception as exc:
                # Stop emri kurulamadıysa pozisyonu korumasız bırakma — girişi geri al.
                self.logger.error("Stop-loss placement failed for %s: %s. Closing entry.", symbol, exc)
                try:
                    self.place_order(symbol, exit_side, filled_qty, reference_price, market_context)
                except Exception as close_exc:
                    self.logger.critical(
                        "Failed to unwind unprotected position for %s: %s", symbol, close_exc
                    )
                entry_order["status"] = "ABORTED_NO_STOP"
                entry_order["exit_plans"] = {"tp_orders": [], "stop_loss": stop_loss_price, "stop_loss_order": None}
                return entry_order

        for i, target in enumerate(tp_targets):
            target_price = target["price"]
            # Hedef miktarı hesapla (örneğin %50'si)
            target_qty = filled_qty * target["close_pct"]
            
            # Son parça için miktar düzeltmesi (yuvarlama hatalarını önlemek için)
            if i == len(tp_targets) - 1:
                remaining_qty = filled_qty - sum(float(o["amount"]) for o in tp_orders)
                target_qty = max(0.0, remaining_qty)

            if target_qty <= 0:
                continue

            self.logger.info(f"Placing TP{i+1} limit {exit_side} order at {target_price} for {target_qty}")
            
            try:
                # Limit emri olarak gönderiyoruz
                tp_order = self._place_single_exit_order(symbol, exit_side, target_qty, target_price)
                tp_orders.append(tp_order)
            except Exception as exc:
                self.logger.error(f"Failed to place TP order {i+1}: {exc}")

        entry_order["exit_plans"] = {
            "tp_orders": tp_orders,
            "stop_loss": stop_loss_price,
            "stop_loss_order": stop_loss_order,
        }

        return entry_order

    def _place_single_exit_order(self, symbol: str, side: str, amount: float, price: float) -> dict[str, Any]:
        """Basit bir limit çıkış emri gönderir (beklemeden)."""
        if self.config.mode == "paper":
            order = {
                "id": f"paper-tp-{int(time.time() * 1000)}",
                "symbol": symbol,
                "side": side,
                "type": "limit",
                "amount": amount,
                "price": price,
                "status": "NEW"
            }
            self._save_order(order, symbol, side, "limit", amount, price)
            return order

        # Real exchange logic
        v_price, v_amount = self.exchange.validate_order(symbol, price, amount)
        return self.exchange.create_limit_order(symbol, side, v_amount, v_price)

    def place_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        reference_price: float,
        market_context: dict[str, float] | None = None,
        signal_timestamp: datetime | None = None,
        client_order_id: str | None = None,
    ) -> dict[str, Any]:
        side = side.lower()
        order_type = self.config.order_type.lower()
        context = market_context or {}
        volatility = float(context.get("volatility", 0.0))
        volume = float(context.get("volume", 1.0))
        signal_ts = signal_timestamp or self._now()

        book_ref_price = self._estimate_reference_price(symbol, side, amount, reference_price)
        slippage_price = self._apply_slippage(side, book_ref_price, volatility, volume)
        maker_fee, taker_fee = self.exchange.get_fee_rates(symbol)
        fee_rate = maker_fee if order_type == "limit" else taker_fee

        try:
            validated_price, validated_amount = self.exchange.validate_order(symbol, slippage_price, amount)
        except OrderValidationError:
            raise
        except Exception as exc:
            raise RuntimeError(f"Order validation failed for {symbol}: {exc}") from exc

        drift_pct = abs(validated_price - reference_price) / max(reference_price, 1e-9)
        if drift_pct > self.config.max_execution_drift_pct:
            raise RuntimeError(
                f"Execution drift too high ({drift_pct:.4%} > {self.config.max_execution_drift_pct:.4%}), aborting trade"
            )

        if self.config.mode == "paper":
            if self.config.simulate_order_delay_seconds > 0:
                time.sleep(self.config.simulate_order_delay_seconds)

            # Post-only limit simülasyonu: maker fee ve hafif fiyat iyileştirmesi
            if self.config.use_post_only_entry and order_type != "limit":
                limit_price = self._post_only_limit_price(side, validated_price)
                # %70 ihtimalle limit dolar (kalan %30'da market fallback simüle edilir)
                if random.random() < 0.70:
                    fill_price_sim = limit_price
                    fee_rate = maker_fee  # post-only → maker
                else:
                    fill_price_sim = validated_price  # market gibi doldu
                    # fee_rate taker olarak kalır
                validated_price = fill_price_sim

            fill_ratio = max(self.config.partial_fill_min_ratio, random.uniform(self.config.partial_fill_min_ratio, 1.0))
            filled = validated_amount * fill_ratio
            lifecycle = ["NEW", "PARTIALLY_FILLED"] if fill_ratio < 0.98 else ["NEW"]
            if fill_ratio < 0.98:
                time.sleep(self.config.poll_interval_seconds)
                filled = validated_amount
            lifecycle.append("FILLED")

            notional = filled * validated_price
            fee_paid = self._calculate_fee(notional, fee_rate)
            execution_ts = self._now()
            order = {
                "id": f"paper-{int(time.time() * 1000)}",
                "clientOrderId": client_order_id,
                "symbol": symbol,
                "side": side,
                "type": order_type,
                "amount": validated_amount,
                "filled": filled,
                "price": validated_price,
                "average": validated_price,
                "status": "FILLED",
                "fee_rate": fee_rate,
                "fee_paid": fee_paid,
                "notional": notional,
                "signal_timestamp": signal_ts.isoformat(),
                "execution_timestamp": execution_ts.isoformat(),
                "execution_delay_ms": (execution_ts - signal_ts).total_seconds() * 1000,
                "lifecycle": lifecycle,
                "execution_details": {
                    "volatility": volatility,
                    "volume": volume,
                    "expected_price": reference_price,
                    "actual_price": validated_price,
                    "drift_pct": drift_pct,
                },
            }
            self._save_order(order, symbol, side, order_type, validated_amount, validated_price)
            return order

        last_error: Exception | None = None
        for attempt in range(1, self.config.max_retries + 1):
            try:
                # Post-only limit entry: maker fee'ye hak kazanmak için limit olarak dene,
                # timeout içinde dolmazsa market'a düş.
                if self.config.use_post_only_entry and order_type != "limit":
                    limit_price = self._post_only_limit_price(side, validated_price)
                    v_limit_price, v_limit_amount = self.exchange.validate_order(symbol, limit_price, validated_amount)
                    limit_params: dict[str, Any] = {"postOnly": True}
                    if client_order_id:
                        limit_params["clientOrderId"] = client_order_id + "-po"
                    try:
                        created = self.exchange.create_limit_order_with_params(
                            symbol, side, v_limit_amount, v_limit_price, params=limit_params
                        )
                        tracked = self._wait_fill_with_timeout(
                            created["id"], symbol,
                            timeout=self.config.limit_entry_timeout_seconds,
                        )
                        if tracked.get("status") == "FILLED":
                            # Maker fee → ucuz doldu
                            fill_price = float(tracked.get("average") or tracked.get("price") or v_limit_price)
                            fill_qty = float(tracked.get("filled") or 0.0)
                            notional = fill_qty * fill_price
                            actual_fee_rate = maker_fee  # post-only garantili maker
                            fee_paid = self._calculate_fee(notional, actual_fee_rate)
                            execution_ts = self._now()
                            tracked["fee_rate"] = actual_fee_rate
                            tracked["fee_paid"] = fee_paid
                            tracked["notional"] = notional
                            tracked["signal_timestamp"] = signal_ts.isoformat()
                            tracked["execution_timestamp"] = execution_ts.isoformat()
                            tracked["execution_delay_ms"] = (execution_ts - signal_ts).total_seconds() * 1000
                            tracked["order_subtype"] = "post_only_limit"
                            tracked["execution_details"] = {
                                "volatility": volatility,
                                "volume": volume,
                                "expected_price": reference_price,
                                "actual_price": fill_price,
                                "drift_pct": abs(fill_price - reference_price) / max(reference_price, 1e-9),
                            }
                            self._save_order(tracked, symbol, side, "limit", v_limit_amount, fill_price)
                            return tracked
                        # Limit dolmadı → aşağıda market fallback devam eder
                        self.logger.info(
                            "Post-only limit timed out for %s, falling back to market", symbol
                        )
                    except Exception as po_exc:
                        self.logger.warning("Post-only limit failed for %s: %s, using market", symbol, po_exc)

                # Standart market veya limit
                if order_type == "limit":
                    params = {"clientOrderId": client_order_id} if client_order_id else {}
                    created = self.exchange.create_limit_order_with_params(
                        symbol,
                        side,
                        validated_amount,
                        validated_price,
                        params=params,
                    )
                else:
                    params = {"clientOrderId": client_order_id} if client_order_id else {}
                    created = self.exchange.create_market_order_with_params(
                        symbol,
                        side,
                        validated_amount,
                        params=params,
                    )

                tracked = self._wait_fill(created["id"], symbol)
                fill_price = float(tracked.get("average") or tracked.get("price") or validated_price)
                fill_qty = float(tracked.get("filled") or 0.0)
                notional = fill_qty * fill_price
                fee_paid = self._calculate_fee(notional, fee_rate)
                execution_ts = self._now()
                tracked["fee_rate"] = fee_rate
                tracked["fee_paid"] = fee_paid
                tracked["notional"] = notional
                tracked["signal_timestamp"] = signal_ts.isoformat()
                tracked["execution_timestamp"] = execution_ts.isoformat()
                tracked["execution_delay_ms"] = (execution_ts - signal_ts).total_seconds() * 1000
                tracked["execution_details"] = {
                    "volatility": volatility,
                    "volume": volume,
                    "expected_price": reference_price,
                    "actual_price": fill_price,
                    "drift_pct": abs(fill_price - reference_price) / max(reference_price, 1e-9),
                }
                self._save_order(tracked, symbol, side, order_type, validated_amount, fill_price)
                return tracked
            except Exception as exc:
                last_error = exc
                self.logger.error("Execution attempt %s failed: %s", attempt, exc)
                time.sleep(self.config.retry_delay_seconds * attempt)

        raise RuntimeError(f"Order execution failed: {last_error}")

    def _post_only_limit_price(self, side: str, reference_price: float) -> float:
        """
        Post-only limit emri için agresif ama maker garantili fiyat hesaplar.
        BUY → bid'e yakın (referans * (1 - offset)) → satıcıları bekle
        SELL → ask'a yakın (referans * (1 + offset)) → alıcıları bekle
        """
        offset = self.config.limit_entry_offset_pct
        if side.lower() == "buy":
            return reference_price * (1 - offset)
        return reference_price * (1 + offset)

    def _wait_fill_with_timeout(self, order_id: str, symbol: str, timeout: float) -> dict[str, Any]:
        """
        Belirtilen timeout süresi boyunca emrin dolmasını bekler.
        Dolmazsa emri iptal eder ve mevcut durumu döner.
        """
        deadline = time.monotonic() + timeout
        lifecycle = []
        latest: dict[str, Any] = {"id": order_id, "status": "NEW", "filled": 0.0}
        while True:
            order = self.exchange.fetch_order(order_id, symbol)
            normalized = self._normalize_status(str(order.get("status", "")))
            if normalized not in lifecycle:
                lifecycle.append(normalized)
            latest = order
            latest["status"] = normalized
            latest["lifecycle"] = lifecycle.copy()
            if normalized in {"FILLED", "CANCELED"}:
                return latest
            if time.monotonic() >= deadline:
                try:
                    self.exchange.cancel_order(order_id, symbol)
                    latest["status"] = "CANCELED"
                    lifecycle.append("CANCELED")
                    latest["lifecycle"] = lifecycle
                except Exception:
                    pass
                return latest
            time.sleep(self.config.poll_interval_seconds)

    def _wait_fill(self, order_id: str, symbol: str) -> dict[str, Any]:
        deadline = time.monotonic() + self.config.order_fill_timeout_seconds
        lifecycle = []
        latest: dict[str, Any] = {"id": order_id, "status": "NEW", "filled": 0.0}
        while True:
            order = self.exchange.fetch_order(order_id, symbol)
            normalized = self._normalize_status(str(order.get("status", "")))
            if normalized not in lifecycle:
                lifecycle.append(normalized)
            latest = order
            latest["status"] = normalized
            latest["lifecycle"] = lifecycle.copy()

            if normalized in {"FILLED", "CANCELED"}:
                return latest
            if time.monotonic() >= deadline:
                try:
                    self.exchange.cancel_order(order_id, symbol)
                    latest["status"] = "CANCELED"
                    lifecycle.append("CANCELED")
                    latest["lifecycle"] = lifecycle
                except Exception:
                    pass
                return latest
            time.sleep(self.config.poll_interval_seconds)
