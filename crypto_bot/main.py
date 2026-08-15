from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from crypto_bot.backtest.engine import BacktestEngine
from crypto_bot.core.data import DataManager
from crypto_bot.core.performance import PerformanceConfig, StrategyPerformanceTracker
from crypto_bot.core.regime import RegimeDetector
from crypto_bot.core.exchange import ExchangeClient, ExchangeConfig
from crypto_bot.core.execution import ExecutionConfig, ExecutionEngine
from crypto_bot.core.portfolio import Portfolio, Position
from crypto_bot.core.risk import RiskConfig, RiskManager
from crypto_bot.core.sentiment import SentimentAnalyzer
from crypto_bot.database.db import Database
from crypto_bot.database.models import BalanceHistory, LogEntry, Trade
from crypto_bot.strategies.bollinger_mean_reversion_strategy import BollingerMeanReversionStrategy
from crypto_bot.strategies.breakout_strategy import VolatilityBreakoutStrategy
from crypto_bot.strategies.ema_rsi_volume_strategy import EmaRsiVolumeStrategy
from crypto_bot.strategies.ema_strategy import EMACrossoverStrategy
from crypto_bot.strategies.grid_strategy import GridStrategy
from crypto_bot.strategies.rsi_strategy import RSIStrategy
from crypto_bot.strategies.supertrend_strategy import SupertrendStrategy
from crypto_bot.strategies.vwap_strategy import VWAPReversionStrategy
from crypto_bot.strategies.base_strategy import Strategy, StrategyDecision
from crypto_bot.backtest.filters import bear_regime_filter, bull_regime_filter
from crypto_bot.core.funding_rate import FundingRateAnalyzer
from crypto_bot.utils.logger import get_logger, log_event
from crypto_bot.utils.notifier import TelegramNotifier


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def load_settings(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_strategies(settings: dict[str, Any]) -> dict[str, Strategy]:
    built: dict[str, Strategy] = {}
    params = settings["strategy"]["params"]
    enabled = settings["trading"].get("enable_strategies", list(params.keys()))
    for name in enabled:
        lname = name.lower()
        p = params[lname]
        if lname == "ema":
            built[lname] = EMACrossoverStrategy(p)
        elif lname == "rsi":
            built[lname] = RSIStrategy(p)
        elif lname == "ema_rsi_volume":
            built[lname] = EmaRsiVolumeStrategy(p)
        elif lname == "bollinger_mean_reversion":
            built[lname] = BollingerMeanReversionStrategy(p)
        elif lname == "bollinger":
            built[lname] = BollingerMeanReversionStrategy(p)
        elif lname == "volatility_breakout":
            built[lname] = VolatilityBreakoutStrategy(p)
        elif lname == "grid":
            built[lname] = GridStrategy(p)
        elif lname == "vwap_reversion":
            built[lname] = VWAPReversionStrategy(p)
        elif lname == "supertrend":
            built[lname] = SupertrendStrategy(p)
    if not built:
        raise ValueError("No enabled strategies configured")
    return built


def timeframe_to_seconds(timeframe: str) -> int:
    value = int(timeframe[:-1])
    unit = timeframe[-1]
    if unit == "m":
        return value * 60
    if unit == "h":
        return value * 3600
    if unit == "d":
        return value * 86400
    raise ValueError(f"Unsupported timeframe: {timeframe}")


def next_run_sleep(seconds: int) -> float:
    now = time.time()
    return seconds - (now % seconds)


class TradingBot:
    def __init__(self, settings: dict[str, Any], settings_path: str) -> None:
        self.settings = settings
        self.settings_path = settings_path
        self.mode = settings["app"]["mode"].lower()
        self._settings_mtime = Path(settings_path).stat().st_mtime
        self._last_config_check = 0.0
        self._last_alerted_health = 0.0
        self._stop_requested = False

        runtime_dir = Path(settings["app"].get("data_dir", "runtime"))
        runtime_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = Path(settings["runtime"]["state_file"])
        self.state_file.parent.mkdir(parents=True, exist_ok=True)

        self.logger = get_logger(
            "crypto_bot",
            level=settings["app"].get("log_level", "INFO"),
            as_json=bool(settings["app"].get("log_json", True)),
            log_dir=settings["app"].get("log_dir", "runtime/logs"),
        )
        db_url = os.getenv("DATABASE_URL", settings["database"]["url"])
        self.db = Database(db_url)
        self.db.create_tables()
        self.notifier = TelegramNotifier(
            enabled=bool(settings["notifications"].get("telegram_enabled", False)),
            token=os.getenv("TELEGRAM_BOT_TOKEN"),
            chat_id=os.getenv("TELEGRAM_CHAT_ID"),
        )

        exchange_cfg = ExchangeConfig(
            name=settings["app"]["exchange"],
            api_key=os.getenv("EXCHANGE_API_KEY", ""),
            api_secret=os.getenv("EXCHANGE_SECRET", ""),
            password=os.getenv("EXCHANGE_PASSWORD", None),
            mode=self.mode,
            max_retries=settings["execution"]["max_retries"],
            retry_delay_seconds=settings["execution"]["retry_delay_seconds"],
        )
        self.exchange = ExchangeClient(exchange_cfg, self.logger)
        self.exchange.load_markets()
        self.data = DataManager(self.exchange, self.logger)
        self.portfolio = Portfolio()
        self.strategies = build_strategies(settings)
        self.primary_strategy_name = settings["strategy"]["name"].lower()
        self.regime_detector = self._build_regime_detector(settings)
        self.performance_tracker = self._build_performance_tracker(settings)
        self.risk = self._build_risk_manager(settings)
        self.execution = self._build_execution_engine(settings)
        self.sentiment = SentimentAnalyzer(settings.get("sentiment", {"enabled": True}), self.logger)
        self.funding_rate = FundingRateAnalyzer(settings.get("funding_rate", {"enabled": True}), self.logger)

        self.processed_signal_at: dict[str, datetime] = {}
        self.last_trade_at: dict[str, datetime] = {}
        self._last_prices: dict[str, float] = {}
        self.paper_cash = float(settings["app"].get("initial_paper_balance", 10_000.0))
        self._open_orders_snapshot: dict[str, dict[str, Any]] = {}
        self._submitted_order_keys: set[str] = set()
        self._private_stream_task: asyncio.Task | None = None
        self._last_private_event_at: datetime | None = None

        self._load_state()
        self._recover_open_trades_from_db()
        self._require_live_double_confirmation()
        if self.mode == "live":
            self._sync_positions_with_exchange()

    def _build_risk_manager(self, settings: dict[str, Any]) -> RiskManager:
        cfg = settings["risk"]
        return RiskManager(
            RiskConfig(
                risk_per_trade=cfg["risk_per_trade"],
                stop_loss_pct=cfg["stop_loss_pct"],
                take_profit_pct=cfg["take_profit_pct"],
                max_open_trades=settings["trading"]["max_open_trades"],
                max_daily_loss_pct=cfg["max_daily_loss_pct"],
                max_weekly_loss_pct=float(cfg.get("max_weekly_loss_pct", 0.15)),
                trailing_stop_pct=cfg["trailing_stop_pct"],
                break_even_trigger_pct=cfg["break_even_trigger_pct"],
                max_risk_per_symbol_pct=cfg["max_risk_per_symbol_pct"],
                volatility_position_scale=cfg["volatility_position_scale"],
                max_trade_size_quote=cfg["max_trade_size_quote"],
                max_consecutive_losses=cfg["max_consecutive_losses"],
                volatility_spike_threshold=cfg["volatility_spike_threshold"],
                slippage_breaker_threshold=cfg["slippage_breaker_threshold"],
                correlation_penalty=float(cfg.get("correlation_penalty", 0.5)),
                use_kelly=bool(cfg.get("use_kelly", False)),
                kelly_fraction=float(cfg.get("kelly_fraction", 0.5)),
            ),
            self.logger,
        )

    def _build_regime_detector(self, settings: dict[str, Any]) -> RegimeDetector:
        cfg = settings.get("market_intelligence", {})
        return RegimeDetector(
            atr_period=int(cfg.get("atr_period", 14)),
            adx_period=int(cfg.get("adx_period", 14)),
            trend_adx_threshold=float(cfg.get("trend_adx_threshold", 25)),
            high_volatility_threshold=float(cfg.get("high_volatility_threshold", 0.02)),
            low_volatility_threshold=float(cfg.get("low_volatility_threshold", 0.005)),
        )

    def _build_performance_tracker(self, settings: dict[str, Any]) -> StrategyPerformanceTracker:
        cfg = settings.get("performance", {})
        return StrategyPerformanceTracker(
            PerformanceConfig(
                min_trades_for_eval=int(cfg.get("min_trades_for_eval", 10)),
                min_win_rate=float(cfg.get("min_win_rate", 0.4)),
                max_drawdown=float(cfg.get("max_drawdown", 0.2)),
                min_pnl=float(cfg.get("min_pnl", -10.0)),
                drawdown_risk_multiplier=float(cfg.get("drawdown_risk_multiplier", 0.7)),
                high_volatility_risk_multiplier=float(cfg.get("high_volatility_risk_multiplier", 0.6)),
                stable_profit_risk_multiplier=float(cfg.get("stable_profit_risk_multiplier", 1.05)),
            )
        )

    def _build_execution_engine(self, settings: dict[str, Any]) -> ExecutionEngine:
        cfg = settings["execution"]
        return ExecutionEngine(
            self.exchange,
            self.db,
            self.logger,
            ExecutionConfig(
                mode=self.mode,
                order_type=cfg["order_type"],
                max_retries=cfg["max_retries"],
                retry_delay_seconds=cfg["retry_delay_seconds"],
                poll_interval_seconds=cfg["poll_interval_seconds"],
                max_slippage_pct=settings["risk"]["max_slippage_pct"],
                slippage_model=cfg.get("slippage_model", "static"),
                slippage_volatility_factor=float(cfg.get("slippage_volatility_factor", 1.0)),
                slippage_volume_factor=float(cfg.get("slippage_volume_factor", 0.0)),
                use_order_book=bool(cfg.get("use_order_book", False)),
                order_book_depth=int(cfg.get("order_book_depth", 50)),
                simulate_order_delay_seconds=float(cfg.get("simulate_order_delay_seconds", 0.0)),
                partial_fill_min_ratio=float(cfg.get("partial_fill_min_ratio", 1.0)),
                order_fill_timeout_seconds=float(cfg.get("order_fill_timeout_seconds", 20)),
                max_execution_drift_pct=float(cfg.get("max_execution_drift_pct", 0.003)),
                use_post_only_entry=bool(cfg.get("use_post_only_entry", False)),
                limit_entry_offset_pct=float(cfg.get("limit_entry_offset_pct", 0.0002)),
                limit_entry_timeout_seconds=float(cfg.get("limit_entry_timeout_seconds", 15.0)),
            ),
        )

    def _require_live_double_confirmation(self) -> None:
        safety = self.settings.get("safety", {})
        if self.mode != "live":
            return
        env1 = safety.get("live_confirmation_env", "LIVE_TRADING_CONFIRM")
        val1 = str(safety.get("live_confirmation_value", "YES"))
        env2 = safety.get("live_runtime_confirmation_env", "LIVE_RUNTIME_CONFIRM")
        val2 = str(safety.get("live_runtime_confirmation_value", "CONFIRMED"))
        if os.getenv(env1, "") != val1 or os.getenv(env2, "") != val2:
            raise RuntimeError(f"Live mode blocked. Set {env1}={val1} and {env2}={val2}.")
        log_event(self.logger, "WARNING", "live_mode_active", "LIVE MODE ACTIVE WITH REAL FUNDS")

    def _trade_id(self) -> str:
        return uuid.uuid4().hex[:16]

    def _trace(self, trade_id: str, stage: str, **fields: Any) -> None:
        log_event(self.logger, "INFO", "trade_trace", f"{trade_id}::{stage}", trade_id=trade_id, stage=stage, **fields)

    def _load_state(self) -> None:
        if not self.state_file.exists():
            return
        try:
            payload = json.loads(self.state_file.read_text(encoding="utf-8"))
            self.paper_cash = float(payload.get("paper_cash", self.paper_cash))
            self.portfolio.restore(payload.get("portfolio", {}))
            self.processed_signal_at = {
                k: datetime.fromisoformat(v) for k, v in payload.get("processed_signal_at", {}).items()
            }
            self.last_trade_at = {k: datetime.fromisoformat(v) for k, v in payload.get("last_trade_at", {}).items()}
            self._last_prices = {k: float(v) for k, v in payload.get("last_prices", {}).items()}
            self._open_orders_snapshot = payload.get("open_orders", {})
            self.performance_tracker.restore(payload.get("strategy_performance", {}))
            self._submitted_order_keys = set(payload.get("submitted_order_keys", []))
        except Exception as exc:
            self.logger.warning("Failed to restore state: %s", exc)

    def _save_state(self) -> None:
        payload = {
            "paper_cash": self.paper_cash,
            "portfolio": self.portfolio.snapshot(),
            "processed_signal_at": {k: v.isoformat() for k, v in self.processed_signal_at.items()},
            "last_trade_at": {k: v.isoformat() for k, v in self.last_trade_at.items()},
            "last_prices": self._last_prices,
            "open_orders": self._open_orders_snapshot,
            "strategy_performance": self.performance_tracker.snapshot(),
            "submitted_order_keys": list(self._submitted_order_keys)[-5000:],
            "updated_at": utcnow().isoformat(),
        }
        self.state_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _safe_shutdown(self, reason: str) -> None:
        self.logger.error("Safe shutdown: %s", reason)
        try:
            self._refresh_exchange_open_orders()
            self._save_state()
        finally:
            self._stop_requested = True
            if self._private_stream_task:
                self._private_stream_task.cancel()

    def _recover_open_trades_from_db(self) -> None:
        if self.portfolio.open_count() > 0:
            return
        with self.db.session_scope() as session:
            for row in session.query(Trade).filter(Trade.status == "open").all():
                stop, take = self.risk.get_stop_take_prices(str(row.side or "BUY"), float(row.entry_price))
                self.portfolio.open_position(
                    Position(
                        symbol=row.symbol,
                        side=row.side,
                        entry_price=float(row.entry_price),
                        qty=float(row.qty),
                        entry_fee=max(0.0, -float(row.pnl or 0.0)),
                        stop_loss=stop,
                        take_profit=take,
                        peak_price=float(row.entry_price),
                        strategy_name="recovered",
                        opened_at=row.opened_at,
                    )
                )

    def _sync_positions_with_exchange(self) -> None:
        positions = self.exchange.fetch_positions(self.settings["trading"]["symbols"])
        for symbol, qty in positions.items():
            if qty <= 0:
                continue
            if self.portfolio.is_open(symbol):
                continue
            ticker = self.exchange.fetch_ticker(symbol)
            entry = float(ticker.get("last") or ticker.get("close") or 0.0)
            if entry <= 0:
                continue
            stop, take = self.risk.get_stop_take_prices("BUY", entry)
            self.portfolio.open_position(
                Position(
                    symbol=symbol,
                    side="BUY",
                    entry_price=entry,
                    qty=qty,
                    entry_fee=0.0,
                    stop_loss=stop,
                    take_profit=take,
                    peak_price=entry,
                    strategy_name="synced",
                )
            )
            log_event(self.logger, "WARNING", "position_sync", "External position synced", symbol=symbol, qty=qty)

    def _refresh_exchange_open_orders(self) -> None:
        snapshot: dict[str, dict[str, Any]] = {}
        for symbol in self.settings["trading"]["symbols"]:
            try:
                for order in self.exchange.get_open_orders(symbol):
                    oid = str(order.get("id"))
                    snapshot[oid] = order
            except Exception:
                continue
        self._open_orders_snapshot = snapshot

    def _reconcile_state(self) -> None:
        # Paper modda exchange'e authenticated istek atma — pozisyonlar dahili olarak izlenir
        if self.mode == "paper":
            return
        self._refresh_exchange_open_orders()
        ext_positions = self.exchange.fetch_positions(self.settings["trading"]["symbols"])
        for symbol in self.settings["trading"]["symbols"]:
            ext_qty = float(ext_positions.get(symbol, 0.0))
            int_pos = self.portfolio.positions.get(symbol)
            if int_pos and ext_qty == 0 and not self._has_open_order_for_symbol(symbol):
                log_event(self.logger, "WARNING", "desync_fix", "Removing ghost internal position", symbol=symbol)
                self.portfolio.positions.pop(symbol, None)
            elif (not int_pos) and ext_qty > 0:
                ticker = self.exchange.fetch_ticker(symbol)
                price = float(ticker.get("last") or ticker.get("close") or 0.0)
                stop, take = self.risk.get_stop_take_prices("BUY", price)
                self.portfolio.open_position(
                    Position(
                        symbol=symbol,
                        side="BUY",
                        entry_price=price,
                        qty=ext_qty,
                        entry_fee=0.0,
                        stop_loss=stop,
                        take_profit=take,
                        peak_price=price,
                        strategy_name="synced",
                    )
                )
                log_event(self.logger, "WARNING", "desync_fix", "Added missing external position", symbol=symbol, qty=ext_qty)

    async def _run_private_stream(self) -> None:
        symbols = self.settings["trading"]["symbols"]
        try:
            async for event in self.exchange.stream_private_updates(symbols):
                if self._stop_requested:
                    return
                self._last_private_event_at = utcnow()
                etype = event.get("type")
                data = event.get("data", {}) or {}
                if etype == "order":
                    oid = str(data.get("id", ""))
                    if oid:
                        self._open_orders_snapshot[oid] = data
                        status = str(data.get("status", "")).lower()
                        if status in {"closed", "filled", "canceled", "cancelled"}:
                            self._open_orders_snapshot.pop(oid, None)
                elif etype == "position":
                    symbol = str(data.get("symbol", ""))
                    if symbol and symbol in self.settings["trading"]["symbols"]:
                        qty = float(data.get("contracts") or data.get("positionAmt") or data.get("size") or 0.0)
                        if qty <= 0 and symbol in self.portfolio.positions and not self._has_open_order_for_symbol(symbol):
                            self.portfolio.positions.pop(symbol, None)
        except Exception as exc:
            log_event(self.logger, "WARNING", "private_stream_error", "Private stream failed", error=str(exc))

    def _has_open_order_for_symbol(self, symbol: str) -> bool:
        return any((o.get("symbol") == symbol) for o in self._open_orders_snapshot.values())

    def _is_position_consistent(self, symbol: str) -> tuple[bool, str]:
        pos = self.portfolio.positions.get(symbol)
        if not pos:
            return True, "no_position"
        if pos.qty <= 0:
            return False, "invalid_internal_position_size"
        return True, "ok"

    def _health_guard(self) -> bool:
        cfg = self.settings["execution"]
        if self.exchange.should_pause_trading(
            max_consecutive_failures=int(cfg.get("max_exchange_failures_before_pause", 5)),
            timeout_spike_ms=float(cfg.get("timeout_spike_ms", 8000)),
        ):
            if time.time() - self._last_alerted_health > 60:
                self._last_alerted_health = time.time()
                health = self.exchange.get_health_snapshot()
                log_event(self.logger, "WARNING", "health_guard", "Trading paused: exchange degraded", **health)
                self.notifier.send(f"Trading paused: exchange degraded {health}")
            return True
        return False

    def _log_db(self, level: str, source: str, message: str) -> None:
        with self.db.session_scope() as session:
            session.add(LogEntry(level=level, source=source, message=message))

    def _get_available_balance(self) -> float:
        if self.mode == "paper":
            return self.paper_cash
        balance = self.exchange.get_balance()
        quote = self.settings["trading"]["quote_currency"]
        return float(balance.get("total", {}).get(quote, 0.0))

    def _get_total_equity(self) -> float:
        if self.mode != "paper":
            return self._get_available_balance()
        equity = self.paper_cash
        for symbol, pos in self.portfolio.positions.items():
            mark = self._last_prices.get(symbol, pos.entry_price)
            # Long: tutulan varlık değeri (+); Short: geri alma yükümlülüğü (-)
            equity += pos.qty * mark if pos.side.upper() == "BUY" else -pos.qty * mark
        return equity

    def _record_balance(self, total_balance: float) -> None:
        with self.db.session_scope() as session:
            session.add(BalanceHistory(total_balance=total_balance, currency=self.settings["trading"]["quote_currency"], timestamp=utcnow()))

    def _context_metrics(self, df: Any) -> dict[str, float]:
        close = df["close"].astype(float)
        returns = close.pct_change().dropna().tail(30)
        volatility = float(returns.std(ddof=0)) if not returns.empty else 0.0
        volume = float(df["volume"].astype(float).iloc[-1]) if not df.empty else 1.0
        return {"volatility": volatility, "volume": volume}

    def _process_telegram_commands(self) -> None:
        self.notifier.poll_commands(lambda cmd: "OK" if cmd == "/status" else "")

    def _can_trade_symbol(self, symbol: str) -> bool:
        cooldown = int(self.settings["trading"].get("cooldown_seconds", 0))
        last = self.last_trade_at.get(symbol)
        return not last or (utcnow() - last).total_seconds() >= cooldown

    def _idempotency_key(self, symbol: str, side: str, candle_ts: datetime, strategy_name: str) -> str:
        raw = f"{symbol}|{side}|{candle_ts.isoformat()}|{strategy_name}"
        digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:24]
        return f"cb-{digest}"

    def _is_duplicate_submission(self, key: str) -> bool:
        return key in self._submitted_order_keys

    def _mark_submission(self, key: str) -> None:
        self._submitted_order_keys.add(key)
        if len(self._submitted_order_keys) > 10000:
            self._submitted_order_keys = set(list(self._submitted_order_keys)[-6000:])

    def _stale_or_clock_drift_detected(self, symbol: str, candle_ts: datetime) -> tuple[bool, str]:
        max_age = int(self.settings.get("runtime", {}).get("max_stale_data_seconds", 180))
        age = (utcnow() - candle_ts).total_seconds()
        if age > max_age:
            return True, f"stale data for {symbol}: age={age:.1f}s"

        max_drift_ms = int(self.settings.get("runtime", {}).get("max_clock_drift_ms", 5000))
        server_ms = self.exchange.fetch_server_time_ms()
        if server_ms is not None:
            local_ms = int(time.time() * 1000)
            drift = abs(local_ms - server_ms)
            if drift > max_drift_ms:
                return True, f"clock drift too high: {drift}ms"
        return False, "ok"

    def _select_strategy(self, regime: str) -> tuple[Strategy | None, str]:
        enabled = [s.lower() for s in self.settings["trading"].get("enable_strategies", list(self.strategies.keys()))]
        for name in enabled:
            strategy = self.strategies.get(name)
            if not strategy:
                continue
            if regime not in strategy.supported_regimes:
                continue
            disabled, reason = self.performance_tracker.is_disabled(name)
            if disabled:
                continue
            return strategy, "selected"
        return None, "no_compatible_strategy"

    def _active_strategies_for_regime(self, regime: str) -> list[Strategy]:
        """Rejime uyumlu ve devre dışı bırakılmamış tüm stratejileri döner."""
        enabled = [s.lower() for s in self.settings["trading"].get("enable_strategies", list(self.strategies.keys()))]
        result = []
        for name in enabled:
            strategy = self.strategies.get(name)
            if not strategy:
                continue
            if regime not in strategy.supported_regimes:
                continue
            disabled, _ = self.performance_tracker.is_disabled(name)
            if disabled:
                continue
            result.append(strategy)
        return result

    def _consensus_decision(
        self, regime: str, frames: dict[str, Any]
    ) -> tuple[StrategyDecision | None, list[str], dict[str, Any]]:
        """
        Tüm uyumlu stratejileri çalıştırır ve çoğunluk oylamasıyla karar verir.
        Basit çoğunluk gerekir: aynı yönde en az floor(n/2)+1 oy.
        Tek strateji varsa doğrudan o kullanılır.
        Dönüş: (karar, oy_veren_stratejiler, oylama_meta)
        """
        strategies = self._active_strategies_for_regime(regime)
        if not strategies:
            return None, [], {"reason": "no_compatible_strategy"}

        votes: dict[str, list[str]] = {"BUY": [], "SELL": [], "HOLD": []}
        all_meta: dict[str, Any] = {}
        for strategy in strategies:
            decision = strategy.generate_decision(frames)
            sig = decision.signal if decision.signal in votes else "HOLD"
            votes[sig].append(strategy.name)
            all_meta[strategy.name] = {
                "signal": sig,
                "reason": decision.reason,
                **decision.metadata,
            }

        n = len(strategies)
        # Yapılandırılabilir eşik: min_consensus_votes > 0 ise o değer kullanılır
        # (n ile sınırlı, en az 1), aksi halde basit çoğunluk (n//2+1).
        cfg_votes = int(self.settings["trading"].get("min_consensus_votes", 0))
        if cfg_votes > 0:
            threshold = max(1, min(cfg_votes, n))
        else:
            threshold = n // 2 + 1

        for signal in ("BUY", "SELL"):
            if len(votes[signal]) >= threshold:
                voters = votes[signal]
                avg_confirmation = sum(
                    float(all_meta[v].get("confirmation", 0.5)) for v in voters
                ) / len(voters)
                decision = StrategyDecision(
                    signal=signal,
                    reason=f"consensus_{signal.lower()}:{','.join(voters)}",
                    metadata={
                        "confirmation": avg_confirmation,
                        "voters": voters,
                        "votes": {k: len(v) for k, v in votes.items()},
                        **{k: v for k, v in all_meta.items()},
                    },
                )
                return decision, voters, {"votes": votes, "threshold": threshold}

        # Çoğunluk sağlanamadı → HOLD
        hold_decision = StrategyDecision(
            signal="HOLD",
            reason=f"no_consensus:buy={len(votes['BUY'])},sell={len(votes['SELL'])},threshold={threshold}",
            metadata={"votes": {k: len(v) for k, v in votes.items()}, **all_meta},
        )
        return hold_decision, [], {"votes": votes, "threshold": threshold}

    def _register_trade_result(self, strategy_name: str, pnl: float) -> None:
        """İşlem sonucunu performans takipçisine kaydeder ve risk yöneticisine
        güncel (gerçek) küresel kazanma oranıyla birlikte iletir → Kelly adaptif olur."""
        self.performance_tracker.record_trade(strategy_name, pnl)
        win_rate, total_trades = self.performance_tracker.global_win_rate()
        min_trades = int(self.settings.get("performance", {}).get("min_trades_for_eval", 10))
        self.risk.register_trade_outcome(pnl, win_rate=win_rate if total_trades >= min_trades else None)

    def _apply_paper_close_cash(self, side: str, qty: float, price: float, fee: float = 0.0) -> None:
        """Paper modda pozisyon kapanışının nakit etkisini uygular.
        Long kapanışı (sat) → nakit girer; short kapanışı (geri al) → nakit çıkar.
        """
        if self.mode != "paper":
            return
        if side.upper() == "BUY":
            self.paper_cash += qty * price - fee
        else:
            self.paper_cash -= qty * price + fee

    def _check_protective_levels(self, symbol: str, current_price: float) -> bool:
        """
        Paper modda SL, tam TP ve kademeli TP seviyelerini fiyata göre kontrol eder.
        Live modda exchange kendi emirlerini yönetir.
        True döner → pozisyon kapatıldı/değiştirildi.
        """
        if self.mode != "paper":
            return False
        pos = self.portfolio.positions.get(symbol)
        if not pos:
            return False
        is_long = pos.side.upper() == "BUY"
        side = pos.side

        # Stop-loss tetiklendi (long: fiyat ≤ SL, short: fiyat ≥ SL)
        sl_hit = (is_long and current_price <= pos.stop_loss) or ((not is_long) and current_price >= pos.stop_loss)
        if sl_hit:
            trade = self.portfolio.close_position(symbol, current_price)
            if trade:
                self._apply_paper_close_cash(side, trade["qty"], current_price)
                self._register_trade_result(pos.strategy_name, trade["pnl"])
                log_event(self.logger, "WARNING", "stop_loss_hit", "SL triggered",
                          symbol=symbol, side=side, price=current_price, sl=pos.stop_loss, pnl=trade["pnl"])
                self.notifier.send(f"SL hit {symbol} @ {current_price:.4f} PnL={trade['pnl']:.2f}")
            return True

        # Tam TP tetiklendi (long: fiyat ≥ TP, short: fiyat ≤ TP)
        tp_hit = (is_long and current_price >= pos.take_profit) or ((not is_long) and current_price <= pos.take_profit)
        if tp_hit:
            trade = self.portfolio.close_position(symbol, current_price)
            if trade:
                self._apply_paper_close_cash(side, trade["qty"], current_price)
                self._register_trade_result(pos.strategy_name, trade["pnl"])
                log_event(self.logger, "INFO", "take_profit_hit", "TP triggered",
                          symbol=symbol, side=side, price=current_price, tp=pos.take_profit, pnl=trade["pnl"])
                self.notifier.send(f"TP hit {symbol} @ {current_price:.4f} PnL={trade['pnl']:.2f}")
            return True

        # Kademeli TP seviyeleri (long: fiyat ≥ hedef, short: fiyat ≤ hedef)
        for target in pos.partial_tp_targets:
            if target.get("hit"):
                continue
            reached = (is_long and current_price >= target["price"]) or ((not is_long) and current_price <= target["price"])
            if reached:
                close_qty = pos.qty * target["close_pct"]
                trade = self.portfolio.partial_close_position(symbol, close_qty, current_price)
                target["hit"] = True
                if trade:
                    self._apply_paper_close_cash(side, trade["qty"], current_price)
                    self._register_trade_result(pos.strategy_name, trade["pnl"])
                    log_event(self.logger, "INFO", "partial_tp_hit", "Partial TP triggered",
                              symbol=symbol, price=current_price, qty=trade["qty"], pnl=trade["pnl"])
                    self.notifier.send(f"Partial TP {symbol} @ {current_price:.4f} qty={trade['qty']:.4f} PnL={trade['pnl']:.2f}")

        # Trailing stop ve TP güncelle
        if self.portfolio.is_open(symbol):
            pos = self.portfolio.positions[symbol]
            pos.stop_loss, pos.take_profit = self.risk.update_protective_levels(pos, current_price)
            self.portfolio.update_peak(symbol, current_price)

        return False

    def _spread_pct(self, symbol: str) -> float:
        try:
            book = self.exchange.fetch_order_book(symbol, limit=5)
            bid = float(book.get("bids", [[0.0, 0.0]])[0][0])
            ask = float(book.get("asks", [[0.0, 0.0]])[0][0])
            mid = (bid + ask) / 2 if (bid + ask) > 0 else 0.0
            return abs(ask - bid) / mid if mid > 0 else 0.0
        except Exception:
            return 0.0

    def _order_book_imbalance(self, symbol: str) -> float:
        """
        Order book ağırlıklı alıcı/satıcı dengesizliğini hesaplar.

        Formül: (bid_volume - ask_volume) / (bid_volume + ask_volume)
        Aralık: -1.0 (tamamen satış baskısı) .. +1.0 (tamamen alış baskısı)
        0.0 döner → order book kullanılamıyor veya dengeli

        Uygulama:
          imbalance > +cfg eşiği  → BUY sinyalini güçlendirir
          imbalance < -cfg eşiği  → BUY sinyalini zayıflatır / filtreler
        """
        cfg = self.settings.get("market_intelligence", {})
        if not self.settings["execution"].get("use_order_book", False):
            return 0.0
        depth = int(self.settings["execution"].get("order_book_depth", 20))
        try:
            book = self.exchange.fetch_order_book(symbol, limit=depth)
            bids = book.get("bids", [])
            asks = book.get("asks", [])
            bid_vol = sum(float(b[1]) for b in bids if len(b) >= 2)
            ask_vol = sum(float(a[1]) for a in asks if len(a) >= 2)
            total = bid_vol + ask_vol
            if total <= 0:
                return 0.0
            return round((bid_vol - ask_vol) / total, 4)
        except Exception as exc:
            self.logger.debug("Order book imbalance failed for %s: %s", symbol, exc)
            return 0.0

    def _passes_trade_filters(self, symbol: str, context: dict[str, float], regime: str, adx: float) -> tuple[bool, str]:
        cfg = self.settings.get("market_intelligence", {})
        min_volume = float(cfg.get("min_volume_threshold", 0.0))
        max_spread = float(cfg.get("max_spread_pct", 0.01))
        min_vol = float(cfg.get("min_volatility_filter", 0.0002))
        max_vol = float(cfg.get("max_volatility_filter", 0.05))
        chop_adx = float(cfg.get("no_trade_adx_threshold", 18.0))

        # Not: Order book dengesizliği tek noktada (_order_book_imbalance) hesaplanıp
        # _process_symbol içinde ob_imbalance_min eşiğiyle uygulanıyor. Burada ayrıca
        # order book çekmiyoruz (mükerrer API çağrısını önlemek için).

        spread = self._spread_pct(symbol)
        if context["volume"] < min_volume:
            return False, "volume_below_threshold"
        if spread > max_spread:
            return False, "spread_too_wide"
        if context["volatility"] < min_vol or context["volatility"] > max_vol:
            return False, "volatility_out_of_bounds"
        if regime == "RANGE" and adx < chop_adx:
            return False, "choppy_no_trade_zone"
        return True, "ok"

    def _net_edge_ok(self, symbol: str, side: str, entry_price: float, tp_targets: list[dict]) -> tuple[bool, str]:
        """İlk kademeli TP hedefi, gidiş-dönüş maliyetini (fee + slippage) yeterli
        marjla aşmıyorsa işlemi reddeder → negatif beklenen değerli scalp'leri önler.
        """
        cfg = self.settings.get("market_intelligence", {})
        margin = float(cfg.get("min_net_edge_margin", 1.5))
        if not tp_targets:
            return True, "no_tp_targets"
        first_tp = tp_targets[0]["price"]
        tp_dist_pct = abs(first_tp - entry_price) / max(entry_price, 1e-9)

        maker, taker = self.exchange.get_fee_rates(symbol)
        # Giriş post-only ise maker, çıkış limit ise maker; güvenli taraf için taker varsay
        entry_fee = taker if not self.settings["execution"].get("use_post_only_entry", False) else maker
        exit_fee = maker  # kademeli TP'ler limit (maker) emir
        slippage = float(self.settings["risk"]["max_slippage_pct"])
        round_trip_cost = entry_fee + exit_fee + 2 * slippage
        required = round_trip_cost * margin
        if tp_dist_pct < required:
            return False, f"edge_below_cost:{tp_dist_pct:.4f}<{required:.4f}"
        return True, "ok"

    def _position_quality_ok(self, decision: Any, adx: float) -> tuple[bool, str]:
        cfg = self.settings.get("market_intelligence", {})
        min_trend = float(cfg.get("min_trend_strength_adx", 20.0))
        min_confirm = float(cfg.get("min_confirmation_score", 0.4))
        trend_strength = float(decision.metadata.get("trend_strength", adx / 100.0))
        confirmation = float(decision.metadata.get("confirmation", decision.metadata.get("confidence", 0.0)))
        if adx < min_trend and trend_strength < min_confirm:
            return False, "weak_trend_strength"
        if confirmation < min_confirm and decision.signal == "BUY":
            return False, "weak_confirmation"
        return True, "ok"

    def _needed_timeframes(self, primary_tf: str) -> list[str]:
        """primary + higher_timeframe + tüm etkin stratejilerin confirm/primary tf'lerinin birleşimi.

        Stratejiler kendi confirm_timeframe'lerini frames içinde bulamayınca primary'e
        düşüp MTF onayını etkisiz bırakıyordu; bu yüzden ihtiyaç duyulan tüm zaman
        dilimlerini önceden yüklüyoruz.
        """
        tfs: list[str] = [primary_tf]
        higher_tf = self.settings["trading"].get("higher_timeframe", primary_tf)
        if higher_tf not in tfs:
            tfs.append(higher_tf)
        for strategy in self.strategies.values():
            for key in ("primary_timeframe", "confirm_timeframe"):
                tf = strategy.params.get(key)
                if tf and tf not in tfs:
                    tfs.append(tf)
        return tfs

    def _load_symbol_frames(self, symbol: str, primary_tf: str) -> dict[str, Any]:
        frames: dict[str, Any] = {}
        for tf in self._needed_timeframes(primary_tf):
            df = self.data.get_cached_data(symbol, tf)
            if df.empty or len(df) < 100:
                df = self.data.fetch_symbol_data(symbol, tf, limit=700)
            frames[tf] = df
        return frames

    def _update_trade_row_on_close(self, symbol: str) -> None:
        with self.db.session_scope() as session:
            session.query(Trade).filter(Trade.symbol == symbol, Trade.status == "open").update({"status": "closed"})

    def _apply_exit(self, symbol: str, order: dict[str, Any], fallback_price: float) -> None:
        if order.get("status") != "FILLED":
            return
        pos = self.portfolio.positions.get(symbol)
        if not pos:
            return
        closing_side = pos.side  # kapatılan pozisyonun yönü (nakit işareti için)
        fill_price = float(order.get("average") or order.get("price") or fallback_price)
        fill_qty = float(order.get("filled") or pos.qty)
        fee_paid = float(order.get("fee_paid", 0.0))
        trade = self.portfolio.close_position(symbol, fill_price, fee_paid=fee_paid)
        if not trade:
            return
        self._apply_paper_close_cash(closing_side, fill_qty, fill_price, fee_paid)
        self._update_trade_row_on_close(symbol)
        with self.db.session_scope() as session:
            session.add(
                Trade(
                    symbol=trade["symbol"],
                    side=trade["side"],
                    qty=trade["qty"],
                    entry_price=trade["entry_price"],
                    exit_price=trade["exit_price"],
                    pnl=trade["pnl"],
                    status="closed",
                    opened_at=trade["opened_at"],
                    closed_at=trade["closed_at"],
                )
            )
        self._register_trade_result(str(trade.get("strategy_name", "unknown")), float(trade["pnl"]))
        expected = float(order.get("execution_details", {}).get("expected_price", fallback_price))
        drift = abs(fill_price - expected) / max(expected, 1e-9)
        log_event(
            self.logger,
            "INFO",
            "live_backtest_drift",
            "Fill drift measured",
            symbol=symbol,
            expected_price=expected,
            actual_price=fill_price,
            drift_pct=drift,
            strategy_name=trade.get("strategy_name", "unknown"),
        )

    def _process_symbol(self, symbol: str, timeframe: str, current_equity: float, frames: dict | None = None) -> None:
        if frames is None:
            frames = self._load_symbol_frames(symbol, timeframe)
        df = frames[timeframe]
        if df.empty:
            return
        candle_ts = df.iloc[-1]["timestamp"].to_pydatetime()
        if self.processed_signal_at.get(symbol) and candle_ts <= self.processed_signal_at[symbol]:
            return
        self.processed_signal_at[symbol] = candle_ts
        stale, stale_reason = self._stale_or_clock_drift_detected(symbol, candle_ts)
        if stale:
            self._safe_shutdown(stale_reason)
            return
        last_price = float(df.iloc[-1]["close"])
        self._last_prices[symbol] = last_price

        if self._check_protective_levels(symbol, last_price):
            return

        consistent, reason = self._is_position_consistent(symbol)
        if not consistent:
            self._safe_shutdown(f"Position consistency error on {symbol}: {reason}")
            return

        context = self._context_metrics(df)
        regime_state = self.regime_detector.detect(df)
        decision, voters, vote_meta = self._consensus_decision(regime_state.regime, frames)
        if decision is None:
            log_event(
                self.logger,
                "INFO",
                "strategy_skipped",
                "No strategy selected for regime",
                symbol=symbol,
                regime=regime_state.regime,
                reason=vote_meta.get("reason", "no_compatible_strategy"),
                strategy_performance=self.performance_tracker.snapshot(),
            )
            return
        trade_id = self._trade_id()
        signal_ts = utcnow()
        self._trace(
            trade_id,
            "signal",
            symbol=symbol,
            signal=decision.signal,
            reason=decision.reason,
            price=last_price,
            voters=voters,
            vote_counts=vote_meta.get("votes", {}),
            regime=regime_state.regime,
            regime_adx=regime_state.adx,
            regime_atr_pct=regime_state.atr_pct,
        )

        if self.risk.check_market_breakers(context["volatility"], float(self.settings["risk"]["max_slippage_pct"])):
            self.notifier.send("Circuit breaker: volatility/slippage threshold breached")
            self._stop_requested = True
            return

        # --- Pozisyon yönetimi: aç / ters yönlü sinyalde kapat ---
        pos = self.portfolio.positions.get(symbol)
        if pos:
            # Ters yönlü sinyal → pozisyonu kapat. Aynı yön → piramitleme yok.
            if pos.side.upper() == "BUY" and decision.signal == "SELL":
                self._close_open_position(symbol, "sell", last_price, candle_ts, context, signal_ts, trade_id)
            elif pos.side.upper() == "SELL" and decision.signal == "BUY":
                self._close_open_position(symbol, "buy", last_price, candle_ts, context, signal_ts, trade_id)
            return

        allow_short = bool(self.settings["trading"].get("allow_short", False))
        if decision.signal == "BUY":
            entry_side = "buy"
        elif decision.signal == "SELL" and allow_short:
            entry_side = "sell"
        else:
            return
        if not self._can_trade_symbol(symbol):
            return
        filter_ok, filter_reason = self._direction_filter_ok(entry_side, frames, timeframe)
        if not filter_ok:
            log_event(
                self.logger, "INFO", "entry_blocked_by_regime_filter",
                "Regime direction filter blocked entry",
                symbol=symbol, side=entry_side, reason=filter_reason,
            )
            return
        self._open_trade(
            symbol, entry_side, last_price, candle_ts, context,
            regime_state, decision, voters, current_equity, trade_id, signal_ts,
        )

    def _direction_filter_ok(self, entry_side: str, frames: dict, primary_tf: str) -> tuple[bool, str]:
        """Rejim yön filtresi (backtest'te doğrulandı: ayı yılında kayıpları ciddi keser).

        BUY  → sadece boğa rejiminde (fiyat > EMA200, EMA50 > EMA200, EMA200 yükseliyor)
        SELL → sadece ayı rejiminde (ayna koşullar)

        Filtre higher_timeframe verisi üzerinde hesaplanır; kapalı olmayan son mumun
        etkisini azaltmak için bir önceki mumun değeri kullanılır.
        """
        if not bool(self.settings["trading"].get("regime_entry_filter", False)):
            return True, "filter_disabled"

        higher_tf = self.settings["trading"].get("higher_timeframe", primary_tf)
        df = frames.get(higher_tf)
        if df is None or df.empty:
            df = frames.get(primary_tf)
        if df is None or len(df) < 210:
            return True, "insufficient_data_for_filter"  # fail-open: veri yoksa engelleme

        if entry_side == "buy":
            allowed = bull_regime_filter(df)
        else:
            allowed = bear_regime_filter(df)
        # Son mum kapanmamış olabilir → bir önceki (kapalı) muma bak
        idx = -2 if len(allowed) >= 2 else -1
        ok = bool(allowed.iloc[idx])
        return ok, "ok" if ok else f"regime_not_{'bull' if entry_side == 'buy' else 'bear'}"

    def _close_open_position(
        self, symbol: str, close_side: str, last_price: float,
        candle_ts: datetime, context: dict, signal_ts: datetime, trade_id: str,
    ) -> None:
        pos = self.portfolio.positions[symbol]
        client_order_id = self._idempotency_key(symbol, close_side, candle_ts, pos.strategy_name)
        if self._is_duplicate_submission(client_order_id):
            self._trace(trade_id, "duplicate_skip", client_order_id=client_order_id)
            return
        self._trace(trade_id, "execution_start", side=close_side, qty=pos.qty)
        order = self.execution.place_order(
            symbol, close_side, pos.qty, last_price,
            market_context=context, signal_timestamp=signal_ts, client_order_id=client_order_id,
        )
        self._mark_submission(client_order_id)
        self._trace(trade_id, "order_update", status=order.get("status"), lifecycle=order.get("lifecycle"))
        self._apply_exit(symbol, order, fallback_price=last_price)
        self.last_trade_at[symbol] = utcnow()

    def _open_trade(
        self, symbol: str, side: str, last_price: float, candle_ts: datetime,
        context: dict, regime_state: Any, decision: Any, voters: list,
        current_equity: float, trade_id: str, signal_ts: datetime,
    ) -> None:
        side_u = side.upper()
        if self._has_open_order_for_symbol(symbol):
            self._trace(trade_id, "risk_block", reason="existing_open_order")
            return

        filters_ok, filter_reason = self._passes_trade_filters(symbol, context, regime_state.regime, regime_state.adx)
        if not filters_ok:
            self._trace(trade_id, "filter_block", reason=filter_reason, regime=regime_state.regime)
            return

        quality_ok, quality_reason = self._position_quality_ok(decision, regime_state.adx)
        if not quality_ok:
            self._trace(trade_id, "quality_block", reason=quality_reason)
            return

        available = self._get_available_balance()
        ok, risk_reason = self.risk.validate_trade(self.portfolio, symbol, available, current_equity, last_price)
        if not ok:
            self._trace(trade_id, "risk_block", reason=risk_reason)
            return

        # --- Order Book Imbalance (yön-bağımlı) ---
        # BUY: aşırı satış baskısında reddet; SELL: aşırı alış baskısında reddet.
        ob_imbalance = self._order_book_imbalance(symbol)
        ob_threshold = float(self.settings.get("market_intelligence", {}).get("ob_imbalance_min", -0.3))
        if side_u == "BUY" and ob_imbalance < ob_threshold:
            self._trace(trade_id, "ob_filter", reason="sell_pressure_dominates", imbalance=ob_imbalance, threshold=ob_threshold)
            return
        if side_u == "SELL" and ob_imbalance > -ob_threshold:
            self._trace(trade_id, "ob_filter", reason="buy_pressure_dominates", imbalance=ob_imbalance, threshold=-ob_threshold)
            return
        favor = max(0.0, ob_imbalance) if side_u == "BUY" else max(0.0, -ob_imbalance)
        ob_size_mult = 1.0 + favor * 0.2  # yön lehine baskıda maks %20 boyut artışı

        # --- Sentiment (yön-bağımlı) ---
        sentiment = self.sentiment.fetch_latest_sentiment(symbol)
        if self.sentiment.should_filter_trade(side_u):
            self._trace(trade_id, "sentiment_filter", score=sentiment.score, label=sentiment.label)
            return
        sentiment_multiplier = self.sentiment.get_risk_multiplier()

        # --- Funding Rate (yön-bağımlı) ---
        fr_signal = self.funding_rate.fetch_signal(symbol)
        fr_filter = fr_signal.filter_buy if side_u == "BUY" else fr_signal.filter_sell
        if fr_filter:
            self._trace(trade_id, "funding_rate_filter", reason=f"{side}_filtered_by_funding_rate",
                        funding_rate=fr_signal.funding_rate, funding_label=fr_signal.funding_label,
                        long_short_ratio=fr_signal.long_short_ratio)
            return
        funding_multiplier = fr_signal.risk_multiplier

        # ATR + ADX ile dinamik SL/TP
        current_atr = regime_state.atr_pct * last_price
        stop_loss, take_profit = self.risk.get_stop_take_prices(side_u, last_price, atr=current_atr, adx=regime_state.adx)

        risk_mult = self.performance_tracker.adaptive_risk_multiplier(context["volatility"])
        corr_mult = self.risk.correlation_size_multiplier(self.portfolio, symbol)
        qty = self.risk.calculate_position_size(
            available * risk_mult * sentiment_multiplier * ob_size_mult * corr_mult * funding_multiplier,
            last_price, stop_loss, volatility=context["volatility"],
        )
        if qty <= 0:
            return

        tp_targets = self.risk.get_partial_tp_targets(side_u, last_price)

        # Maliyet-farkındalıklı edge filtresi (fee + slippage'ı aşamayan işlemi atla)
        edge_ok, edge_reason = self._net_edge_ok(symbol, side, last_price, tp_targets)
        if not edge_ok:
            self._trace(trade_id, "edge_filter", reason=edge_reason)
            return

        consensus_name = ",".join(voters) if voters else "consensus"
        self._trace(trade_id, "execution_start", side=side, qty=qty, sentiment_score=sentiment.score)
        client_order_id = self._idempotency_key(symbol, side, candle_ts, consensus_name)
        if self._is_duplicate_submission(client_order_id):
            self._trace(trade_id, "duplicate_skip", client_order_id=client_order_id)
            return

        order = self.execution.place_managed_trade(
            symbol, side, qty, last_price,
            tp_targets=tp_targets, stop_loss_price=stop_loss, market_context=context,
        )
        self._mark_submission(client_order_id)
        self._trace(trade_id, "order_update", status=order.get("status"), lifecycle=order.get("lifecycle"),
                    drift=order.get("execution_details", {}).get("drift_pct"),
                    execution_delay_ms=order.get("execution_delay_ms"))
        if order.get("status") != "FILLED":
            return

        fill_price = float(order.get("average") or order.get("price") or last_price)
        fill_qty = float(order.get("filled") or qty)
        fee_paid = float(order.get("fee_paid", 0.0))
        if self.mode == "paper":
            if side_u == "BUY":
                self.paper_cash -= (fill_qty * fill_price) + fee_paid
            else:  # short açılışı: satış geliri nakde eklenir, fee düşülür
                self.paper_cash += (fill_qty * fill_price) - fee_paid

        tracked_tp_targets = [{"price": t["price"], "close_pct": t["close_pct"], "hit": False} for t in tp_targets]
        self.portfolio.open_position(
            Position(
                symbol=symbol, side=side_u, entry_price=fill_price, qty=fill_qty, entry_fee=fee_paid,
                stop_loss=stop_loss, take_profit=take_profit, peak_price=fill_price,
                strategy_name=consensus_name, partial_tp_targets=tracked_tp_targets,
            )
        )
        with self.db.session_scope() as session:
            session.add(
                Trade(symbol=symbol, side=side_u, qty=fill_qty, entry_price=fill_price,
                      exit_price=None, pnl=-fee_paid, status="open")
            )
        self.last_trade_at[symbol] = utcnow()

    def _reload_config_if_changed(self) -> None:
        try:
            interval = float(self.settings.get("runtime", {}).get("config_reload_seconds", 15))
            now = time.time()
            if now - self._last_config_check < interval:
                return
            self._last_config_check = now
            
            p = Path(self.settings_path)
            if not p.exists():
                return
                
            mtime = p.stat().st_mtime
            if mtime <= self._settings_mtime:
                return
            
            self._settings_mtime = mtime
            perf_state = self.performance_tracker.snapshot()
            new_settings = load_settings(self.settings_path)
            
            # Başarıyla yüklendiyse güncelle
            self.settings = new_settings
            self.strategies = build_strategies(self.settings)
            self.primary_strategy_name = self.settings["strategy"]["name"].lower()
            self.regime_detector = self._build_regime_detector(self.settings)
            self.performance_tracker = self._build_performance_tracker(self.settings)
            self.performance_tracker.restore(perf_state)
            self.risk = self._build_risk_manager(self.settings)
            self.execution = self._build_execution_engine(self.settings)
            self.logger.info("Configuration reloaded successfully")
        except Exception as e:
            # Hata alırsak logla ama botu durdurma
            self.logger.warning(f"Could not reload config: {e}")

    def run_backtest(self) -> None:
        symbol = self.settings["trading"]["symbols"][0]
        data = self.data.fetch_symbol_data(symbol, self.settings["trading"]["timeframe"], limit=3000)
        maker, taker = self.exchange.get_fee_rates(symbol)
        strategy = self.strategies.get(self.primary_strategy_name) or next(iter(self.strategies.values()))
        result = BacktestEngine(
            strategy=strategy,
            initial_balance=float(self.settings["app"].get("initial_paper_balance", 10_000)),
            maker_fee_pct=maker,
            taker_fee_pct=taker,
            slippage_pct=self.settings["risk"]["max_slippage_pct"],
            execution_delay_candles=int(self.settings["execution"].get("backtest_execution_delay_candles", 1)),
        ).run(data)
        log_event(self.logger, "INFO", "backtest_result", "Backtest completed", metrics=result.metrics)

    async def _run_polling_loop(self) -> None:
        timeframe = self.settings["trading"]["timeframe"]
        cadence = max(
            timeframe_to_seconds(timeframe),
            int(self.settings["trading"].get("scheduler_interval_seconds", 60)),
        )
        self.risk.start_day(self._get_total_equity())
        self.risk.start_week(self._get_total_equity())
        if self.settings["execution"].get("use_private_ws", True):
            self._private_stream_task = asyncio.create_task(self._run_private_stream())
        while not self._stop_requested:
            try:
                if self._private_stream_task and self._private_stream_task.done():
                    err = self._private_stream_task.exception()
                    if err:
                        log_event(self.logger, "WARNING", "private_stream_stopped", "Private stream stopped", error=str(err))
                        self._private_stream_task = asyncio.create_task(self._run_private_stream())
                if self._health_guard():
                    await asyncio.sleep(5)
                    continue
                self._reload_config_if_changed()
                self._process_telegram_commands()
                self._reconcile_state()
                equity = self._get_total_equity()
                self.risk.roll_periods(equity)
                self._record_balance(equity)
                log_event(self.logger, "INFO", "heartbeat", "Loop cycle",
                          equity=round(equity, 2),
                          open_positions=self.portfolio.open_count(),
                          cash=round(self.paper_cash, 2) if self.mode == "paper" else None,
                          exchange_failures=self.exchange.consecutive_failures)
                symbols = self.settings["trading"]["symbols"]

                # Tüm semboller için veri çekimini paralel yap (IO bağımlı → thread pool ile)
                loop = asyncio.get_event_loop()
                fetch_tasks = [
                    loop.run_in_executor(None, self._load_symbol_frames, sym, timeframe)
                    for sym in symbols
                ]
                prefetched_list = await asyncio.gather(*fetch_tasks, return_exceptions=True)
                prefetched: dict[str, Any] = {}
                for sym, result in zip(symbols, prefetched_list):
                    if isinstance(result, Exception):
                        log_event(self.logger, "WARNING", "prefetch_failed",
                                  f"Data fetch failed for {sym}", symbol=sym, error=str(result))
                    else:
                        prefetched[sym] = result

                # Sinyal işleme sıralı (paylaşılan durum koruması için)
                for symbol in symbols:
                    self._process_symbol(symbol, timeframe, equity, frames=prefetched.get(symbol))
                if self.risk.check_circuit_breaker(equity):
                    self._safe_shutdown("Circuit breaker active")
                    return
                self._save_state()
                await asyncio.sleep(next_run_sleep(cadence))
            except Exception as exc:
                log_event(self.logger, "ERROR", "polling_loop_error", str(exc))
                try:
                    self._log_db("ERROR", "polling_loop", str(exc))
                except Exception:
                    pass
                self.notifier.send(f"Loop error: {exc}")
                await asyncio.sleep(5)
        if self._private_stream_task:
            self._private_stream_task.cancel()

    async def run_live_or_paper(self) -> None:
        return await self._run_polling_loop()


async def run_with_restart(bot: TradingBot) -> None:
    backoff = 3
    while True:
        try:
            if bot.mode == "backtest":
                bot.run_backtest()
                return
            await bot.run_live_or_paper()
            return
        except Exception as exc:
            log_event(bot.logger, "ERROR", "fatal_crash", f"Fatal crash, restarting in {backoff}s", error=str(exc))
            bot._safe_shutdown(f"Fatal crash: {exc}")
            bot._stop_requested = False  # sonraki döngü için sıfırla
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60)


def main() -> None:
    load_dotenv()
    settings_path = "crypto_bot/config/settings.yaml"
    bot = TradingBot(load_settings(settings_path), settings_path)
    try:
        asyncio.run(run_with_restart(bot))
    except KeyboardInterrupt:
        bot._safe_shutdown("KeyboardInterrupt")


if __name__ == "__main__":
    main()
