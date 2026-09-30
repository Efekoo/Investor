from __future__ import annotations

"""Bot replay backtest: gerçek TradingBot'u geçmiş veride mum mum çalıştırır.

Ayrı bir backtest motoru yerine botun kendi kodu (konsensüs, rejim + yön filtresi,
ATR/ADX SL-TP, kademeli TP, trailing, risk boyutlama, kaldıraç/likidasyon, paper
execution ve komisyonlar) kullanılır; böylece test edilen şey canlıda çalışanla aynıdır.

Simülasyona özgü olanlar:
  - Saat: main/portfolio/risk/execution modüllerindeki datetime.now simülasyon saatine bağlanır
  - Borsa: ReplayExchange geçmiş fiyatları, sabit maker/taker ücretini ve simetrik
    (dengesizliği 0) bir emir defteri döndürür
  - Yalnızca canlıda anlamlı filtreler (sentiment, anlık funding/long-short, order book
    dengesizliği) nötrdür — rapor bunu belirtir
  - Mum içi SL/TP: önce aleyhte uç (SL/likidasyon, gap'te açılış fiyatı), sonra lehte
    uç (TP limitleri kendi fiyatından) kontrol edilir → ihtiyatlı sıra
  - Funding: geçmiş funding oranlarıyla her 8 saatte bir açık pozisyonlardan tahsil edilir
  - Devre kesiciler: canlıda bot durur ve yeniden başlar; burada ertesi güne kadar yeni
    işlem açılmaz
"""

import argparse
import copy
import logging
import random
import tempfile
import time
from contextlib import ExitStack
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import pandas as pd

from crypto_bot.backtest.data_loader import load_or_download, resample_ohlcv, timeframe_to_ms
from crypto_bot.backtest.metrics import max_drawdown

DEFAULT_SETTINGS = Path(__file__).resolve().parents[1] / "config" / "settings.yaml"
HTF_WINDOW = 700  # canlı bot her zaman dilimi için 700 mum çeker


# ── Simülasyon saati ─────────────────────────────────────────────────────────
class SimClock:
    now_value: datetime = datetime(2020, 1, 1, tzinfo=timezone.utc)


class SimDatetime(datetime):
    @classmethod
    def now(cls, tz=None):  # type: ignore[override]
        value = SimClock.now_value
        return value if tz is not None else value.replace(tzinfo=None)


def _patch_clock(stack: ExitStack) -> None:
    for module in ("crypto_bot.main", "crypto_bot.core.portfolio", "crypto_bot.core.risk", "crypto_bot.core.execution"):
        stack.enter_context(mock.patch(f"{module}.datetime", SimDatetime))


# ── Sahte borsa ve nötr canlı-veri kaynakları ────────────────────────────────
class ReplayExchange:
    def __init__(self, maker_fee: float, taker_fee: float, spread_pct: float, futures: bool) -> None:
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.spread_pct = spread_pct
        self.prices: dict[str, float] = {}
        self.consecutive_failures = 0
        self.config = type("Cfg", (), {"market_type": "swap" if futures else "spot"})()
        self.is_futures = futures

    def load_markets(self, force_reload: bool = False) -> dict:
        return {}

    def configure_leverage(self, *args: Any, **kwargs: Any) -> None:
        return None

    def get_fee_rates(self, symbol: str) -> tuple[float, float]:
        return self.maker_fee, self.taker_fee

    def validate_order(self, symbol: str, price: float, amount: float) -> tuple[float, float]:
        if price <= 0 or amount <= 0:
            from crypto_bot.core.exchange import OrderValidationError

            raise OrderValidationError("non-positive order")
        return price, amount

    def estimate_fill_price(self, symbol: str, side: str, amount: float, limit: int = 50) -> float:
        half = self.prices[symbol] * self.spread_pct / 2
        return self.prices[symbol] + (half if side.lower() == "buy" else -half)

    def fetch_order_book(self, symbol: str, limit: int = 50) -> dict:
        p = self.prices[symbol]
        half = p * self.spread_pct / 2
        return {"bids": [[p - half, 1.0]], "asks": [[p + half, 1.0]]}

    def fetch_ticker(self, symbol: str) -> dict:
        return {"last": self.prices[symbol], "close": self.prices[symbol]}

    def fetch_server_time_ms(self) -> None:
        return None

    def should_pause_trading(self, *args: Any, **kwargs: Any) -> bool:
        return False

    def get_health_snapshot(self) -> dict:
        return {}

    def get_balance(self) -> dict:
        return {"total": {}}

    def get_open_orders(self, symbol: str | None = None) -> list:
        return []

    def fetch_positions(self, symbols: list[str]) -> dict[str, float]:
        return {s: 0.0 for s in symbols}


class _NeutralSentiment:
    def fetch_latest_sentiment(self, symbol: str) -> Any:
        return type("S", (), {"score": 50, "label": "NEUTRAL"})()

    def should_filter_trade(self, side: str) -> bool:
        return False

    def get_risk_multiplier(self) -> float:
        return 1.0


class _NeutralFunding:
    def fetch_signal(self, symbol: str) -> Any:
        return type("F", (), {"filter_buy": False, "filter_sell": False, "risk_multiplier": 1.0,
                              "funding_rate": 0.0, "funding_label": "NEUTRAL", "long_short_ratio": 1.0})()


class _SilentNotifier:
    def send(self, message: str) -> None:
        return None

    def poll_commands(self, handler: Any) -> None:
        return None


# ── Ayarlar ──────────────────────────────────────────────────────────────────
def retime_settings(settings: dict[str, Any], timeframe: str) -> dict[str, Any]:
    """Botu ve tüm stratejileri verilen ana zaman dilimine taşır.

    Stratejilerin primary_timeframe'i ana zaman dilimi olur; confirm_timeframe ana
    zaman diliminden küçük/eşitse higher_timeframe'e yükseltilir.
    """
    s = copy.deepcopy(settings)
    s["trading"]["timeframe"] = timeframe
    higher = s["trading"].get("higher_timeframe", "1h")
    if timeframe_to_ms(higher) <= timeframe_to_ms(timeframe):
        higher = "4h" if timeframe_to_ms(timeframe) < timeframe_to_ms("4h") else "1d"
        s["trading"]["higher_timeframe"] = higher
    for params in s["strategy"]["params"].values():
        params["primary_timeframe"] = timeframe
        confirm = params.get("confirm_timeframe")
        if confirm and timeframe_to_ms(confirm) <= timeframe_to_ms(timeframe):
            params["confirm_timeframe"] = higher
    return s


def _sim_settings(base: dict[str, Any], workdir: Path, symbols: list[str], leverage: float | None) -> dict[str, Any]:
    s = copy.deepcopy(base)
    s["app"]["mode"] = "paper"
    s["app"]["log_level"] = "CRITICAL"
    s["app"]["log_json"] = False
    s["app"]["log_dir"] = str(workdir / "logs")
    s["app"]["data_dir"] = str(workdir)
    s["trading"]["symbols"] = symbols
    s["runtime"]["state_file"] = str(workdir / "state.json")
    s["runtime"]["commands_dir"] = str(workdir / "commands")
    s["database"]["url"] = f"sqlite:///{workdir / 'sim.db'}"
    s.setdefault("safety", {})["kill_switch_file"] = str(workdir / "KILL_SWITCH")
    s["safety"]["global_kill_switch"] = False
    s["notifications"]["telegram_enabled"] = False
    s["execution"]["simulate_order_delay_seconds"] = 0
    s["execution"]["poll_interval_seconds"] = 0
    if leverage is not None:
        lev = s.setdefault("leverage", {})
        lev["enabled"] = leverage > 1
        lev["leverage"] = max(1.0, leverage)
        lev["max_leverage"] = max(float(lev.get("max_leverage", 10)), leverage)
    return s


# ── Veri ─────────────────────────────────────────────────────────────────────
@dataclass
class SymbolData:
    frames: dict[str, pd.DataFrame]          # tf → tüm geçmiş
    ts_ns: dict[str, np.ndarray]             # tf → mum açılış zamanları (ns)
    close_ns: dict[str, np.ndarray]          # tf → mum kapanış zamanları (ns)
    funding_ns: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))  # funding zamanları
    funding_rates: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))

    def set_funding(self, series: pd.Series) -> None:
        if series.empty:
            return
        series = series.sort_index()
        self.funding_ns = series.index.to_numpy(dtype="datetime64[ns]").astype(np.int64)
        self.funding_rates = series.to_numpy(dtype=float)


def prepare_symbol_data(base_df: pd.DataFrame, timeframes: list[str], base_tf: str) -> SymbolData:
    frames, ts_ns, close_ns = {}, {}, {}
    base_df = base_df.sort_values("timestamp").reset_index(drop=True)
    for tf in timeframes:
        df = base_df if tf == base_tf else resample_ohlcv(base_df, tf)
        df = df.reset_index(drop=True)
        frames[tf] = df
        ts = df["timestamp"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
        ts_ns[tf] = ts
        close_ns[tf] = ts + timeframe_to_ms(tf) * 1_000_000
    return SymbolData(frames, ts_ns, close_ns)


def load_funding_history(exchange_id: str, symbol: str, days: int, cache_dir: str | Path) -> pd.Series:
    """Geçmiş funding oranları (8 saatlik). Ağ yoksa boş seri döner."""
    cache = Path(cache_dir) / f"{exchange_id}_{symbol.replace('/', '').replace(':', '_')}_funding.csv"
    since = int((pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)).timestamp() * 1000)
    rows: list[dict] = []
    if cache.exists():
        cached = pd.read_csv(cache)
        cached["timestamp"] = pd.to_datetime(cached["timestamp"], utc=True, format="ISO8601")
        if not cached.empty and cached["timestamp"].min() <= pd.Timestamp(since, unit="ms", tz="UTC") + pd.Timedelta(hours=8) \
                and cached["timestamp"].max() >= pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=9):
            return cached.set_index("timestamp")["rate"]
    try:
        import ccxt

        ex = getattr(ccxt, exchange_id)({"enableRateLimit": True, "options": {"defaultType": "swap"}})
        cursor = since
        while True:
            batch = ex.fetch_funding_rate_history(symbol, since=cursor, limit=1000)
            if not batch:
                break
            rows.extend({"timestamp": pd.Timestamp(r["timestamp"], unit="ms", tz="UTC"), "rate": float(r["fundingRate"])} for r in batch)
            if batch[-1]["timestamp"] <= cursor or len(batch) < 1000:
                break
            cursor = batch[-1]["timestamp"] + 1
    except Exception as exc:
        print(f"  ! funding geçmişi alınamadı ({symbol}): {exc}")
        return pd.Series(dtype=float)
    df = pd.DataFrame(rows).drop_duplicates("timestamp").sort_values("timestamp")
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False)
    return df.set_index("timestamp")["rate"]


# ── Simülasyon ───────────────────────────────────────────────────────────────
@dataclass
class ReplayResult:
    label: str
    metrics: dict[str, Any]
    equity: pd.DataFrame
    trades: pd.DataFrame
    errors: dict[str, int]
    funnel: dict[str, int] = field(default_factory=dict)


def run_replay(
    settings: dict[str, Any],
    data: dict[str, SymbolData],
    start: pd.Timestamp,
    *,
    leverage: float | None = None,
    label: str = "",
    spread_pct: float = 0.0001,
    maker_fee: float | None = None,
    taker_fee: float | None = None,
    seed: int = 7,
    progress: bool = False,
) -> ReplayResult:
    """Botu `start` anından verinin sonuna kadar çalıştırır."""
    from crypto_bot import main as bot_main

    symbols = list(data)
    tf = settings["trading"]["timeframe"]
    tf_ns = timeframe_to_ms(tf) * 1_000_000
    workdir = Path(tempfile.mkdtemp(prefix="replay_"))
    sim = _sim_settings(settings, workdir, symbols, leverage)
    futures = bool(sim.get("leverage", {}).get("enabled", False))
    fee = float(sim["execution"].get("fee_pct", 0.001))
    maker = maker_fee if maker_fee is not None else (0.0002 if futures else fee)
    taker = taker_fee if taker_fee is not None else (0.0005 if futures else fee)
    exchange = ReplayExchange(maker, taker, spread_pct, futures)
    random.seed(seed)

    # Zaman ekseni: ana zaman diliminin tüm mumlarının birleşimi
    timeline = np.unique(np.concatenate([d.ts_ns[tf] for d in data.values()]))
    timeline = timeline[timeline >= start.value]
    if timeline.size == 0:
        raise ValueError("Başlangıçtan sonra veri yok")

    logger = logging.getLogger("crypto_bot")
    old_level = logger.level
    errors: dict[str, int] = {}
    equity_rows: list[tuple[int, float]] = []
    funding_paid = 0.0
    entries = 0
    funnel: dict[str, int] = {}

    def counting_log_event(_logger: Any, _level: str, event: str, _message: str, **fields: Any) -> None:
        """Botun karar hunisini sayar: her sinyal nerede elendi?"""
        if event == "trade_trace":
            stage = str(fields.get("stage", ""))
            if stage == "signal":
                key = f"signal:{fields.get('signal')}"
            elif stage in {"execution_start", "order_update", "duplicate_skip"}:
                key = f"{stage}:{fields.get('status', '')}".rstrip(":")
            else:
                reason = str(fields.get("reason", "")).split(":")[0]
                key = f"{stage}:{reason}".rstrip(":")
        elif event in {"entry_blocked_by_regime_filter", "strategy_skipped"}:
            key = f"{event}:{fields.get('reason', '')}"
        else:
            key = event
        funnel[key] = funnel.get(key, 0) + 1
    breaker_candles = 0

    with ExitStack() as stack:
        _patch_clock(stack)
        SimClock.now_value = pd.Timestamp(timeline[0], tz="UTC").to_pydatetime()
        stack.enter_context(mock.patch.object(bot_main, "ExchangeClient", lambda cfg, log: exchange))
        stack.enter_context(mock.patch.object(bot_main, "log_event", counting_log_event))
        bot = bot_main.TradingBot(sim, str(DEFAULT_SETTINGS))
        logger.setLevel(logging.CRITICAL)
        bot.sentiment = _NeutralSentiment()
        bot.funding_rate = _NeutralFunding()
        bot.notifier = _SilentNotifier()
        bot._stale_or_clock_drift_detected = lambda symbol, ts: (False, "ok")
        bot._record_balance = lambda equity: None
        bot._log_db = lambda *a, **k: None
        bot._save_state = lambda: None

        original_open = bot.portfolio.open_position

        def counting_open(position: Any) -> None:
            nonlocal entries
            entries += 1
            original_open(position)

        bot.portfolio.open_position = counting_open

        needed = bot._needed_timeframes(tf)
        ohlc = {sym: sd.frames[tf][["open", "high", "low", "close"]].to_numpy(dtype=float) for sym, sd in data.items()}
        bot.risk.start_day(bot._get_total_equity())
        started = time.monotonic()

        for step, t_ns in enumerate(timeline):
            now_ns = int(t_ns) + tf_ns
            SimClock.now_value = pd.Timestamp(now_ns, tz="UTC").to_pydatetime()
            equity = bot._get_total_equity()
            bot.risk.roll_periods(equity)

            for symbol, sd in data.items():
                base_ts = sd.ts_ns[tf]
                i = int(np.searchsorted(base_ts, t_ns))
                if i >= base_ts.size or base_ts[i] != t_ns:
                    continue
                o, h, l, c = (float(x) for x in ohlc[symbol][i])

                # Funding: bu mumun içinde bir funding anı geçtiyse açık pozisyondan tahsil et
                pos = bot.portfolio.positions.get(symbol)
                if pos is not None and futures and sd.funding_ns.size:
                    a = int(np.searchsorted(sd.funding_ns, int(t_ns), side="right"))
                    b = int(np.searchsorted(sd.funding_ns, now_ns, side="right"))
                    for rate in sd.funding_rates[a:b]:
                        cost = pos.qty * c * float(rate) * (1 if pos.side.upper() == "BUY" else -1)
                        bot.paper_cash -= cost
                        funding_paid += cost

                # Mum içi koruyucu seviyeler: önce aleyhte uç, sonra lehte uç
                pos = bot.portfolio.positions.get(symbol)
                if pos is not None:
                    is_long = pos.side.upper() == "BUY"
                    adverse = None
                    if is_long and l <= pos.stop_loss:
                        adverse = o if o <= pos.stop_loss else pos.stop_loss
                    elif (not is_long) and h >= pos.stop_loss:
                        adverse = o if o >= pos.stop_loss else pos.stop_loss
                    if pos.liquidation_price > 0 and adverse is None:
                        if (is_long and l <= pos.liquidation_price) or ((not is_long) and h >= pos.liquidation_price):
                            adverse = pos.liquidation_price
                    exchange.prices[symbol] = adverse if adverse is not None else c
                    closed = adverse is not None and bot._check_protective_levels(symbol, adverse)
                    if not closed and bot.portfolio.is_open(symbol):
                        favorable = h if is_long else l
                        exchange.prices[symbol] = favorable
                        bot._check_protective_levels(symbol, favorable)

                exchange.prices[symbol] = c
                frames = {}
                for ftf in needed:
                    idx = int(np.searchsorted(sd.close_ns[ftf], now_ns, side="right"))
                    frames[ftf] = sd.frames[ftf].iloc[max(0, idx - HTF_WINDOW): idx]
                try:
                    bot._process_symbol(symbol, tf, equity, frames=frames)
                except Exception as exc:  # canlıda döngü hatası olarak loglanır
                    key = str(exc).split("(")[0][:80]
                    errors[key] = errors.get(key, 0) + 1
                if bot._stop_requested:  # piyasa devre kesicisi: ertesi güne kadar bekle
                    bot._stop_requested = False

            equity = bot._get_total_equity()
            bot.risk.check_circuit_breaker(equity)
            if bot.risk.circuit_breaker_triggered:
                breaker_candles += 1
            equity_rows.append((now_ns, equity))
            if progress and step % 2000 == 0:
                print(f"    {label}: {step}/{timeline.size} mum, varlık {equity:,.0f}, {time.monotonic() - started:.0f} sn")

        # Açık pozisyonları son fiyattan kapat (sonuç karşılaştırılabilir olsun)
        for symbol in list(bot.portfolio.positions):
            pos = bot.portfolio.positions[symbol]
            price = exchange.prices[symbol]
            fee_paid = bot._paper_exit_fee(symbol, pos.qty * price, maker=False)
            trade = bot.portfolio.close_position(symbol, price, fee_paid=fee_paid)
            if trade:
                trade["strategy_name"] = f"{trade['strategy_name']}_end_of_test"
                bot._apply_paper_close_cash(trade, fee_paid)
        final_equity = bot._get_total_equity()
        trades = pd.DataFrame(bot.portfolio.trade_history)
        initial = float(sim["app"].get("initial_paper_balance", 10_000))
        logger.setLevel(old_level)

    equity_df = pd.DataFrame(equity_rows, columns=["t_ns", "equity"])
    equity_df["timestamp"] = pd.to_datetime(equity_df["t_ns"], utc=True)
    equity_df.loc[len(equity_df)] = [equity_df["t_ns"].iloc[-1], final_equity, equity_df["timestamp"].iloc[-1]]
    pnl = trades["pnl"].astype(float) if not trades.empty else pd.Series(dtype=float)
    wins, losses = pnl[pnl > 0], pnl[pnl < 0]
    fees = (trades.get("fee_paid", pd.Series(dtype=float)).astype(float).sum()
            + trades.get("entry_fee", pd.Series(dtype=float)).astype(float).sum()) if not trades.empty else 0.0
    days = (timeline[-1] - timeline[0]) / 86_400e9 or 1.0
    metrics = {
        "timeframe": tf,
        "leverage": float(sim.get("leverage", {}).get("leverage", 1.0)) if futures else 1.0,
        "days": round(days, 1),
        "final_equity": final_equity,
        "return_pct": final_equity / initial - 1,
        "max_drawdown": max_drawdown(equity_df["equity"]),
        "entries": entries,
        "exits": int(len(pnl)),
        "win_rate": float(len(wins) / len(pnl)) if len(pnl) else 0.0,
        "profit_factor": float(wins.sum() / abs(losses.sum())) if len(losses) and losses.sum() != 0 else (float("inf") if len(wins) else 0.0),
        "fees": float(fees),
        "funding": float(funding_paid),
        "liquidations": int(trades.get("liquidated", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()) if not trades.empty else 0,
        "breaker_candles": breaker_candles,
        "errors": int(sum(errors.values())),
    }
    return ReplayResult(label=label or f"{tf} {metrics['leverage']:g}x", metrics=metrics, equity=equity_df,
                        trades=trades, errors=errors, funnel=funnel)


# ── CLI ──────────────────────────────────────────────────────────────────────
def _load_data(args: argparse.Namespace, settings: dict[str, Any], timeframes: list[str]) -> dict[str, SymbolData]:
    futures = args.market == "futures"
    base_tf = args.base_timeframe
    warmup_days = max(10, args.warmup_days)
    out: dict[str, SymbolData] = {}
    for symbol in args.symbols.split(","):
        symbol = symbol.strip()
        ex_symbol = f"{symbol}:USDT" if futures and ":" not in symbol else symbol
        print(f"  veri: {ex_symbol} {base_tf} {args.days + warmup_days} gün…")
        df = load_or_download(args.exchange, ex_symbol, base_tf, days=args.days + warmup_days, cache_dir=args.cache_dir)
        if df.empty:
            print(f"  ! {symbol} için veri yok, atlanıyor")
            continue
        needed: set[str] = set(timeframes) | {base_tf}
        for tf in timeframes:
            rs = retime_settings(settings, tf)
            needed.add(rs["trading"]["higher_timeframe"])
            for p in rs["strategy"]["params"].values():
                needed.update(x for x in (p.get("primary_timeframe"), p.get("confirm_timeframe")) if x)
        sd = prepare_symbol_data(df, sorted(needed, key=timeframe_to_ms), base_tf)
        if futures and not args.no_funding:
            sd.set_funding(load_funding_history(args.exchange, ex_symbol, args.days + warmup_days, args.cache_dir))
        out[symbol] = sd
    return out


def _print_table(results: list[ReplayResult]) -> None:
    rows = []
    for r in results:
        m = r.metrics
        rows.append({
            "senaryo": r.label,
            "getiri %": round(m["return_pct"] * 100, 2),
            "max DD %": round(m["max_drawdown"] * 100, 2),
            "işlem": m["entries"],
            "kazanç %": round(m["win_rate"] * 100, 1),
            "PF": round(m["profit_factor"], 2) if m["profit_factor"] != float("inf") else "inf",
            "komisyon $": round(m["fees"], 2),
            "funding $": round(m["funding"], 2),
            "likidasyon": m["liquidations"],
            "hata": m["errors"],
        })
    print(pd.DataFrame(rows).to_string(index=False))


def main(argv: list[str] | None = None) -> None:
    import yaml

    p = argparse.ArgumentParser(description="Botu geçmiş veride mum mum çalıştırır (replay backtest).")
    p.add_argument("--settings", default=str(DEFAULT_SETTINGS))
    p.add_argument("--exchange", default="binance")
    p.add_argument("--symbols", default=None, help="varsayılan: settings.yaml'daki semboller")
    p.add_argument("--days", type=int, default=30, help="test süresi (gün)")
    p.add_argument("--warmup-days", type=int, default=10, help="göstergeler için ön veri (gün)")
    p.add_argument("--timeframes", default=None, help="virgüllü, örn. 1m,5m,15m (varsayılan: settings)")
    p.add_argument("--leverages", default=None, help="virgüllü, örn. 1,3 (varsayılan: settings)")
    p.add_argument("--market", choices=["futures", "spot"], default="futures")
    p.add_argument("--base-timeframe", default="1m", help="indirilecek en küçük zaman dilimi")
    p.add_argument("--cache-dir", default="runtime/data")
    p.add_argument("--out", default="runtime/reports")
    p.add_argument("--no-funding", action="store_true")
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args(argv)

    settings = yaml.safe_load(Path(args.settings).read_text(encoding="utf-8"))
    args.symbols = args.symbols or ",".join(settings["trading"]["symbols"])
    timeframes = [t.strip() for t in (args.timeframes or settings["trading"]["timeframe"]).split(",")]
    if min(timeframe_to_ms(t) for t in timeframes) < timeframe_to_ms(args.base_timeframe):
        p.error("--base-timeframe test edilen en küçük zaman diliminden büyük olamaz")
    lev_cfg = settings.get("leverage", {}) or {}
    default_lev = str(lev_cfg.get("leverage", 1)) if lev_cfg.get("enabled") else "1"
    leverages = [float(x) for x in (args.leverages or default_lev).split(",")]

    print(f"Replay backtest: {args.symbols} | {args.days} gün | tf={timeframes} | kaldıraç={leverages}")
    data = _load_data(args, settings, timeframes)
    if not data:
        raise SystemExit("Veri yok")
    end = max(sd.ts_ns[args.base_timeframe][-1] for sd in data.values())
    start = pd.Timestamp(end, tz="UTC") - pd.Timedelta(days=args.days)

    results = []
    for tf in timeframes:
        for lev in leverages:
            label = f"{tf} {lev:g}x"
            print(f"  ▶ {label} çalışıyor…")
            t0 = time.monotonic()
            res = run_replay(retime_settings(settings, tf), data, start, leverage=lev, label=label, seed=args.seed, progress=True)
            print(f"    bitti: {time.monotonic() - t0:.0f} sn, getiri %{res.metrics['return_pct'] * 100:.2f}")
            if res.errors:
                for msg, n in sorted(res.errors.items(), key=lambda kv: -kv[1])[:3]:
                    print(f"    ! {n}× {msg}")
            results.append(res)

    print()
    _print_table(results)
    for r in results:
        print(f"\nKarar hunisi ({r.label}):")
        for key, n in sorted(r.funnel.items(), key=lambda kv: -kv[1])[:20]:
            print(f"  {n:>7}  {key}")
    print("\nNot: sentiment, order book dengesizliği ve anlık funding/long-short filtreleri geçmiş veriyle "
          "test edilemez; bu backtest'te nötrdür. Funding ücretleri geçmiş oranlarla tahsil edilmiştir.")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + "_".join(r.label.replace(" ", "") for r in results)
    pd.DataFrame([{"scenario": r.label, **r.metrics} for r in results]).to_csv(out / f"replay_{stamp}.csv", index=False)
    pd.DataFrame([{"scenario": r.label, "stage": k, "count": v} for r in results for k, v in r.funnel.items()]) \
        .to_csv(out / f"replay_{stamp}_funnel.csv", index=False)
    for r in results:
        if not r.trades.empty:
            r.trades.to_csv(out / f"replay_{stamp}_{r.label.replace(' ', '_')}_trades.csv", index=False)
    print(f"Rapor: {out / f'replay_{stamp}.csv'}")


if __name__ == "__main__":
    main()
