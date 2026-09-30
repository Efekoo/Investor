from __future__ import annotations

"""Dashboard için saf hesaplamalar (Streamlit'ten bağımsız, test edilebilir).

state.json → pozisyon satırları, hesap özeti ve kapanmış işlem istatistikleri.
Muhasebe botla aynıdır: kaldıraçlı pozisyon (margin > 0) için varlık =
teminat + gerçekleşmemiş PnL; spot pozisyonda long = qty*fiyat, short = -qty*fiyat.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass
class PositionView:
    symbol: str
    side: str            # LONG | SHORT
    leverage: float
    entry_price: float
    mark_price: float
    qty: float
    notional: float
    margin: float
    unrealized_pnl: float
    roe_pct: float       # PnL / teminat (spotta PnL / giriş notional'ı)
    stop_loss: float
    take_profit: float
    liquidation_price: float
    liq_distance_pct: float | None  # Güncel fiyattan likidasyona kalan mesafe (%); None → likidasyon yok
    strategy: str
    opened_at: str


@dataclass
class AccountSummary:
    cash: float
    equity: float
    margin_used: float
    margin_usage_pct: float
    unrealized_pnl: float
    open_positions: int


def parse_positions(state: dict[str, Any]) -> list[dict[str, Any]]:
    raw = (state.get("portfolio") or {}).get("positions", [])
    if isinstance(raw, dict):
        raw = list(raw.values())
    return [p for p in raw if isinstance(p, dict) and "symbol" in p]


def position_view(pos: dict[str, Any], mark_price: float | None) -> PositionView:
    entry = float(pos.get("entry_price", 0.0))
    mark = float(mark_price) if mark_price else entry
    qty = float(pos.get("qty", 0.0))
    is_long = str(pos.get("side", "BUY")).upper() == "BUY"
    margin = float(pos.get("margin", 0.0))
    leverage = float(pos.get("leverage", 1.0))
    liq = float(pos.get("liquidation_price", 0.0))

    pnl = (mark - entry) * qty * (1 if is_long else -1)
    base = margin if margin > 0 else entry * qty
    liq_distance = None
    if liq > 0 and mark > 0:
        liq_distance = ((mark - liq) / mark if is_long else (liq - mark) / mark) * 100

    return PositionView(
        symbol=str(pos["symbol"]),
        side="LONG" if is_long else "SHORT",
        leverage=leverage,
        entry_price=entry,
        mark_price=mark,
        qty=qty,
        notional=qty * mark,
        margin=margin,
        unrealized_pnl=pnl,
        roe_pct=pnl / base * 100 if base > 0 else 0.0,
        stop_loss=float(pos.get("stop_loss", 0.0)),
        take_profit=float(pos.get("take_profit", 0.0)),
        liquidation_price=liq,
        liq_distance_pct=liq_distance,
        strategy=str(pos.get("strategy_name", "-")),
        opened_at=str(pos.get("opened_at", "")),
    )


def position_views(state: dict[str, Any]) -> list[PositionView]:
    prices = state.get("last_prices", {}) or {}
    return [position_view(p, prices.get(p["symbol"])) for p in parse_positions(state)]


def account_summary(state: dict[str, Any], views: list[PositionView] | None = None) -> AccountSummary:
    views = position_views(state) if views is None else views
    cash = float(state.get("paper_cash", 0.0))
    equity = cash
    for v in views:
        if v.margin > 0:
            equity += v.margin + v.unrealized_pnl
        else:
            equity += v.qty * v.mark_price if v.side == "LONG" else -v.qty * v.mark_price
    margin_used = sum(v.margin for v in views)
    return AccountSummary(
        cash=cash,
        equity=equity,
        margin_used=margin_used,
        margin_usage_pct=margin_used / equity * 100 if equity > 0 else 0.0,
        unrealized_pnl=sum(v.unrealized_pnl for v in views),
        open_positions=len(views),
    )


def liquidation_alerts(views: list[PositionView], threshold_pct: float) -> list[PositionView]:
    """Likidasyona threshold_pct'ten daha yakın pozisyonlar (en yakın önce)."""
    risky = [v for v in views if v.liq_distance_pct is not None and v.liq_distance_pct <= threshold_pct]
    return sorted(risky, key=lambda v: v.liq_distance_pct)


def closed_trades(state: dict[str, Any]) -> list[dict[str, Any]]:
    trades = (state.get("portfolio") or {}).get("trade_history", []) or []
    return sorted(trades, key=lambda t: str(t.get("closed_at", "")), reverse=True)


def trade_stats(trades: list[dict[str, Any]]) -> dict[str, float]:
    pnls = [float(t.get("pnl", 0.0)) for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    return {
        "count": len(pnls),
        "total_pnl": sum(pnls),
        "win_rate": len(wins) / len(pnls) * 100 if pnls else 0.0,
        "profit_factor": sum(wins) / abs(sum(losses)) if losses else (float("inf") if wins else 0.0),
        "liquidations": sum(1 for t in trades if t.get("liquidated")),
    }


def state_age_seconds(state: dict[str, Any], now: datetime | None = None) -> float | None:
    raw = state.get("updated_at")
    if not raw:
        return None
    try:
        updated = datetime.fromisoformat(str(raw))
    except ValueError:
        return None
    if updated.tzinfo is None:
        updated = updated.replace(tzinfo=timezone.utc)
    now = now or datetime.now(timezone.utc)
    return (now - updated).total_seconds()
