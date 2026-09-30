from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class Position:
    symbol: str
    side: str
    entry_price: float
    qty: float
    entry_fee: float
    stop_loss: float
    take_profit: float
    peak_price: float
    strategy_name: str = "unknown"
    opened_at: datetime = field(default_factory=utcnow)
    partial_tp_targets: list = field(default_factory=list)  # [{"price": float, "close_pct": float, "hit": bool}]
    leverage: float = 1.0
    margin: float = 0.0             # Kaldıraçlı pozisyonda kilitlenen teminat (0 → spot muhasebesi)
    liquidation_price: float = 0.0  # 0 → likidasyon yok (kaldıraç 1x)


class Portfolio:
    def __init__(self) -> None:
        self.positions: dict[str, Position] = {}
        self.trade_history: list[dict] = []

    def open_position(self, position: Position) -> None:
        self.positions[position.symbol] = position

    def close_position(self, symbol: str, exit_price: float, fee_paid: float = 0.0) -> dict | None:
        pos = self.positions.pop(symbol, None)
        if not pos:
            return None
        pnl = (exit_price - pos.entry_price) * pos.qty
        if pos.side.upper() == "SELL":
            pnl = -pnl
        pnl -= (pos.entry_fee + fee_paid)
        trade = {
            "margin_released": pos.margin,
            "leverage": pos.leverage,
            "symbol": pos.symbol,
            "side": pos.side,
            "qty": pos.qty,
            "entry_price": pos.entry_price,
            "exit_price": exit_price,
            "pnl": pnl,
            "fee_paid": fee_paid,
            "entry_fee": pos.entry_fee,
            "opened_at": pos.opened_at,
            "closed_at": utcnow(),
            "strategy_name": pos.strategy_name,
        }
        self.trade_history.append(trade)
        return trade

    def partial_close_position(self, symbol: str, close_qty: float, exit_price: float, fee_paid: float = 0.0) -> dict | None:
        """Pozisyonun bir kısmını kapatır. qty sıfırlanırsa tam kapatma yapar."""
        pos = self.positions.get(symbol)
        if not pos or pos.qty <= 0:
            return None
        actual_qty = min(close_qty, pos.qty)
        pnl = (exit_price - pos.entry_price) * actual_qty
        if pos.side.upper() == "SELL":
            pnl = -pnl
        pnl -= fee_paid
        margin_released = pos.margin * (actual_qty / pos.qty)
        pos.margin -= margin_released
        pos.qty -= actual_qty
        trade = {
            "margin_released": margin_released,
            "leverage": pos.leverage,
            "symbol": pos.symbol,
            "side": pos.side,
            "qty": actual_qty,
            "entry_price": pos.entry_price,
            "exit_price": exit_price,
            "pnl": pnl,
            "fee_paid": fee_paid,
            "entry_fee": 0.0,
            "opened_at": pos.opened_at,
            "closed_at": utcnow(),
            "strategy_name": pos.strategy_name + "_partial_tp",
        }
        self.trade_history.append(trade)
        if pos.qty <= 1e-10:
            self.positions.pop(symbol, None)
        return trade

    def update_peak(self, symbol: str, price: float) -> None:
        pos = self.positions.get(symbol)
        if not pos:
            return
        if pos.side.upper() == "BUY":
            pos.peak_price = max(pos.peak_price, price)
        else:
            pos.peak_price = min(pos.peak_price, price)

    def is_open(self, symbol: str) -> bool:
        return symbol in self.positions

    def open_count(self) -> int:
        return len(self.positions)

    def unrealized_pnl(self, symbol: str, mark_price: float) -> float:
        pos = self.positions.get(symbol)
        if not pos:
            return 0.0
        pnl = (mark_price - pos.entry_price) * pos.qty
        return -pnl if pos.side.upper() == "SELL" else pnl

    def margin_used(self) -> float:
        return sum(pos.margin for pos in self.positions.values())

    def snapshot(self) -> dict:
        return {
            "positions": [
                {
                    **asdict(pos),
                    "opened_at": pos.opened_at.isoformat(),
                    "partial_tp_targets": pos.partial_tp_targets,
                }
                for pos in self.positions.values()
            ],
            "trade_history": [
                {
                    **trade,
                    "opened_at": trade["opened_at"].isoformat() if hasattr(trade["opened_at"], "isoformat") else trade["opened_at"],
                    "closed_at": trade["closed_at"].isoformat() if hasattr(trade["closed_at"], "isoformat") else trade["closed_at"],
                }
                for trade in self.trade_history[-500:]
            ],
        }

    def restore(self, state: dict) -> None:
        self.positions.clear()
        self.trade_history = state.get("trade_history", [])
        for row in state.get("positions", []):
            self.positions[row["symbol"]] = Position(
                symbol=row["symbol"],
                side=row["side"],
                entry_price=float(row["entry_price"]),
                qty=float(row["qty"]),
                entry_fee=float(row.get("entry_fee", 0.0)),
                stop_loss=float(row["stop_loss"]),
                take_profit=float(row["take_profit"]),
                peak_price=float(row.get("peak_price", row["entry_price"])),
                strategy_name=str(row.get("strategy_name", "unknown")),
                opened_at=datetime.fromisoformat(row["opened_at"]),
                partial_tp_targets=list(row.get("partial_tp_targets", [])),
                leverage=float(row.get("leverage", 1.0)),
                margin=float(row.get("margin", 0.0)),
                liquidation_price=float(row.get("liquidation_price", 0.0)),
            )
