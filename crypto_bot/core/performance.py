from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class StrategyStats:
    trades: int = 0
    wins: int = 0
    pnl: float = 0.0
    equity_peak: float = 0.0
    drawdown: float = 0.0
    disabled: bool = False
    disable_reason: str = ""


@dataclass
class PerformanceConfig:
    min_trades_for_eval: int
    min_win_rate: float
    max_drawdown: float
    min_pnl: float
    drawdown_risk_multiplier: float
    high_volatility_risk_multiplier: float
    stable_profit_risk_multiplier: float


class StrategyPerformanceTracker:
    def __init__(self, config: PerformanceConfig) -> None:
        self.config = config
        self.stats: dict[str, StrategyStats] = {}
        self.global_pnl_series: list[float] = []

    def _ensure(self, strategy_name: str) -> StrategyStats:
        if strategy_name not in self.stats:
            self.stats[strategy_name] = StrategyStats()
        return self.stats[strategy_name]

    def record_trade(self, strategy_name: str, pnl: float) -> None:
        s = self._ensure(strategy_name)
        s.trades += 1
        s.wins += 1 if pnl > 0 else 0
        s.pnl += pnl
        s.equity_peak = max(s.equity_peak, s.pnl)
        if s.equity_peak > 0:
            s.drawdown = min(s.drawdown, (s.pnl - s.equity_peak) / s.equity_peak)
        self.global_pnl_series.append(pnl)
        self.global_pnl_series = self.global_pnl_series[-200:]
        self._evaluate_strategy(strategy_name)

    def _evaluate_strategy(self, strategy_name: str) -> None:
        s = self._ensure(strategy_name)
        if s.trades < self.config.min_trades_for_eval:
            return
        win_rate = s.wins / max(s.trades, 1)
        if win_rate < self.config.min_win_rate:
            s.disabled = True
            s.disable_reason = f"win_rate {win_rate:.2f} below {self.config.min_win_rate:.2f}"
            return
        if s.pnl < self.config.min_pnl:
            s.disabled = True
            s.disable_reason = f"pnl {s.pnl:.2f} below {self.config.min_pnl:.2f}"
            return
        if s.drawdown <= -abs(self.config.max_drawdown):
            s.disabled = True
            s.disable_reason = f"drawdown {s.drawdown:.2%} exceeded {self.config.max_drawdown:.2%}"

    def global_win_rate(self) -> tuple[float, int]:
        """Tüm stratejilerin toplamı üzerinden (kazanma_oranı, toplam_işlem) döner."""
        total_trades = sum(s.trades for s in self.stats.values())
        total_wins = sum(s.wins for s in self.stats.values())
        if total_trades <= 0:
            return 0.0, 0
        return total_wins / total_trades, total_trades

    def is_disabled(self, strategy_name: str) -> tuple[bool, str]:
        s = self._ensure(strategy_name)
        return s.disabled, s.disable_reason

    def snapshot(self) -> dict[str, dict]:
        return {
            k: {
                "trades": v.trades,
                "wins": v.wins,
                "pnl": v.pnl,
                "drawdown": v.drawdown,
                "disabled": v.disabled,
                "disable_reason": v.disable_reason,
            }
            for k, v in self.stats.items()
        }

    def restore(self, payload: dict[str, dict]) -> None:
        self.stats = {}
        for name, row in payload.items():
            self.stats[name] = StrategyStats(
                trades=int(row.get("trades", 0)),
                wins=int(row.get("wins", 0)),
                pnl=float(row.get("pnl", 0.0)),
                drawdown=float(row.get("drawdown", 0.0)),
                disabled=bool(row.get("disabled", False)),
                disable_reason=str(row.get("disable_reason", "")),
                equity_peak=max(0.0, float(row.get("pnl", 0.0))),
            )

    def adaptive_risk_multiplier(self, current_volatility: float) -> float:
        mult = 1.0
        global_pnl = sum(self.global_pnl_series[-20:]) if self.global_pnl_series else 0.0
        if global_pnl < 0:
            mult *= self.config.drawdown_risk_multiplier
        if current_volatility > 0.02:
            mult *= self.config.high_volatility_risk_multiplier
        if global_pnl > 0 and current_volatility < 0.01:
            mult *= self.config.stable_profit_risk_multiplier
        return max(0.25, min(mult, 1.25))
