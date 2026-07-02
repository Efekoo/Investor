from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from crypto_bot.backtest.metrics import max_drawdown, profit_factor, sharpe_ratio, win_rate


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    equity_curve: pd.DataFrame
    metrics: dict


class BacktestEngine:
    def __init__(
        self,
        strategy,
        initial_balance: float,
        maker_fee_pct: float = 0.0002,
        taker_fee_pct: float = 0.0006,
        slippage_pct: float = 0.001,
        execution_delay_candles: int = 1,
    ):
        self.strategy = strategy
        self.initial_balance = initial_balance
        self.maker_fee_pct = maker_fee_pct
        self.taker_fee_pct = taker_fee_pct
        self.slippage_pct = slippage_pct
        self.execution_delay_candles = max(0, execution_delay_candles)

    def _context(self, data: pd.DataFrame, idx: int) -> tuple[float, float]:
        start = max(0, idx - 30)
        window = data.iloc[start : idx + 1]
        returns = window["close"].astype(float).pct_change().dropna()
        volatility = float(returns.std(ddof=0)) if not returns.empty else 0.0
        volume = float(window["volume"].astype(float).iloc[-1]) if not window.empty else 1.0
        return volatility, volume

    def run(self, data: pd.DataFrame) -> BacktestResult:
        cash = self.initial_balance
        qty = 0.0
        entry_price = 0.0
        entry_fee = 0.0
        trades = []
        equity = []
        pending_signal: tuple[str, int] | None = None

        for i in range(60, len(data)):
            window = data.iloc[: i + 1]
            signal = self.strategy.generate_signal(window)
            if signal in {"BUY", "SELL"}:
                pending_signal = (signal, i + self.execution_delay_candles)

            if pending_signal and i >= pending_signal[1]:
                action = pending_signal[0]
                pending_signal = None
                close = float(data.iloc[i]["close"])
                volatility, volume = self._context(data, i)
                dynamic_slippage = self.slippage_pct * (1 + 5 * volatility + (50 / max(volume, 1.0)))

                if action == "BUY" and qty == 0.0:
                    buy_price = close * (1 + dynamic_slippage)
                    qty = cash / buy_price if buy_price > 0 else 0.0
                    notional = qty * buy_price
                    entry_fee = notional * self.taker_fee_pct
                    cash -= notional + entry_fee
                    entry_price = buy_price
                    trades.append(
                        {
                            "timestamp": data.iloc[i]["timestamp"],
                            "action": "BUY",
                            "price": buy_price,
                            "qty": qty,
                            "fee": entry_fee,
                            "pnl": 0.0,
                        }
                    )
                elif action == "SELL" and qty > 0.0:
                    sell_price = close * (1 - dynamic_slippage)
                    gross = qty * sell_price
                    exit_fee = gross * self.taker_fee_pct
                    pnl = (sell_price - entry_price) * qty - entry_fee - exit_fee
                    cash += gross - exit_fee
                    trades.append(
                        {
                            "timestamp": data.iloc[i]["timestamp"],
                            "action": "SELL",
                            "price": sell_price,
                            "qty": qty,
                            "fee": exit_fee,
                            "pnl": pnl,
                        }
                    )
                    qty = 0.0
                    entry_price = 0.0
                    entry_fee = 0.0

            mark = float(data.iloc[i]["close"])
            mark_equity = cash + (qty * mark)
            equity.append({"timestamp": data.iloc[i]["timestamp"], "equity": mark_equity})

        trades_df = pd.DataFrame(trades)
        sell_trades = trades_df[trades_df["action"] == "SELL"].copy() if not trades_df.empty else pd.DataFrame(columns=["pnl"])
        equity_df = pd.DataFrame(equity)
        ret = equity_df["equity"].pct_change().dropna() if not equity_df.empty else pd.Series(dtype=float)

        metrics = {
            "final_equity": float(equity_df["equity"].iloc[-1]) if not equity_df.empty else self.initial_balance,
            "win_rate": win_rate(sell_trades) if not sell_trades.empty else 0.0,
            "sharpe_ratio": sharpe_ratio(ret),
            "max_drawdown": max_drawdown(equity_df["equity"]) if not equity_df.empty else 0.0,
            "profit_factor": profit_factor(sell_trades) if not sell_trades.empty else 0.0,
            "total_trades": int(len(sell_trades)),
        }
        return BacktestResult(trades=sell_trades, equity_curve=equity_df, metrics=metrics)
