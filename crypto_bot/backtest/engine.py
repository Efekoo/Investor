from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from crypto_bot.core.leverage import clamp_stop_to_liquidation, liquidation_price

from crypto_bot.backtest.metrics import max_drawdown, profit_factor, sharpe_ratio, win_rate

_TF_MINUTES = {"m": 1, "h": 60, "d": 1440, "w": 10080}


def _periods_per_year(timeframe: str | None) -> int:
    """Annualization factor for Sharpe based on candle timeframe."""
    if not timeframe:
        return 365 * 24 * 60  # legacy default (1m)
    minutes = int(timeframe[:-1]) * _TF_MINUTES[timeframe[-1]]
    return int(365 * 24 * 60 / minutes)


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    equity_curve: pd.DataFrame
    metrics: dict


class BacktestEngine:
    """Event-driven, long-only backtest engine.

    Realism features:
      - taker fees + dynamic slippage (volatility & volume scaled)
      - execution delay in candles (signal on close -> fill on next candle)
      - intra-candle stop-loss / take-profit / trailing-stop using high/low
        (conservative: if SL and TP hit in the same candle, SL is assumed first)
      - risk-based position sizing instead of all-in (optional)
      - optional higher-timeframe data feed for multi-timeframe strategies,
        sliced without lookahead (only fully closed HTF candles are visible)
      - optional entry filter (e.g. bull-regime): blocks new longs on candles
        where the filter is False; exits keep working so positions can close
      - optional leverage (isolated margin): only margin is locked, SL is pulled
        inside the liquidation price, and a candle whose low crosses the
        liquidation price wipes out the whole margin
    """

    def __init__(
        self,
        strategy,
        initial_balance: float,
        maker_fee_pct: float = 0.0002,
        taker_fee_pct: float = 0.0006,
        slippage_pct: float = 0.001,
        execution_delay_candles: int = 1,
        stop_loss_pct: float | None = None,
        take_profit_pct: float | None = None,
        trailing_stop_pct: float | None = None,
        risk_per_trade: float | None = None,
        timeframe: str | None = None,
        max_window: int = 500,
        warmup: int = 60,
        leverage: float = 1.0,
        maintenance_margin_rate: float = 0.005,
        liquidation_buffer: float = 0.5,
    ):
        self.strategy = strategy
        self.initial_balance = initial_balance
        self.maker_fee_pct = maker_fee_pct
        self.taker_fee_pct = taker_fee_pct
        self.slippage_pct = slippage_pct
        self.execution_delay_candles = max(0, execution_delay_candles)
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.trailing_stop_pct = trailing_stop_pct
        self.risk_per_trade = risk_per_trade
        self.timeframe = timeframe
        self.max_window = max_window
        self.warmup = warmup
        self.leverage = max(1.0, leverage)
        self.maintenance_margin_rate = maintenance_margin_rate
        self.liquidation_buffer = liquidation_buffer

    def _context(self, data: pd.DataFrame, idx: int) -> tuple[float, float]:
        start = max(0, idx - 30)
        window = data.iloc[start : idx + 1]
        returns = window["close"].astype(float).pct_change().dropna()
        volatility = float(returns.std(ddof=0)) if not returns.empty else 0.0
        volume = float(window["volume"].astype(float).iloc[-1]) if not window.empty else 1.0
        return volatility, volume

    def _dynamic_slippage(self, volatility: float, volume: float) -> float:
        return self.slippage_pct * (1 + 5 * volatility + (50 / max(volume, 1.0)))

    def _position_qty(self, cash: float, buy_price: float, stop_price: float | None) -> float:
        """Risk-based sizing when configured; otherwise all-in (legacy)."""
        if buy_price <= 0:
            return 0.0
        # Kaldıraçta yalnızca teminat (notional / lev) + fee nakitten düşer
        max_affordable = cash / (buy_price * (1 / self.leverage + self.taker_fee_pct))
        if self.risk_per_trade is None or stop_price is None or stop_price >= buy_price:
            return max_affordable
        per_unit_risk = buy_price - stop_price
        qty = (cash * self.risk_per_trade) / per_unit_risk
        return min(qty, max_affordable)

    def run(
        self,
        data: pd.DataFrame,
        htf_data: pd.DataFrame | None = None,
        primary_tf: str | None = None,
        confirm_tf: str | None = None,
        htf_duration: pd.Timedelta | None = None,
        entry_filter: pd.Series | None = None,
    ) -> BacktestResult:
        cash = self.initial_balance
        qty = 0.0
        entry_price = 0.0
        entry_fee = 0.0
        margin = 0.0
        liq_price = 0.0
        liquidations = 0
        stop_price: float | None = None
        take_price: float | None = None
        trades: list[dict] = []
        equity: list[dict] = []
        pending_signal: tuple[str, int] | None = None
        fees_paid = 0.0
        candles_in_market = 0

        use_mtf = htf_data is not None and primary_tf and confirm_tf
        if use_mtf and htf_duration is None:
            deltas = htf_data["timestamp"].diff().dropna()
            htf_duration = deltas.mode().iloc[0] if not deltas.empty else pd.Timedelta(hours=1)

        n = len(data)
        timestamps = data["timestamp"]

        leveraged = self.leverage > 1.0

        def close_position(i: int, exit_price: float, reason: str) -> None:
            nonlocal cash, qty, entry_price, entry_fee, stop_price, take_price, fees_paid, margin, liq_price
            gross = qty * exit_price
            if reason == "liquidation":
                # Isolated margin: teminatın tamamı kaybedilir, nakde bir şey dönmez
                exit_fee = 0.0
                pnl = -(margin + entry_fee)
            else:
                exit_fee = gross * self.taker_fee_pct
                pnl = (exit_price - entry_price) * qty - entry_fee - exit_fee
                if leveraged:
                    cash += margin + (exit_price - entry_price) * qty - exit_fee
                else:
                    cash += gross - exit_fee
            fees_paid += exit_fee
            trades.append(
                {
                    "timestamp": data.iloc[i]["timestamp"],
                    "action": "SELL",
                    "price": exit_price,
                    "qty": qty,
                    "fee": exit_fee,
                    "pnl": pnl,
                    "reason": reason,
                }
            )
            qty = 0.0
            entry_price = 0.0
            entry_fee = 0.0
            margin = 0.0
            liq_price = 0.0
            stop_price = None
            take_price = None

        for i in range(self.warmup, n):
            row = data.iloc[i]
            high = float(row["high"])
            low = float(row["low"])
            close = float(row["close"])

            # --- 1) protective exits first (intra-candle, conservative) ---
            if qty > 0.0:
                candles_in_market += 1
                if self.trailing_stop_pct is not None:
                    candidate = high * (1 - self.trailing_stop_pct)
                    stop_price = max(stop_price or 0.0, candidate) if stop_price else candidate
                if liq_price > 0 and low <= liq_price:
                    liquidations += 1
                    close_position(i, liq_price, "liquidation")
                elif stop_price is not None and low <= stop_price:
                    close_position(i, stop_price, "stop_loss")
                elif take_price is not None and high >= take_price:
                    close_position(i, take_price, "take_profit")

            # --- 2) strategy signal on closed candle ---
            w_start = max(0, i + 1 - self.max_window)
            window = data.iloc[w_start : i + 1]
            if use_mtf:
                cutoff = timestamps.iloc[i] - htf_duration
                htf_visible = htf_data[htf_data["timestamp"] <= cutoff]
                signal = self.strategy.generate_signal({primary_tf: window, confirm_tf: htf_visible})
            else:
                signal = self.strategy.generate_signal(window)

            if signal in {"BUY", "SELL"}:
                pending_signal = (signal, i + self.execution_delay_candles)

            # --- 3) delayed execution ---
            if pending_signal and i >= pending_signal[1]:
                action = pending_signal[0]
                pending_signal = None
                volatility, volume = self._context(data, i)
                slip = self._dynamic_slippage(volatility, volume)

                entry_allowed = entry_filter is None or bool(entry_filter.iloc[i])
                if action == "BUY" and qty == 0.0 and entry_allowed:
                    buy_price = close * (1 + slip)
                    sl = buy_price * (1 - self.stop_loss_pct) if self.stop_loss_pct else None
                    if leveraged and sl is not None:
                        sl = clamp_stop_to_liquidation(
                            "BUY", buy_price, sl, self.leverage,
                            self.maintenance_margin_rate, self.liquidation_buffer,
                        )
                    tp = buy_price * (1 + self.take_profit_pct) if self.take_profit_pct else None
                    new_qty = self._position_qty(cash, buy_price, sl)
                    notional = new_qty * buy_price
                    fee = notional * self.taker_fee_pct
                    locked = notional / self.leverage
                    if new_qty > 0 and locked + fee <= cash + 1e-9:
                        cash -= locked + fee
                        margin = locked if leveraged else 0.0
                        liq_price = liquidation_price("BUY", buy_price, self.leverage, self.maintenance_margin_rate)
                        qty = new_qty
                        entry_price = buy_price
                        entry_fee = fee
                        fees_paid += fee
                        stop_price, take_price = sl, tp
                        trades.append(
                            {
                                "timestamp": row["timestamp"],
                                "action": "BUY",
                                "price": buy_price,
                                "qty": qty,
                                "fee": fee,
                                "pnl": 0.0,
                                "reason": "signal",
                            }
                        )
                elif action == "SELL" and qty > 0.0:
                    close_position(i, close * (1 - slip), "signal")

            if leveraged:
                mark_equity = cash + margin + (close - entry_price) * qty if qty > 0 else cash
            else:
                mark_equity = cash + qty * close
            equity.append({"timestamp": row["timestamp"], "equity": mark_equity})

        trades_df = pd.DataFrame(trades)
        sell_trades = (
            trades_df[trades_df["action"] == "SELL"].copy()
            if not trades_df.empty
            else pd.DataFrame(columns=["pnl"])
        )
        equity_df = pd.DataFrame(equity)
        ret = equity_df["equity"].pct_change().dropna() if not equity_df.empty else pd.Series(dtype=float)

        final_equity = float(equity_df["equity"].iloc[-1]) if not equity_df.empty else self.initial_balance
        first_close = float(data.iloc[self.warmup]["close"]) if n > self.warmup else 0.0
        last_close = float(data.iloc[-1]["close"]) if n else 0.0
        buy_hold = (last_close / first_close - 1) if first_close > 0 else 0.0
        candles_total = max(1, n - self.warmup)

        metrics = {
            "final_equity": final_equity,
            "total_return_pct": final_equity / self.initial_balance - 1,
            "buy_hold_return_pct": buy_hold,
            "win_rate": win_rate(sell_trades) if not sell_trades.empty else 0.0,
            "sharpe_ratio": sharpe_ratio(ret, periods_per_year=_periods_per_year(self.timeframe)),
            "max_drawdown": max_drawdown(equity_df["equity"]) if not equity_df.empty else 0.0,
            "profit_factor": profit_factor(sell_trades) if not sell_trades.empty else 0.0,
            "total_trades": int(len(sell_trades)),
            "fees_paid": fees_paid,
            "avg_trade_pnl": float(sell_trades["pnl"].mean()) if not sell_trades.empty else 0.0,
            "exposure_pct": candles_in_market / candles_total,
            "leverage": self.leverage,
            "liquidations": liquidations,
        }
        return BacktestResult(trades=sell_trades, equity_curve=equity_df, metrics=metrics)
