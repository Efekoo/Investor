from __future__ import annotations

import pandas as pd

from crypto_bot.strategies.base_strategy import Strategy, StrategyDecision


class EMACrossoverStrategy(Strategy):
    def __init__(self, params: dict) -> None:
        super().__init__("ema", params)
        self.supported_regimes = ["TREND", "VOLATILE"]

    def _extract(self, data: pd.DataFrame | dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
        if isinstance(data, dict):
            primary_tf = self.params.get("primary_timeframe", "1m")
            confirm_tf = self.params.get("confirm_timeframe", primary_tf)
            primary = data[primary_tf] if primary_tf in data else next(iter(data.values()))
            confirm = data[confirm_tf] if confirm_tf in data else primary
            return primary, confirm
        return data, data

    def generate_signal(self, data: pd.DataFrame | dict[str, pd.DataFrame]) -> str:
        return self.generate_decision(data).signal

    def generate_decision(self, data: pd.DataFrame | dict[str, pd.DataFrame]) -> StrategyDecision:
        primary, confirm = self._extract(data)
        if primary.empty or len(primary) < max(self.params["fast_period"], self.params["slow_period"]) + 2:
            return StrategyDecision(signal="HOLD", reason="insufficient_data", metadata={})

        close = primary["close"].astype(float)
        fast = close.ewm(span=self.params["fast_period"], adjust=False).mean()
        slow = close.ewm(span=self.params["slow_period"], adjust=False).mean()

        crossed_up = fast.iloc[-2] <= slow.iloc[-2] and fast.iloc[-1] > slow.iloc[-1]
        crossed_down = fast.iloc[-2] >= slow.iloc[-2] and fast.iloc[-1] < slow.iloc[-1]

        # HTF trend filtresi
        htf_bullish = True
        htf_bearish = True
        if len(confirm) >= 52:
            close_c = confirm["close"].astype(float)
            htf_fast = close_c.ewm(span=20, adjust=False).mean().iloc[-1]
            htf_slow = close_c.ewm(span=50, adjust=False).mean().iloc[-1]
            htf_bullish = htf_fast > htf_slow
            htf_bearish = htf_fast < htf_slow

        if crossed_up and htf_bullish:
            signal = "BUY"
            reason = "ema_crossover_up_mtf_confirmed"
        elif crossed_down and htf_bearish:
            signal = "SELL"
            reason = "ema_crossover_down_mtf_confirmed"
        elif crossed_up and not htf_bullish:
            signal = "HOLD"
            reason = "ema_crossover_up_htf_blocked"
        elif crossed_down and not htf_bearish:
            signal = "HOLD"
            reason = "ema_crossover_down_htf_blocked"
        else:
            signal = "HOLD"
            reason = "no_crossover"

        return StrategyDecision(
            signal=signal,
            reason=reason,
            metadata={
                "fast_ema": float(fast.iloc[-1]),
                "slow_ema": float(slow.iloc[-1]),
                "trend_strength": abs(float(fast.iloc[-1] - slow.iloc[-1])) / max(abs(float(slow.iloc[-1])), 1e-9),
                "htf_bullish": htf_bullish,
            },
        )
