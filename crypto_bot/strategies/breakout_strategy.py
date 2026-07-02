from __future__ import annotations
import pandas as pd
import numpy as np
from crypto_bot.strategies.base_strategy import Strategy, StrategyDecision

class VolatilityBreakoutStrategy(Strategy):
    def __init__(self, params: dict) -> None:
        super().__init__("volatility_breakout", params)
        self.supported_regimes = ["RANGE", "TREND"]

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

        lookback = self.params.get("lookback", 20)
        if len(primary) < lookback + 5:
            return StrategyDecision(signal="HOLD", reason="insufficient_data", metadata={})

        close = primary["close"].astype(float)
        high = primary["high"].astype(float)
        low = primary["low"].astype(float)

        upper_barrier = high.iloc[-lookback-1:-1].max()
        lower_barrier = low.iloc[-lookback-1:-1].min()
        current_price = close.iloc[-1]

        # HTF trend filtresi: Yalnızca trend yönünde breakout al
        htf_bullish = True
        htf_bearish = True
        if len(confirm) >= 52:
            close_c = confirm["close"].astype(float)
            htf_fast = close_c.ewm(span=20, adjust=False).mean().iloc[-1]
            htf_slow = close_c.ewm(span=50, adjust=False).mean().iloc[-1]
            htf_bullish = htf_fast > htf_slow
            htf_bearish = htf_fast < htf_slow

        # Yukarı kırılım — sadece HTF yukarı trendde
        if current_price > upper_barrier and htf_bullish:
            return StrategyDecision(
                signal="BUY",
                reason="breakout_above_resistance_mtf_confirmed",
                metadata={"price": current_price, "barrier": upper_barrier, "htf_bullish": True}
            )

        # Aşağı kırılım — sadece HTF aşağı trendde
        if current_price < lower_barrier and htf_bearish:
            return StrategyDecision(
                signal="SELL",
                reason="breakout_below_support_mtf_confirmed",
                metadata={"price": current_price, "barrier": lower_barrier, "htf_bearish": True}
            )

        # HTF ile çelişen kırılımları engelle
        if current_price > upper_barrier and not htf_bullish:
            return StrategyDecision(signal="HOLD", reason="breakout_up_htf_blocked", metadata={})
        if current_price < lower_barrier and not htf_bearish:
            return StrategyDecision(signal="HOLD", reason="breakout_down_htf_blocked", metadata={})

        return StrategyDecision(signal="HOLD", reason="within_range", metadata={})
