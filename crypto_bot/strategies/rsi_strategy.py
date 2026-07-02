from __future__ import annotations

import pandas as pd

from crypto_bot.strategies.base_strategy import Strategy, StrategyDecision


class RSIStrategy(Strategy):
    def __init__(self, params: dict) -> None:
        super().__init__("rsi", params)
        self.supported_regimes = ["RANGE"]

    def _extract(self, data: pd.DataFrame | dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
        if isinstance(data, dict):
            primary_tf = self.params.get("primary_timeframe", "1m")
            confirm_tf = self.params.get("confirm_timeframe", primary_tf)
            primary = data[primary_tf] if primary_tf in data else next(iter(data.values()))
            confirm = data[confirm_tf] if confirm_tf in data else primary
            return primary, confirm
        return data, data

    @staticmethod
    def _rsi(close: pd.Series, period: int) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / loss.replace(0, 1e-9)
        return 100 - (100 / (1 + rs))

    def generate_signal(self, data: pd.DataFrame | dict[str, pd.DataFrame]) -> str:
        return self.generate_decision(data).signal

    def generate_decision(self, data: pd.DataFrame | dict[str, pd.DataFrame]) -> StrategyDecision:
        primary, confirm = self._extract(data)
        period = self.params["period"]
        if primary.empty or len(primary) < period + 2:
            return StrategyDecision(signal="HOLD", reason="insufficient_data", metadata={})

        rsi_value = float(self._rsi(primary["close"].astype(float), period).iloc[-1])

        # HTF trend filtresi: Trend çok güçlüyse mean-reversion sinyalini engelle
        htf_bullish = True
        htf_bearish = True
        if len(confirm) >= 52:
            close_c = confirm["close"].astype(float)
            htf_fast = close_c.ewm(span=20, adjust=False).mean().iloc[-1]
            htf_slow = close_c.ewm(span=50, adjust=False).mean().iloc[-1]
            htf_bullish = htf_fast > htf_slow
            htf_bearish = htf_fast < htf_slow

        if rsi_value <= self.params["oversold"] and htf_bullish:
            # HTF yukarı trendde, RSI aşırı satım → alış (ortalamaya dönüş)
            signal = "BUY"
            reason = "rsi_oversold_htf_bullish"
        elif rsi_value >= self.params["overbought"] and htf_bearish:
            # HTF aşağı trendde, RSI aşırı alım → satış
            signal = "SELL"
            reason = "rsi_overbought_htf_bearish"
        elif rsi_value <= self.params["oversold"] and not htf_bullish:
            signal = "HOLD"
            reason = "rsi_oversold_htf_blocked"
        elif rsi_value >= self.params["overbought"] and not htf_bearish:
            signal = "HOLD"
            reason = "rsi_overbought_htf_blocked"
        else:
            signal = "HOLD"
            reason = "rsi_neutral"

        confidence = abs(rsi_value - 50) / 50
        return StrategyDecision(
            signal=signal,
            reason=reason,
            metadata={"rsi": rsi_value, "confidence": confidence, "htf_bullish": htf_bullish},
        )
