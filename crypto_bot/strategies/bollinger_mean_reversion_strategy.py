from __future__ import annotations
import pandas as pd
from crypto_bot.strategies.base_strategy import Strategy, StrategyDecision

class BollingerMeanReversionStrategy(Strategy):
    def __init__(self, params: dict) -> None:
        super().__init__("bollinger_mean_reversion", params)
        self.supported_regimes = ["RANGE"]

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

        period = self.params.get("period", 20)
        std_dev = self.params.get("std_dev", 2.0)

        if len(primary) < period + 5:
            return StrategyDecision(signal="HOLD", reason="insufficient_data", metadata={})

        close = primary["close"].astype(float)
        ma = close.rolling(period).mean()
        std = close.rolling(period).std()

        upper_band = ma + (std_dev * std)
        lower_band = ma - (std_dev * std)

        current_price = close.iloc[-1]
        prev_price = close.iloc[-2]
        rsi = self._rsi(close, 14).iloc[-1]

        # HTF trend filtresi: Güçlü trend varken mean-reversion riski artar
        htf_bullish = True
        htf_bearish = True
        if len(confirm) >= 52:
            close_c = confirm["close"].astype(float)
            htf_fast = close_c.ewm(span=20, adjust=False).mean().iloc[-1]
            htf_slow = close_c.ewm(span=50, adjust=False).mean().iloc[-1]
            htf_bullish = htf_fast > htf_slow
            htf_bearish = htf_fast < htf_slow

        # AL: Alt bantta geri dönüş + HTF yukarı trend veya yatay (bearish değilse)
        if prev_price < lower_band.iloc[-2] and current_price > lower_band.iloc[-1] and rsi < 45 and not htf_bearish:
            return StrategyDecision(
                signal="BUY",
                reason="bollinger_lower_bounce_mtf_confirmed",
                metadata={"rsi": rsi, "price": current_price, "band": "lower", "htf_bullish": htf_bullish}
            )

        # SAT: Üst bantta geri dönüş + HTF aşağı trend veya yatay (bullish değilse)
        if prev_price > upper_band.iloc[-2] and current_price < upper_band.iloc[-1] and rsi > 55 and not htf_bullish:
            return StrategyDecision(
                signal="SELL",
                reason="bollinger_upper_reversal_mtf_confirmed",
                metadata={"rsi": rsi, "price": current_price, "band": "upper", "htf_bearish": htf_bearish}
            )

        return StrategyDecision(signal="HOLD", reason="within_bands", metadata={"rsi": rsi})

    @staticmethod
    def _rsi(close: pd.Series, period: int) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / loss.replace(0, 1e-9)
        return 100 - (100 / (1 + rs))
