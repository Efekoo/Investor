from __future__ import annotations

import pandas as pd

from crypto_bot.strategies.base_strategy import Strategy, StrategyDecision


class EmaRsiVolumeStrategy(Strategy):
    """
    EMA kesişimi + Stochastic RSI + hacim onayı + MTF filtresi.

    Plain RSI yerine Stochastic RSI kullanır:
    - Daha hızlı overbought/oversold tespiti
    - Daha az gecikme (RSI'dan RSI hesaplar, trend değişimlerini daha erken yakalar)

    Stochastic RSI bantları:
      %K < stoch_lower → Aşırı Satım (BUY için uygun)
      %K > stoch_upper → Aşırı Alım (SELL için uygun)
    """

    def __init__(self, params: dict) -> None:
        super().__init__("ema_rsi_volume", params)
        self.supported_regimes = ["TREND", "VOLATILE"]

    @staticmethod
    def _stoch_rsi(
        close: pd.Series,
        rsi_period: int = 14,
        stoch_period: int = 14,
        smooth_k: int = 3,
        smooth_d: int = 3,
    ) -> tuple[pd.Series, pd.Series]:
        """
        Stochastic RSI hesaplar.

        Adımlar:
          1. RSI'yı hesapla
          2. RSI üzerinde Stochastic uygula → ham %K
          3. ham %K'yı smooth_k periyodunda yumuşat → %K
          4. %K'yı smooth_d periyodunda yumuşat → %D

        Dönüş: (%K serisi, %D serisi)  — her ikisi de 0-100 aralığında
        """
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(rsi_period).mean()
        loss = (-delta.clip(upper=0)).rolling(rsi_period).mean()
        rs = gain / loss.replace(0, 1e-9)
        rsi = 100.0 - (100.0 / (1.0 + rs))

        rsi_min = rsi.rolling(stoch_period).min()
        rsi_max = rsi.rolling(stoch_period).max()
        stoch_k_raw = 100.0 * (rsi - rsi_min) / (rsi_max - rsi_min).replace(0, 1e-9)

        stoch_k = stoch_k_raw.rolling(smooth_k).mean()
        stoch_d = stoch_k.rolling(smooth_d).mean()
        return stoch_k, stoch_d

    def _extract(self, data: pd.DataFrame | dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
        if isinstance(data, dict):
            primary_tf = self.params.get("primary_timeframe", "1m")
            confirm_tf = self.params.get("confirm_timeframe", primary_tf)
            primary = data[primary_tf] if primary_tf in data else next(iter(data.values()))
            confirm = data[confirm_tf] if confirm_tf in data else primary
            return primary, confirm
        return data, data

    def generate_signal(self, data: pd.DataFrame | dict[str, pd.DataFrame]) -> str:
        decision = self.generate_decision(data)
        return decision.signal

    def generate_decision(self, data: pd.DataFrame | dict[str, pd.DataFrame]) -> StrategyDecision:
        primary, confirm = self._extract(data)

        ema_fast_p = int(self.params["ema_fast"])
        ema_slow_p = int(self.params["ema_slow"])
        rsi_period = int(self.params["rsi_period"])
        stoch_period = int(self.params.get("stoch_period", 14))
        smooth_k = int(self.params.get("smooth_k", 3))
        smooth_d = int(self.params.get("smooth_d", 3))
        stoch_lower = float(self.params.get("stoch_lower", 25))   # Aşırı satım eşiği
        stoch_upper = float(self.params.get("stoch_upper", 75))   # Aşırı alım eşiği

        min_len = max(ema_fast_p, ema_slow_p, rsi_period + stoch_period + smooth_k + smooth_d, 100)
        if len(primary) < min_len or len(confirm) < min_len:
            return StrategyDecision(signal="HOLD", reason="insufficient_data", metadata={})

        # --- Primary Timeframe Analizi ---
        close_p = primary["close"].astype(float)
        ema_fast = close_p.ewm(span=ema_fast_p, adjust=False).mean()
        ema_slow = close_p.ewm(span=ema_slow_p, adjust=False).mean()

        # Stochastic RSI
        stoch_k, stoch_d = self._stoch_rsi(close_p, rsi_period, stoch_period, smooth_k, smooth_d)
        current_k = float(stoch_k.iloc[-1])
        current_d = float(stoch_d.iloc[-1])

        # Hacim onayı
        volume = primary["volume"].astype(float)
        vol_window = int(self.params.get("volume_window", 20))
        min_vol_ratio = float(self.params.get("min_volume_ratio", 1.0))
        vol_ratio = float(
            volume.iloc[-1] / max(volume.rolling(vol_window).mean().iloc[-1], 1e-9)
        )

        # EMA crossover
        crossed_up = (
            ema_fast.iloc[-2] <= ema_slow.iloc[-2]
            and ema_fast.iloc[-1] > ema_slow.iloc[-1]
        )
        crossed_down = (
            ema_fast.iloc[-2] >= ema_slow.iloc[-2]
            and ema_fast.iloc[-1] < ema_slow.iloc[-1]
        )

        # --- HTF Onayı ---
        close_c = confirm["close"].astype(float)
        htf_fast = float(close_c.ewm(span=20, adjust=False).mean().iloc[-1])
        htf_slow = float(close_c.ewm(span=50, adjust=False).mean().iloc[-1])
        htf_trend_bullish = htf_fast > htf_slow
        htf_trend_bearish = htf_fast < htf_slow

        meta = {
            "stoch_k": round(current_k, 2),
            "stoch_d": round(current_d, 2),
            "vol_ratio": round(vol_ratio, 3),
        }

        # BUY: EMA yukarı kestı + StochRSI aşırı satım bölgesinden çıkıyor + hacim + HTF bullish
        if (
            crossed_up
            and htf_trend_bullish
            and current_k < stoch_upper          # Henüz aşırı alım değil
            and current_k > stoch_lower          # Aşırı satım bölgesinden çıkmak üzere (veya çıktı)
            and current_k > current_d            # %K %D'nin üstüne geçti → momentum yukarı
            and vol_ratio >= min_vol_ratio
        ):
            return StrategyDecision(
                signal="BUY",
                reason="ema_up_stochrsi_crossover_volume_mtf",
                metadata={**meta, "htf_trend": "bullish"},
            )

        # SELL: EMA aşağı kestı + StochRSI aşırı alım bölgesinden dönüyor + HTF bearish
        if (
            crossed_down
            and htf_trend_bearish
            and current_k > stoch_lower          # Henüz aşırı satım değil
            and current_k < stoch_upper          # Aşırı alım bölgesinden çıkmak üzere
            and current_k < current_d            # %K %D'nin altına geçti → momentum aşağı
        ):
            return StrategyDecision(
                signal="SELL",
                reason="ema_down_stochrsi_crossover_mtf",
                metadata={**meta, "htf_trend": "bearish"},
            )

        return StrategyDecision(
            signal="HOLD",
            reason="no_mtf_confluence",
            metadata={**meta, "htf_bullish": htf_trend_bullish},
        )
