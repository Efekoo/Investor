from __future__ import annotations

import pandas as pd

from crypto_bot.strategies.base_strategy import Strategy, StrategyDecision


class VWAPReversionStrategy(Strategy):
    """
    VWAP bant sapması tabanlı ortalamaya-dönüş stratejisi.

    Hacim ağırlıklı ortalama fiyat (VWAP) ve hacim ağırlıklı standart sapma
    kullanarak dinamik destek/direnç bantları oluşturur.

    Sinyal mantığı:
      - Fiyat VWAP - N*std altına düşerse → BUY (aşırı satım, ortalamaya dönüş beklenir)
      - Fiyat VWAP + N*std üstüne çıkarsa → SELL (aşırı alım, ortalamaya dönüş beklenir)

    Hem RANGE hem TREND rejiminde çalışır (TREND'de onay filtresi ile).
    """

    def __init__(self, params: dict) -> None:
        super().__init__("vwap_reversion", params)
        self.supported_regimes = ["RANGE", "TREND"]

    def _extract(self, data: pd.DataFrame | dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
        if isinstance(data, dict):
            primary_tf = self.params.get("primary_timeframe", "1m")
            confirm_tf = self.params.get("confirm_timeframe", primary_tf)
            primary = data.get(primary_tf, next(iter(data.values())))
            confirm = data.get(confirm_tf, primary)
            return primary, confirm
        return data, data

    @staticmethod
    def _vwap_bands(df: pd.DataFrame, period: int) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Kayan pencere VWAP ve ±1/2 standart sapma bantlarını hesaplar.

        typical_price = (high + low + close) / 3
        vwap = sum(tp * volume, period) / sum(volume, period)
        vwap_std = sqrt(sum(volume * (tp - vwap)^2, period) / sum(volume, period))
        """
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        close = df["close"].astype(float)
        volume = df["volume"].astype(float).replace(0, 1e-9)

        tp = (high + low + close) / 3.0
        tp_vol = tp * volume

        sum_tp_vol = tp_vol.rolling(period).sum()
        sum_vol = volume.rolling(period).sum().replace(0, 1e-9)

        vwap = sum_tp_vol / sum_vol

        # Hacim ağırlıklı standart sapma
        variance = (volume * (tp - vwap) ** 2).rolling(period).sum() / sum_vol
        std = variance.apply(lambda x: x ** 0.5 if x >= 0 else 0.0)

        return vwap, std, tp

    def generate_signal(self, data: pd.DataFrame | dict[str, pd.DataFrame]) -> str:
        return self.generate_decision(data).signal

    def generate_decision(self, data: pd.DataFrame | dict[str, pd.DataFrame]) -> StrategyDecision:
        primary, confirm = self._extract(data)

        period = int(self.params.get("vwap_period", 240))
        buy_band_mult = float(self.params.get("buy_band_mult", 1.0))   # VWAP - N*std'de BUY
        sell_band_mult = float(self.params.get("sell_band_mult", 1.0))  # VWAP + N*std'de SELL

        min_len = period + 5
        if len(primary) < min_len:
            return StrategyDecision(signal="HOLD", reason="insufficient_data", metadata={})

        vwap, std, tp = self._vwap_bands(primary, period)

        if vwap.isna().iloc[-1] or std.isna().iloc[-1]:
            return StrategyDecision(signal="HOLD", reason="vwap_nan", metadata={})

        current_tp = float(tp.iloc[-1])
        current_vwap = float(vwap.iloc[-1])
        current_std = float(std.iloc[-1])

        if current_std <= 0 or current_vwap <= 0:
            return StrategyDecision(signal="HOLD", reason="zero_std_or_vwap", metadata={})

        lower_band = current_vwap - buy_band_mult * current_std
        upper_band = current_vwap + sell_band_mult * current_std

        # Fiyatın VWAP'tan sapması (normalize)
        deviation = (current_tp - current_vwap) / current_std

        # HTF trend filtresi: TREND rejiminde VWAP sinyali HTF ile uyuşmalı
        htf_bullish = True
        htf_bearish = True
        if len(confirm) >= 52:
            close_c = confirm["close"].astype(float)
            htf_fast = float(close_c.ewm(span=20, adjust=False).mean().iloc[-1])
            htf_slow = float(close_c.ewm(span=50, adjust=False).mean().iloc[-1])
            htf_bullish = htf_fast > htf_slow
            htf_bearish = htf_fast < htf_slow

        meta = {
            "vwap": round(current_vwap, 6),
            "current_tp": round(current_tp, 6),
            "std": round(current_std, 6),
            "deviation": round(deviation, 3),
            "lower_band": round(lower_band, 6),
            "upper_band": round(upper_band, 6),
        }

        # BUY: fiyat alt bantın altına düştü ve HTF destekliyor
        if current_tp <= lower_band and htf_bullish:
            return StrategyDecision(
                signal="BUY",
                reason="vwap_oversold_reversion",
                metadata={**meta, "htf_trend": "bullish"},
            )

        # SELL: fiyat üst bantın üstüne çıktı ve HTF destekliyor
        if current_tp >= upper_band and htf_bearish:
            return StrategyDecision(
                signal="SELL",
                reason="vwap_overbought_reversion",
                metadata={**meta, "htf_trend": "bearish"},
            )

        return StrategyDecision(
            signal="HOLD",
            reason="within_vwap_bands",
            metadata=meta,
        )
