from __future__ import annotations

import numpy as np
import pandas as pd

from crypto_bot.strategies.base_strategy import Strategy, StrategyDecision


class SupertrendStrategy(Strategy):
    """
    Supertrend göstergesi tabanlı dinamik trend takip stratejisi.

    ATR çarpanı ile hesaplanan dinamik destek/direnç bandı kullanır.
    Supertrend çizgisi alt bant olduğunda yukarı trend (BUY yönü),
    üst bant olduğunda aşağı trend (SELL yönü) demektir.

    Sinyal: Supertrend yönü değiştiğinde (flip) → işlem aç.
    HTF onayı ile whipsaw riski düşürülür.

    TREND ve VOLATILE rejimlerinde aktiftir.
    """

    def __init__(self, params: dict) -> None:
        super().__init__("supertrend", params)
        self.supported_regimes = ["TREND", "VOLATILE"]

    def _extract(self, data: pd.DataFrame | dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
        if isinstance(data, dict):
            primary_tf = self.params.get("primary_timeframe", "1m")
            confirm_tf = self.params.get("confirm_timeframe", primary_tf)
            primary = data.get(primary_tf, next(iter(data.values())))
            confirm = data.get(confirm_tf, primary)
            return primary, confirm
        return data, data

    @staticmethod
    def _supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> tuple[np.ndarray, np.ndarray]:
        """
        Supertrend hesaplar.

        Dönüş:
          supertrend_values : Her mum için Supertrend seviyesi
          trend_direction   : +1 = yukarı trend (BUY), -1 = aşağı trend (SELL)
        """
        high = df["high"].astype(float).values
        low = df["low"].astype(float).values
        close = df["close"].astype(float).values
        n = len(close)

        # ATR — Wilder smoothing (EWM ile eşdeğer)
        tr = np.zeros(n)
        tr[0] = high[0] - low[0]
        for i in range(1, n):
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i] - close[i - 1]),
            )

        atr = np.zeros(n)
        atr[0] = tr[0]
        alpha = 1.0 / period
        for i in range(1, n):
            atr[i] = alpha * tr[i] + (1 - alpha) * atr[i - 1]

        hl2 = (high + low) / 2.0
        upper_basic = hl2 + multiplier * atr
        lower_basic = hl2 - multiplier * atr

        final_upper = upper_basic.copy()
        final_lower = lower_basic.copy()
        trend = np.ones(n, dtype=int)

        for i in range(1, n):
            # Final Upper Band
            if upper_basic[i] < final_upper[i - 1] or close[i - 1] > final_upper[i - 1]:
                final_upper[i] = upper_basic[i]
            else:
                final_upper[i] = final_upper[i - 1]

            # Final Lower Band
            if lower_basic[i] > final_lower[i - 1] or close[i - 1] < final_lower[i - 1]:
                final_lower[i] = lower_basic[i]
            else:
                final_lower[i] = final_lower[i - 1]

            # Trend yönü
            if close[i] > final_upper[i - 1]:
                trend[i] = 1
            elif close[i] < final_lower[i - 1]:
                trend[i] = -1
            else:
                trend[i] = trend[i - 1]

        # Supertrend değeri: yukarı trendde alt bant, aşağı trendde üst bant
        supertrend_vals = np.where(trend == 1, final_lower, final_upper)
        return supertrend_vals, trend

    def generate_signal(self, data: pd.DataFrame | dict[str, pd.DataFrame]) -> str:
        return self.generate_decision(data).signal

    def generate_decision(self, data: pd.DataFrame | dict[str, pd.DataFrame]) -> StrategyDecision:
        primary, confirm = self._extract(data)

        period = int(self.params.get("atr_period", 10))
        multiplier = float(self.params.get("multiplier", 3.0))

        min_len = period * 3 + 5
        if len(primary) < min_len:
            return StrategyDecision(signal="HOLD", reason="insufficient_data", metadata={})

        st_vals, trend = self._supertrend(primary, period=period, multiplier=multiplier)

        current_trend = int(trend[-1])
        prev_trend = int(trend[-2])
        current_st = float(st_vals[-1])
        current_price = float(primary["close"].astype(float).iloc[-1])

        # Flip tespiti: sadece yön değişiminde sinyal ver
        flipped_up = prev_trend == -1 and current_trend == 1
        flipped_down = prev_trend == 1 and current_trend == -1

        # HTF onay filtresi
        htf_bullish = True
        htf_bearish = True
        if len(confirm) >= 52:
            close_c = confirm["close"].astype(float)
            htf_fast = float(close_c.ewm(span=20, adjust=False).mean().iloc[-1])
            htf_slow = float(close_c.ewm(span=50, adjust=False).mean().iloc[-1])
            htf_bullish = htf_fast > htf_slow
            htf_bearish = htf_fast < htf_slow

        meta = {
            "supertrend": round(current_st, 6),
            "current_price": round(current_price, 6),
            "trend_direction": current_trend,
            "flipped": flipped_up or flipped_down,
        }

        if flipped_up and htf_bullish:
            return StrategyDecision(
                signal="BUY",
                reason="supertrend_flip_up_htf_confirmed",
                metadata={**meta, "htf_trend": "bullish"},
            )

        if flipped_down and htf_bearish:
            return StrategyDecision(
                signal="SELL",
                reason="supertrend_flip_down_htf_confirmed",
                metadata={**meta, "htf_trend": "bearish"},
            )

        # Flip yok ama güçlü trend sürüyor → HOLD (pozisyon yönetimi devam eder)
        return StrategyDecision(
            signal="HOLD",
            reason=f"supertrend_no_flip_trend={'up' if current_trend == 1 else 'down'}",
            metadata=meta,
        )
