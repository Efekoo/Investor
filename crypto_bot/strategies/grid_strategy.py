from __future__ import annotations

import numpy as np
import pandas as pd

from crypto_bot.strategies.base_strategy import Strategy, StrategyDecision


class GridStrategy(Strategy):
    """
    RANGE rejimine özel basitleştirilmiş grid stratejisi.

    Fiyatı son N mumun yüksek/düşük aralığına böler ve grid seviyeleri oluşturur.
    Fiyat alt grid bölgesine yaklaştığında BUY, üst bölgeye yaklaştığında SELL üretir.
    Gerçek grid'in aksine mevcut mimariyle uyumlu tek-pozisyon yaklaşımı kullanır.
    """

    def __init__(self, params: dict) -> None:
        super().__init__("grid", params)
        self.supported_regimes = ["RANGE"]

    def _extract(self, data: pd.DataFrame | dict[str, pd.DataFrame]) -> pd.DataFrame:
        if isinstance(data, dict):
            tf = self.params.get("primary_timeframe", "1m")
            return data.get(tf, next(iter(data.values())))
        return data

    @staticmethod
    def _atr(df: pd.DataFrame, period: int) -> float:
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        close = df["close"].astype(float)
        prev_close = close.shift(1)
        tr = pd.concat(
            [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
            axis=1,
        ).max(axis=1)
        return float(tr.rolling(period).mean().iloc[-1])

    def generate_signal(self, data: pd.DataFrame | dict[str, pd.DataFrame]) -> str:
        return self.generate_decision(data).signal

    def generate_decision(self, data: pd.DataFrame | dict[str, pd.DataFrame]) -> StrategyDecision:
        primary = self._extract(data)

        lookback = int(self.params.get("lookback", 50))
        grid_levels = int(self.params.get("grid_levels", 5))
        proximity_pct = float(self.params.get("proximity_pct", 0.003))
        max_range_pct = float(self.params.get("max_range_pct", 0.05))
        min_range_pct = float(self.params.get("min_range_pct", 0.005))

        min_len = lookback + 10
        if len(primary) < min_len:
            return StrategyDecision(signal="HOLD", reason="insufficient_data", metadata={})

        close = primary["close"].astype(float)
        high = primary["high"].astype(float)
        low = primary["low"].astype(float)

        window = primary.iloc[-lookback:]
        recent_high = float(high.iloc[-lookback:].max())
        recent_low = float(low.iloc[-lookback:].min())
        range_size = recent_high - recent_low

        if range_size <= 0:
            return StrategyDecision(signal="HOLD", reason="zero_range", metadata={})

        range_pct = range_size / max(recent_low, 1e-9)

        # Aralık çok genişse TREND/VOLATILE var → grid uygun değil
        if range_pct > max_range_pct:
            return StrategyDecision(
                signal="HOLD",
                reason="range_too_wide",
                metadata={"range_pct": round(range_pct, 4)},
            )

        # Aralık çok darsa işlem yapma (spread maliyeti kazancı geçer)
        if range_pct < min_range_pct:
            return StrategyDecision(
                signal="HOLD",
                reason="range_too_narrow",
                metadata={"range_pct": round(range_pct, 4)},
            )

        current_price = float(close.iloc[-1])
        step = range_size / grid_levels

        # Grid seviyelerini hesapla
        grid_prices = [recent_low + i * step for i in range(grid_levels + 1)]

        # Fiyatın aralık içindeki konumu (0.0 = alt, 1.0 = üst)
        position_in_range = (current_price - recent_low) / range_size

        for i, level in enumerate(grid_prices):
            if level <= 0:
                continue
            distance_pct = abs(current_price - level) / level
            if distance_pct > proximity_pct:
                continue

            # Alt yarı → destek seviyeleri → BUY
            if i <= grid_levels // 2 and position_in_range <= 0.45:
                return StrategyDecision(
                    signal="BUY",
                    reason=f"grid_support_level_{i}_of_{grid_levels}",
                    metadata={
                        "level": round(level, 6),
                        "range_pct": round(range_pct, 4),
                        "position_in_range": round(position_in_range, 3),
                        "grid_index": i,
                        "recent_low": recent_low,
                        "recent_high": recent_high,
                    },
                )

            # Üst yarı → direnç seviyeleri → SELL
            if i >= (grid_levels + 1) // 2 and position_in_range >= 0.55:
                return StrategyDecision(
                    signal="SELL",
                    reason=f"grid_resistance_level_{i}_of_{grid_levels}",
                    metadata={
                        "level": round(level, 6),
                        "range_pct": round(range_pct, 4),
                        "position_in_range": round(position_in_range, 3),
                        "grid_index": i,
                        "recent_low": recent_low,
                        "recent_high": recent_high,
                    },
                )

        return StrategyDecision(
            signal="HOLD",
            reason="between_grid_levels",
            metadata={"position_in_range": round(position_in_range, 3)},
        )
