from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class RegimeState:
    regime: str
    atr_pct: float
    adx: float
    volatility: float


class RegimeDetector:
    def __init__(
        self,
        atr_period: int = 14,
        adx_period: int = 14,
        trend_adx_threshold: float = 25.0,
        high_volatility_threshold: float = 0.02,
        low_volatility_threshold: float = 0.005,
    ) -> None:
        self.atr_period = atr_period
        self.adx_period = adx_period
        self.trend_adx_threshold = trend_adx_threshold
        self.high_volatility_threshold = high_volatility_threshold
        self.low_volatility_threshold = low_volatility_threshold

    @staticmethod
    def _atr(df: pd.DataFrame, period: int) -> pd.Series:
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        close = df["close"].astype(float)
        prev_close = close.shift(1)
        tr = pd.concat(
            [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
            axis=1,
        ).max(axis=1)
        return tr.rolling(period).mean()

    @staticmethod
    def _adx(df: pd.DataFrame, period: int) -> pd.Series:
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        close = df["close"].astype(float)

        plus_dm = (high.diff()).clip(lower=0)
        minus_dm = (-low.diff()).clip(lower=0)
        plus_dm[plus_dm < minus_dm] = 0
        minus_dm[minus_dm < plus_dm] = 0

        prev_close = close.shift(1)
        tr = pd.concat(
            [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(period).mean().replace(0, 1e-9)

        plus_di = 100 * (plus_dm.rolling(period).sum() / atr)
        minus_di = 100 * (minus_dm.rolling(period).sum() / atr)
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1e-9)
        return dx.rolling(period).mean()

    def detect(self, df: pd.DataFrame) -> RegimeState:
        if df.empty or len(df) < max(self.atr_period, self.adx_period) + 5:
            return RegimeState(regime="RANGE", atr_pct=0.0, adx=0.0, volatility=0.0)

        close = df["close"].astype(float)
        atr = self._atr(df, self.atr_period).iloc[-1]
        adx = float(self._adx(df, self.adx_period).iloc[-1])
        atr_pct = float(atr / max(close.iloc[-1], 1e-9))
        volatility = float(close.pct_change().dropna().tail(30).std(ddof=0))

        if volatility >= self.high_volatility_threshold or atr_pct >= self.high_volatility_threshold:
            regime = "VOLATILE"
        elif adx >= self.trend_adx_threshold and volatility >= self.low_volatility_threshold:
            regime = "TREND"
        else:
            regime = "RANGE"

        return RegimeState(regime=regime, atr_pct=atr_pct, adx=adx, volatility=volatility)

    def as_dict(self, state: RegimeState) -> dict[str, Any]:
        return {
            "regime": state.regime,
            "atr_pct": state.atr_pct,
            "adx": state.adx,
            "volatility": state.volatility,
        }
