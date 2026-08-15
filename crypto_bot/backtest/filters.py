from __future__ import annotations

"""Entry filters for backtests.

A filter is a boolean Series aligned to the OHLCV DataFrame: True = new long
entries allowed on that candle, False = stay flat. Filters only use past data
(EMAs), so there is no lookahead.
"""

import pandas as pd


def bull_regime_filter(
    data: pd.DataFrame,
    trend_ema_period: int = 200,
    confirm_ema_period: int = 50,
    require_slope: bool = True,
) -> pd.Series:
    """Allow long entries only in an up-regime.

    Conditions (all must hold):
      1. close > EMA(trend_ema_period)      -> price above long-term trend
      2. EMA(confirm) > EMA(trend)          -> medium-term above long-term
      3. EMA(trend) rising (optional)       -> long-term trend not falling

    In a bear year this keeps the bot in cash most of the time, which is the
    point: not losing is the first step to compounding.
    """
    close = data["close"].astype(float)
    ema_trend = close.ewm(span=trend_ema_period, adjust=False).mean()
    ema_confirm = close.ewm(span=confirm_ema_period, adjust=False).mean()

    allowed = (close > ema_trend) & (ema_confirm > ema_trend)
    if require_slope:
        allowed &= ema_trend.diff() > 0

    # Warmup: EMA'ler oturana kadar işlem açma
    allowed.iloc[: trend_ema_period] = False
    return allowed.fillna(False)


def bear_regime_filter(
    data: pd.DataFrame,
    trend_ema_period: int = 200,
    confirm_ema_period: int = 50,
    require_slope: bool = True,
) -> pd.Series:
    """Mirror of bull_regime_filter for short entries: only allow shorts in a
    confirmed down-regime."""
    close = data["close"].astype(float)
    ema_trend = close.ewm(span=trend_ema_period, adjust=False).mean()
    ema_confirm = close.ewm(span=confirm_ema_period, adjust=False).mean()

    allowed = (close < ema_trend) & (ema_confirm < ema_trend)
    if require_slope:
        allowed &= ema_trend.diff() < 0

    allowed.iloc[: trend_ema_period] = False
    return allowed.fillna(False)
