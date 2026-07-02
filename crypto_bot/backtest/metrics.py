from __future__ import annotations

import math

import numpy as np
import pandas as pd


def win_rate(trades: pd.DataFrame) -> float:
    if trades.empty:
        return 0.0
    wins = (trades["pnl"] > 0).sum()
    return float(wins / len(trades))


def profit_factor(trades: pd.DataFrame) -> float:
    if trades.empty:
        return 0.0
    gross_profit = trades.loc[trades["pnl"] > 0, "pnl"].sum()
    gross_loss = abs(trades.loc[trades["pnl"] < 0, "pnl"].sum())
    if gross_loss == 0:
        return math.inf if gross_profit > 0 else 0.0
    return float(gross_profit / gross_loss)


def max_drawdown(equity_curve: pd.Series) -> float:
    if equity_curve.empty:
        return 0.0
    running_max = equity_curve.cummax()
    drawdowns = (equity_curve - running_max) / running_max.replace(0, np.nan)
    return float(drawdowns.min())


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 365 * 24 * 60) -> float:
    if returns.empty:
        return 0.0
    excess = returns - (risk_free_rate / periods_per_year)
    std = excess.std(ddof=0)
    if std == 0:
        return 0.0
    return float((excess.mean() / std) * np.sqrt(periods_per_year))
