from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class StrategyDecision:
    signal: str
    reason: str
    metadata: dict[str, Any]


class Strategy(ABC):
    def __init__(self, name: str, params: dict) -> None:
        self.name = name
        self.params = params
        self.supported_regimes: list[str] = ["TREND", "RANGE", "VOLATILE"]

    @abstractmethod
    def generate_signal(self, data: pd.DataFrame | dict[str, pd.DataFrame]) -> str:
        raise NotImplementedError

    def generate_decision(self, data: pd.DataFrame | dict[str, pd.DataFrame]) -> StrategyDecision:
        signal = self.generate_signal(data)
        return StrategyDecision(signal=signal, reason=f"{self.name} signal", metadata={})
