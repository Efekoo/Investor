from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, AsyncGenerator

import pandas as pd


class DataManager:
    def __init__(self, exchange: Any, logger: Any, cache_size: int = 3000) -> None:
        self.exchange = exchange
        self.logger = logger
        self.cache_size = cache_size
        self._cache: dict[tuple[str, str], deque[pd.DataFrame]] = defaultdict(
            lambda: deque(maxlen=cache_size)
        )

    @staticmethod
    def _to_dataframe(ohlcv: list[list[float]]) -> pd.DataFrame:
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        if df.empty:
            return df
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
        return df

    def fetch_symbol_data(self, symbol: str, timeframe: str, limit: int = 500) -> pd.DataFrame:
        raw = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = self._to_dataframe(raw)
        self._cache[(symbol, timeframe)].append(df)
        return df

    def fetch_multi_symbol_data(
        self, symbols: list[str], timeframe: str, limit: int = 500
    ) -> dict[str, pd.DataFrame]:
        data: dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            data[symbol] = self.fetch_symbol_data(symbol, timeframe, limit)
        return data

    def get_cached_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        chunks = self._cache[(symbol, timeframe)]
        if not chunks:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        return pd.concat(list(chunks), ignore_index=True).drop_duplicates(subset=["timestamp"]).sort_values(
            "timestamp"
        )

    async def stream_symbol_data(self, symbol: str, timeframe: str) -> AsyncGenerator[pd.DataFrame, None]:
        async for candle in self.exchange.stream_ohlcv(symbol, timeframe):
            df = self._to_dataframe([candle])
            if not df.empty:
                self._cache[(symbol, timeframe)].append(df)
                yield df
