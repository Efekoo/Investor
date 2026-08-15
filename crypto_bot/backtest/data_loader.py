from __future__ import annotations

"""Historical OHLCV data downloading and caching for backtests.

Usage (CLI examples):
    from crypto_bot.backtest.data_loader import load_or_download
    df = load_or_download("binance", "BTC/USDT", "1h", days=365)

Data is cached under ``runtime/data/`` as parquet (CSV fallback) so repeated
backtests don't re-download. Synthetic data generation is provided for tests
and offline development.
"""

import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]

_TF_SECONDS = {"m": 60, "h": 3600, "d": 86400, "w": 604800}


def timeframe_to_ms(timeframe: str) -> int:
    value = int(timeframe[:-1])
    unit = timeframe[-1]
    if unit not in _TF_SECONDS:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    return value * _TF_SECONDS[unit] * 1000


def _cache_path(cache_dir: Path, exchange_id: str, symbol: str, timeframe: str) -> Path:
    safe_symbol = symbol.replace("/", "-").replace(":", "_")
    return cache_dir / f"{exchange_id}_{safe_symbol}_{timeframe}"


def _read_cache(base: Path) -> pd.DataFrame | None:
    pq, csv = base.with_suffix(".parquet"), base.with_suffix(".csv")
    try:
        if pq.exists():
            return pd.read_parquet(pq)
        if csv.exists():
            df = pd.read_csv(csv)
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            return df
    except Exception:
        return None
    return None


def _write_cache(df: pd.DataFrame, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(base.with_suffix(".parquet"), index=False)
    except Exception:
        df.to_csv(base.with_suffix(".csv"), index=False)


def download_ohlcv(
    exchange_id: str,
    symbol: str,
    timeframe: str,
    since_ms: int,
    until_ms: int | None = None,
    batch_limit: int = 1000,
    rate_limit_sleep: float = 0.2,
) -> pd.DataFrame:
    """Download OHLCV from an exchange with pagination (requires network)."""
    import ccxt  # lazy import: tests must not require ccxt

    exchange = getattr(ccxt, exchange_id)({"enableRateLimit": True})
    until_ms = until_ms or int(time.time() * 1000)
    tf_ms = timeframe_to_ms(timeframe)

    all_rows: list[list[float]] = []
    cursor = since_ms
    while cursor < until_ms:
        batch = exchange.fetch_ohlcv(symbol, timeframe, since=cursor, limit=batch_limit)
        if not batch:
            break
        all_rows.extend(batch)
        last_ts = batch[-1][0]
        if last_ts <= cursor:  # no progress -> avoid infinite loop
            break
        cursor = last_ts + tf_ms
        time.sleep(rate_limit_sleep)

    df = pd.DataFrame(all_rows, columns=OHLCV_COLUMNS)
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df[df["timestamp"] <= pd.Timestamp(until_ms, unit="ms", tz="UTC")]
    return (
        df.sort_values("timestamp")
        .drop_duplicates(subset=["timestamp"])
        .reset_index(drop=True)
    )


def load_or_download(
    exchange_id: str,
    symbol: str,
    timeframe: str,
    days: int = 365,
    cache_dir: str | Path = "runtime/data",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Return cached OHLCV history, downloading (or topping up) when needed."""
    cache_dir = Path(cache_dir)
    base = _cache_path(cache_dir, exchange_id, symbol, timeframe)
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=days)
    tf_ms = timeframe_to_ms(timeframe)

    cached = None if force_refresh else _read_cache(base)
    if cached is not None and not cached.empty:
        cached_start = cached["timestamp"].iloc[0]
        cached_end = cached["timestamp"].iloc[-1]
        fresh_enough = (now - cached_end) < timedelta(milliseconds=2 * tf_ms)
        covers_start = cached_start <= pd.Timestamp(start)
        if covers_start and fresh_enough:
            return cached[cached["timestamp"] >= pd.Timestamp(start)].reset_index(drop=True)
        # top-up: continue from cache end
        since_ms = int(cached_end.timestamp() * 1000) + tf_ms if covers_start else int(start.timestamp() * 1000)
        fresh = download_ohlcv(exchange_id, symbol, timeframe, since_ms)
        merged = (
            pd.concat([cached, fresh], ignore_index=True)
            .drop_duplicates(subset=["timestamp"])
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        _write_cache(merged, base)
        return merged[merged["timestamp"] >= pd.Timestamp(start)].reset_index(drop=True)

    df = download_ohlcv(exchange_id, symbol, timeframe, int(start.timestamp() * 1000))
    if not df.empty:
        _write_cache(df, base)
    return df


def generate_synthetic_ohlcv(
    n: int = 2000,
    timeframe: str = "1h",
    start_price: float = 100.0,
    drift: float = 0.00005,
    volatility: float = 0.01,
    regime_switch_every: int = 500,
    seed: int | None = 42,
) -> pd.DataFrame:
    """Generate regime-switching random-walk OHLCV data for offline tests.

    Alternates between trending and ranging regimes so both trend-following
    and mean-reversion strategies get realistic conditions.
    """
    rng = np.random.default_rng(seed)
    tf_ms = timeframe_to_ms(timeframe)

    closes = np.empty(n)
    price = start_price
    for i in range(n):
        regime = (i // max(1, regime_switch_every)) % 3
        if regime == 0:      # up-trend
            mu, sigma = drift * 20, volatility
        elif regime == 1:    # range
            mu, sigma = 0.0, volatility * 0.6
        else:                # down-trend
            mu, sigma = -drift * 15, volatility * 1.2
        price *= 1 + rng.normal(mu, sigma)
        price = max(price, 0.01)
        closes[i] = price

    opens = np.concatenate([[start_price], closes[:-1]])
    spreads = np.abs(rng.normal(0, volatility / 2, n)) * closes
    highs = np.maximum(opens, closes) + spreads
    lows = np.minimum(opens, closes) - spreads
    volumes = rng.lognormal(mean=7, sigma=0.5, size=n)

    end = pd.Timestamp.now(tz="UTC").floor("h")
    timestamps = pd.date_range(end=end, periods=n, freq=pd.Timedelta(milliseconds=tf_ms))
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        }
    )


def resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Resample lower-timeframe OHLCV to a higher timeframe (e.g. 1m -> 1h)."""
    tf_ms = timeframe_to_ms(timeframe)
    out = (
        df.set_index("timestamp")
        .resample(pd.Timedelta(milliseconds=tf_ms), label="left", closed="left")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna()
        .reset_index()
    )
    return out
