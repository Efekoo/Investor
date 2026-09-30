from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class LeverageConfig:
    """Kaldıraçlı (USDT-M perpetual futures) işlem ayarları.

    enabled=False iken bot spot gibi davranır (kaldıraç 1x, eski nakit muhasebesi).
    """

    enabled: bool = False
    leverage: float = 1.0
    max_leverage: float = 10.0            # Güvenlik tavanı: config'te daha yüksek yazılsa da aşılmaz
    margin_mode: str = "isolated"         # isolated | cross
    market_type: str = "swap"             # ccxt defaultType (perpetual futures)
    settle_currency: str = "USDT"
    maintenance_margin_rate: float = 0.005
    liquidation_buffer: float = 0.5       # SL, likidasyon mesafesinin en fazla bu oranında olabilir

    @classmethod
    def from_settings(cls, settings: dict[str, Any]) -> "LeverageConfig":
        cfg = settings.get("leverage", {}) or {}
        max_lev = max(1.0, float(cfg.get("max_leverage", 10.0)))
        lev = min(max(1.0, float(cfg.get("leverage", 1.0))), max_lev)
        margin_mode = str(cfg.get("margin_mode", "isolated")).lower()
        if margin_mode not in {"isolated", "cross"}:
            raise ValueError(f"Invalid leverage.margin_mode: {margin_mode}")
        buffer = float(cfg.get("liquidation_buffer", 0.5))
        if not 0.0 < buffer < 1.0:
            raise ValueError("leverage.liquidation_buffer must be between 0 and 1")
        return cls(
            enabled=bool(cfg.get("enabled", False)),
            leverage=lev,
            max_leverage=max_lev,
            margin_mode=margin_mode,
            market_type=str(cfg.get("market_type", "swap")),
            settle_currency=str(cfg.get("settle_currency", settings.get("trading", {}).get("quote_currency", "USDT"))),
            maintenance_margin_rate=float(cfg.get("maintenance_margin_rate", 0.005)),
            liquidation_buffer=buffer,
        )

    @property
    def effective_leverage(self) -> float:
        return self.leverage if self.enabled else 1.0


def futures_symbol(symbol: str, settle: str) -> str:
    """'BTC/USDT' → 'BTC/USDT:USDT' (ccxt birleşik perpetual sembolü)."""
    return symbol if ":" in symbol else f"{symbol}:{settle}"


def spot_symbol(symbol: str) -> str:
    """'BTC/USDT:USDT' → 'BTC/USDT' (botun iç sembol biçimi)."""
    return symbol.split(":", 1)[0]


def liquidation_price(side: str, entry_price: float, leverage: float, maintenance_margin_rate: float) -> float:
    """Isolated margin için yaklaşık likidasyon fiyatı. Kaldıraç ≤ 1 ise 0 (likidasyon yok).

    Long:  entry * (1 - 1/lev + mmr)
    Short: entry * (1 + 1/lev - mmr)
    """
    if leverage <= 1.0 or entry_price <= 0:
        return 0.0
    distance = 1.0 / leverage - maintenance_margin_rate
    if side.upper() == "BUY":
        return entry_price * (1.0 - distance)
    return entry_price * (1.0 + distance)


def is_liquidated(side: str, price: float, liq_price: float) -> bool:
    if liq_price <= 0:
        return False
    if side.upper() == "BUY":
        return price <= liq_price
    return price >= liq_price


def clamp_stop_to_liquidation(
    side: str,
    entry_price: float,
    stop_price: float,
    leverage: float,
    maintenance_margin_rate: float,
    buffer: float,
) -> float:
    """Stop-loss'u likidasyondan önce tetiklenecek şekilde içeri çeker.

    Stop mesafesi en fazla (1/lev - mmr) * buffer olabilir; böylece normal
    koşullarda pozisyon likidite olmadan SL ile kapanır.
    """
    if leverage <= 1.0 or entry_price <= 0:
        return stop_price
    max_distance = (1.0 / leverage - maintenance_margin_rate) * buffer
    if side.upper() == "BUY":
        return max(stop_price, entry_price * (1.0 - max_distance))
    return min(stop_price, entry_price * (1.0 + max_distance))
