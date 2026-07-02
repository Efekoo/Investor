from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict

try:
    import requests as _requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False


# Binance USDT-M perpetual endpoints (public, API key gerekmez)
_PREMIUM_INDEX_URL = "https://fapi.binance.com/fapi/v1/premiumIndex"
_LONG_SHORT_URL = "https://fapi.binance.com/futures/data/globalLongShortAccountRatio"
_OPEN_INTEREST_URL = "https://fapi.binance.com/fapi/v1/openInterest"

_DEFAULT_CACHE_TTL = 300  # 5 dakika


@dataclass
class FundingRateSignal:
    symbol: str
    funding_rate: float        # Son funding oranı (örn. 0.0001 = 0.01%)
    funding_label: str         # "HIGH_POSITIVE", "MODERATE_POSITIVE", "NEUTRAL", "MODERATE_NEGATIVE", "HIGH_NEGATIVE"
    long_short_ratio: float    # Long / Short oranı (1.0 = eşit, >1 = uzunlar fazla)
    open_interest: float       # Açık faiz (USDT)
    risk_multiplier: float     # Pozisyon boyutuna uygulanacak çarpan (0.7 – 1.1)
    filter_buy: bool           # True → BUY işlemi filtrele
    filter_sell: bool          # True → SELL işlemi filtrele
    metadata: Dict[str, Any] = field(default_factory=dict)


def _normalize_symbol(symbol: str) -> str:
    """'BTC/USDT' → 'BTCUSDT' dönüşümü"""
    return symbol.replace("/", "")


class FundingRateAnalyzer:
    """
    Binance USDT-M perpetual piyasa yapısı sinyalleri.

    Üç sinyal kaynağı kullanır:
    1. Funding Rate  — Long/short tarafından ödenen periyodik ücret
       - Yüksek pozitif: Uzunlar aşırı yüklenmiş → BUY riski yüksek
       - Yüksek negatif: Kısalar aşırı yüklenmiş → SELL riski yüksek

    2. Long/Short Ratio — Hesap bazlı uzun/kısa pozisyon oranı
       - Aşırı uzun: >2.0 → kontrarian SELL sinyali
       - Aşırı kısa: <0.5 → kontrarian BUY sinyali

    3. Open Interest — Toplam açık pozisyon değeri
       - Trend ile birlikte artıyorsa trend güçlü
       - Trend zıttına artıyorsa dikkat (short/long squeeze riski)
    """

    def __init__(self, config: Dict[str, Any], logger: Any) -> None:
        self.config = config
        self.logger = logger
        self.enabled = config.get("enabled", True)
        self._cache: Dict[str, tuple[float, FundingRateSignal]] = {}
        self._cache_ttl = float(config.get("cache_ttl_seconds", _DEFAULT_CACHE_TTL))

        # Eşik değerleri
        self._ext_pos = float(config.get("extreme_positive_threshold", 0.001))
        self._ext_neg = float(config.get("extreme_negative_threshold", -0.001))
        self._mod_pos = float(config.get("moderate_positive_threshold", 0.0003))
        self._mod_neg = float(config.get("moderate_negative_threshold", -0.0003))

    # ------------------------------------------------------------------
    # Veri çekme
    # ------------------------------------------------------------------
    def _fetch_funding_rate(self, binance_symbol: str) -> float:
        if not _REQUESTS_AVAILABLE:
            return 0.0
        try:
            resp = _requests.get(_PREMIUM_INDEX_URL, params={"symbol": binance_symbol}, timeout=5)
            resp.raise_for_status()
            data = resp.json()
            return float(data.get("lastFundingRate", 0.0))
        except Exception as exc:
            self.logger.debug("Funding rate fetch failed for %s: %s", binance_symbol, exc)
            return 0.0

    def _fetch_long_short_ratio(self, binance_symbol: str) -> float:
        if not _REQUESTS_AVAILABLE:
            return 1.0
        try:
            resp = _requests.get(
                _LONG_SHORT_URL,
                params={"symbol": binance_symbol, "period": "5m", "limit": 1},
                timeout=5,
            )
            resp.raise_for_status()
            data = resp.json()
            if data and isinstance(data, list):
                return float(data[0].get("longShortRatio", 1.0))
            return 1.0
        except Exception as exc:
            self.logger.debug("Long/short ratio fetch failed for %s: %s", binance_symbol, exc)
            return 1.0

    def _fetch_open_interest(self, binance_symbol: str) -> float:
        if not _REQUESTS_AVAILABLE:
            return 0.0
        try:
            resp = _requests.get(_OPEN_INTEREST_URL, params={"symbol": binance_symbol}, timeout=5)
            resp.raise_for_status()
            data = resp.json()
            return float(data.get("openInterest", 0.0))
        except Exception as exc:
            self.logger.debug("Open interest fetch failed for %s: %s", binance_symbol, exc)
            return 0.0

    # ------------------------------------------------------------------
    # Sinyal üretimi
    # ------------------------------------------------------------------
    def _classify_funding(self, rate: float) -> str:
        if rate >= self._ext_pos:
            return "HIGH_POSITIVE"
        if rate >= self._mod_pos:
            return "MODERATE_POSITIVE"
        if rate <= self._ext_neg:
            return "HIGH_NEGATIVE"
        if rate <= self._mod_neg:
            return "MODERATE_NEGATIVE"
        return "NEUTRAL"

    def _compute_risk_multiplier(
        self, funding_rate: float, long_short_ratio: float, label: str
    ) -> float:
        """
        Risk çarpanı hesaplar (0.6 – 1.1 arası).

        Funding mantığı:
          Yüksek pozitif funding → uzunlar fazla → BUY riski yüksek → çarpanı düşür
          Yüksek negatif funding → kısalar fazla → SELL riski yüksek → çarpanı düşür
          Nötr                   → standart

        Long/Short ratio mantığı:
          >2.0 → kalabalık uzun pozisyon → BUY için dikkatli
          <0.5 → kalabalık kısa pozisyon → SELL için dikkatli
        """
        mult = 1.0

        # Funding etkisi
        if label == "HIGH_POSITIVE":
            mult *= 0.70
        elif label == "MODERATE_POSITIVE":
            mult *= 0.85
        elif label == "HIGH_NEGATIVE":
            mult *= 0.70
        elif label == "MODERATE_NEGATIVE":
            mult *= 0.85

        # Long/short crowd etkisi
        if long_short_ratio > 2.5:
            mult *= 0.80
        elif long_short_ratio > 2.0:
            mult *= 0.90
        elif long_short_ratio < 0.4:
            mult *= 0.80
        elif long_short_ratio < 0.5:
            mult *= 0.90

        return round(max(0.60, min(mult, 1.10)), 3)

    def _compute_filters(self, label: str, long_short_ratio: float) -> tuple[bool, bool]:
        """
        (filter_buy, filter_sell) döner.
        True → o yönde işlem açma.
        """
        # Aşırı uzun → BUY filtrele
        filter_buy = label == "HIGH_POSITIVE" or long_short_ratio > 3.0
        # Aşırı kısa → SELL filtrele
        filter_sell = label == "HIGH_NEGATIVE" or long_short_ratio < 0.33
        return filter_buy, filter_sell

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fetch_signal(self, symbol: str) -> FundingRateSignal:
        """
        Sembol için funding rate, long/short ratio ve open interest verilerini çeker,
        sinyal ve çarpanları hesaplayarak FundingRateSignal döner.

        Sonuçlar `cache_ttl_seconds` süresince önbelleğe alınır.
        """
        if not self.enabled:
            return FundingRateSignal(
                symbol=symbol,
                funding_rate=0.0,
                funding_label="NEUTRAL",
                long_short_ratio=1.0,
                open_interest=0.0,
                risk_multiplier=1.0,
                filter_buy=False,
                filter_sell=False,
            )

        now = time.time()
        cached_ts, cached_signal = self._cache.get(symbol, (0.0, None))
        if cached_signal is not None and now - cached_ts < self._cache_ttl:
            return cached_signal

        binance_symbol = _normalize_symbol(symbol)

        funding_rate = self._fetch_funding_rate(binance_symbol)
        long_short_ratio = self._fetch_long_short_ratio(binance_symbol)
        open_interest = self._fetch_open_interest(binance_symbol)

        label = self._classify_funding(funding_rate)
        risk_mult = self._compute_risk_multiplier(funding_rate, long_short_ratio, label)
        filter_buy, filter_sell = self._compute_filters(label, long_short_ratio)

        signal = FundingRateSignal(
            symbol=symbol,
            funding_rate=funding_rate,
            funding_label=label,
            long_short_ratio=long_short_ratio,
            open_interest=open_interest,
            risk_multiplier=risk_mult,
            filter_buy=filter_buy,
            filter_sell=filter_sell,
            metadata={
                "binance_symbol": binance_symbol,
                "funding_rate_pct": round(funding_rate * 100, 5),
                "long_short_ratio": round(long_short_ratio, 3),
                "open_interest": open_interest,
            },
        )

        self._cache[symbol] = (now, signal)
        self.logger.info(
            "FundingRate %s: rate=%.5f%% (%s) l/s=%.2f OI=%.0f mult=%.2f buy_filter=%s sell_filter=%s",
            symbol,
            funding_rate * 100,
            label,
            long_short_ratio,
            open_interest,
            risk_mult,
            filter_buy,
            filter_sell,
        )
        return signal
