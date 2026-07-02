from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict

try:
    import requests as _requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False


_FNG_URL = "https://api.alternative.me/fng/?limit=1&format=json"
_FNG_CACHE_TTL = 300  # saniye — aynı veriyi 5 dakika boyunca yeniden kullan


@dataclass
class SentimentScore:
    score: float  # -1.0 (Aşırı Negatif) ile 1.0 (Aşırı Pozitif) arası
    label: str    # BULLISH, BEARISH, NEUTRAL
    fng_value: int  # 0-100 arası ham Fear & Greed değeri
    fng_classification: str  # "Extreme Fear" .. "Extreme Greed"
    source_count: int
    metadata: Dict[str, Any]


class SentimentAnalyzer:
    def __init__(self, config: Dict[str, Any], logger: Any) -> None:
        self.config = config
        self.logger = logger
        self.enabled = config.get("enabled", False)
        self._last_score = SentimentScore(0.0, "NEUTRAL", 50, "Neutral", 0, {})
        self._last_fetch_ts: float = 0.0

    # ------------------------------------------------------------------
    # Fear & Greed Index (alternative.me/fng)
    # ------------------------------------------------------------------
    def _fetch_fng(self) -> tuple[int, str]:
        """
        alternative.me Fear & Greed API'sini çağırır.
        0 = Extreme Fear, 100 = Extreme Greed
        Dönüş: (ham_değer, sınıflandırma)
        """
        if not _REQUESTS_AVAILABLE:
            self.logger.debug("requests not installed, skipping FnG fetch")
            return 50, "Neutral"
        try:
            resp = _requests.get(_FNG_URL, timeout=5)
            resp.raise_for_status()
            data = resp.json()
            entry = data["data"][0]
            return int(entry["value"]), str(entry["value_classification"])
        except Exception as exc:
            self.logger.warning("Fear & Greed fetch failed: %s", exc)
            return 50, "Neutral"

    def _fng_to_score(self, fng_value: int) -> tuple[float, str]:
        """
        0-100 arasındaki ham Fear & Greed değerini -1.0..+1.0 score ve
        BULLISH/BEARISH/NEUTRAL etiketine dönüştürür.

        Eşikler:
          0-24   → Extreme Fear  → BEARISH  (-1.0 .. -0.5)
          25-44  → Fear          → BEARISH  (-0.5 .. -0.1)
          45-55  → Neutral       → NEUTRAL  (~0.0)
          56-74  → Greed         → BULLISH  (+0.1 .. +0.5)
          75-100 → Extreme Greed → BULLISH  (+0.5 .. +1.0)
        """
        normalized = (fng_value - 50) / 50.0  # -1.0 .. +1.0
        if fng_value <= 24:
            label = "BEARISH"
        elif fng_value <= 44:
            label = "BEARISH"
        elif fng_value <= 55:
            label = "NEUTRAL"
        elif fng_value <= 74:
            label = "BULLISH"
        else:
            label = "BULLISH"
        return round(normalized, 3), label

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fetch_latest_sentiment(self, symbol: str) -> SentimentScore:
        """Fear & Greed Index'i çeker ve SentimentScore olarak döner."""
        if not self.enabled:
            return self._last_score

        now = time.time()
        if now - self._last_fetch_ts < _FNG_CACHE_TTL:
            return self._last_score  # önbellekten sun

        fng_value, fng_classification = self._fetch_fng()
        score_val, label = self._fng_to_score(fng_value)
        self._last_fetch_ts = now

        self._last_score = SentimentScore(
            score=score_val,
            label=label,
            fng_value=fng_value,
            fng_classification=fng_classification,
            source_count=1,
            metadata={"source": "alternative.me/fng", "symbol": symbol},
        )
        self.logger.info(
            "Fear & Greed: %d (%s) → score=%.3f label=%s",
            fng_value, fng_classification, score_val, label,
        )
        return self._last_score

    def get_risk_multiplier(self) -> float:
        """
        Fear & Greed değerine göre pozisyon risk çarpanı döner.
          Extreme Fear  (≤24)  → 0.70  (piyasa paniği, riskini azalt)
          Fear          (25-44) → 0.85
          Neutral       (45-55) → 1.00
          Greed         (56-74) → 1.10  (trend takibi, hafif artır)
          Extreme Greed (≥75)  → 0.80  (aşırı ısınma, tekrar azalt)
        """
        if not self.enabled:
            return 1.0
        v = self._last_score.fng_value
        if v <= 24:
            return 0.70
        if v <= 44:
            return 0.85
        if v <= 55:
            return 1.00
        if v <= 74:
            return 1.10
        return 0.80  # Extreme Greed — zirve riski

    def should_filter_trade(self, side: str) -> bool:
        """
        Duyarlılık sinyalle şiddetle çelişiyorsa işlemi engelle.
          Extreme Fear  → BUY filtrele (≤24 için uzun girmek çok riskli)
          Extreme Greed → SHORT filtrele (≥75)
        """
        if not self.enabled:
            return False
        v = self._last_score.fng_value
        if side.upper() == "BUY" and v <= 20:
            self.logger.warning(
                "BUY filtered: Extreme Fear (FnG=%d)", v
            )
            return True
        if side.upper() == "SELL" and v >= 80:
            self.logger.warning(
                "SELL filtered: Extreme Greed (FnG=%d)", v
            )
            return True
        return False
