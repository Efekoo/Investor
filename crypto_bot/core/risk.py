from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Dict, List


@dataclass
class RiskConfig:
    risk_per_trade: float
    stop_loss_pct: float
    take_profit_pct: float
    max_open_trades: int
    max_daily_loss_pct: float
    max_weekly_loss_pct: float = 0.15  # Yeni: Haftalık maksimum kayıp
    trailing_stop_pct: float = 0.008
    break_even_trigger_pct: float = 0.01
    max_risk_per_symbol_pct: float = 0.25
    volatility_position_scale: float = 10.0
    max_trade_size_quote: float = 2500
    max_consecutive_losses: int = 3
    volatility_spike_threshold: float = 0.03
    slippage_breaker_threshold: float = 0.01
    use_kelly: bool = False  # Yeni: Kelly Criterion kullanımı
    kelly_fraction: float = 0.5  # Yeni: "Fractional Kelly" (daha güvenli)
    partial_tp_levels: List[Dict[str, float]] = field(default_factory=lambda: [
        {"target_pct": 0.015, "close_pct": 0.5},  # %1.5 karda %50 kapat
        {"target_pct": 0.03, "close_pct": 1.0}   # %3 karda kalan %100 kapat
    ])
    # Korelasyon filtresi
    correlated_pairs: List[List[str]] = field(default_factory=lambda: [
        ["BTC/USDT", "ETH/USDT"],  # Yüksek korelasyon grubu
    ])
    correlation_penalty: float = 0.5  # Aynı grupta 2. pozisyon için boyut çarpanı


class RiskManager:
    def __init__(self, config: RiskConfig, logger: Any) -> None:
        self.config = config
        self.logger = logger
        self.daily_start_balance: float | None = None
        self.weekly_start_balance: float | None = None
        self.circuit_breaker_triggered = False
        self.consecutive_losses = 0
        self.win_rate = 0.5 # Varsayılan, işlem yapıldıkça güncellenebilir
        self._has_real_winrate = False  # Gerçek performans verisi gelene kadar Kelly devreye girmez
        self._current_day: date | None = None
        self._current_week: tuple[int, int] | None = None  # (ISO yıl, ISO hafta)

    def start_day(self, balance: float) -> None:
        self.daily_start_balance = balance
        self._current_day = datetime.now(timezone.utc).date()
        self.circuit_breaker_triggered = False
        self.consecutive_losses = 0
        if self.weekly_start_balance is None:
            self.start_week(balance)

    def start_week(self, balance: float) -> None:
        self.weekly_start_balance = balance
        iso = datetime.now(timezone.utc).isocalendar()
        self._current_week = (iso[0], iso[1])

    def roll_periods(self, balance: float) -> None:
        """Gün/hafta değiştiyse referans bakiyeleri ve günlük kesiciyi sıfırlar.

        Her döngüde çağrılır; böylece günlük/haftalık kayıp limitleri her yeni
        dönemde taze bir başlangıç bakiyesine göre değerlendirilir.
        """
        now = datetime.now(timezone.utc)
        today = now.date()
        iso = now.isocalendar()
        week = (iso[0], iso[1])

        if self._current_day is None or today != self._current_day:
            self.daily_start_balance = balance
            self._current_day = today
            # Yeni gün → günlük tetiklenmiş kesiciyi temizle (haftalık limit hâlâ check_circuit_breaker'da korunur)
            self.circuit_breaker_triggered = False
        if self._current_week is None or week != self._current_week:
            self.weekly_start_balance = balance
            self._current_week = week

    def register_trade_outcome(self, pnl: float, win_rate: float | None = None) -> None:
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        
        if win_rate is not None:
            self.win_rate = win_rate
            self._has_real_winrate = True

        if self.consecutive_losses >= self.config.max_consecutive_losses:
            self.logger.warning(f"Max consecutive losses ({self.consecutive_losses}) reached. Triggering breaker.")
            self.circuit_breaker_triggered = True

    def check_market_breakers(self, volatility: float, slippage_pct: float) -> bool:
        if volatility >= self.config.volatility_spike_threshold:
            self.logger.warning(f"Volatility spike detected: {volatility:.4f}")
            self.circuit_breaker_triggered = True
        if slippage_pct >= self.config.slippage_breaker_threshold:
            self.logger.warning(f"High slippage detected: {slippage_pct:.4f}")
            self.circuit_breaker_triggered = True
        return self.circuit_breaker_triggered

    def check_circuit_breaker(self, current_balance: float) -> bool:
        if self.daily_start_balance is None:
            self.daily_start_balance = current_balance
        if self.weekly_start_balance is None:
            self.weekly_start_balance = current_balance

        daily_loss = max(0.0, self.daily_start_balance - current_balance)
        weekly_loss = max(0.0, self.weekly_start_balance - current_balance)

        if self.daily_start_balance > 0 and daily_loss / self.daily_start_balance >= self.config.max_daily_loss_pct:
            self.logger.error("Daily loss limit reached!")
            self.circuit_breaker_triggered = True
        
        if self.weekly_start_balance > 0 and weekly_loss / self.weekly_start_balance >= self.config.max_weekly_loss_pct:
            self.logger.error("Weekly loss limit reached!")
            self.circuit_breaker_triggered = True

        return self.circuit_breaker_triggered

    def calculate_position_size(
        self,
        balance: float,
        entry_price: float,
        stop_price: float,
        volatility: float = 0.0,
        win_rate: float | None = None,
        leverage: float = 1.0,
    ) -> float:
        """Kelly Criterion veya sabit risk kullanarak pozisyon büyüklüğü hesaplar.

        Kaldıraç risk miktarını (stop'ta kaybedilecek tutar) değiştirmez; yalnızca
        teminat başına açılabilecek notional'ı büyütür. Kaldıraçlı modda
        max_trade_size_quote teminata (margin) uygulanır.
        """
        leverage = max(1.0, leverage)
        effective_win_rate = win_rate if win_rate is not None else self.win_rate
        
        # Risk miktarını belirle.
        # Kelly yalnızca gerçek performans verisi biriktikten sonra devreye girer;
        # aksi halde sabit risk_per_trade ile başlanır (varsayılan 0.5 ile aşırı boyutlanmayı önler).
        use_kelly_now = self.config.use_kelly and (self._has_real_winrate or win_rate is not None)
        if use_kelly_now and effective_win_rate > 0:
            # Kelly % = W - [(1-W) / R] 
            # Burada R (Risk/Reward) varsayılan 2.0 kabul edilebilir veya TP/SL'den hesaplanabilir
            tp_dist = self.config.take_profit_pct
            sl_dist = self.config.stop_loss_pct
            reward_risk_ratio = tp_dist / sl_dist if sl_dist > 0 else 2.0
            kelly_pct = effective_win_rate - ((1 - effective_win_rate) / reward_risk_ratio)
            risk_pct = max(0.0, kelly_pct * self.config.kelly_fraction)
        else:
            risk_pct = self.config.risk_per_trade

        risk_amount = balance * risk_pct
        per_unit_risk = abs(entry_price - stop_price)
        
        if per_unit_risk <= 0 or entry_price <= 0:
            return 0.0
            
        raw_size = risk_amount / per_unit_risk
        
        # Volatiliteye göre ölçeklendirme (Yüksek volatilite = Küçük pozisyon)
        volatility_adjustment = 1.0 / (1.0 + (volatility * self.config.volatility_position_scale))
        volatility_size = raw_size * volatility_adjustment
        
        max_affordable = balance * leverage / entry_price
        max_by_trade_cap = (
            self.config.max_trade_size_quote * leverage / entry_price
            if self.config.max_trade_size_quote > 0 else max_affordable
        )
        
        final_size = max(0.0, min(volatility_size, max_affordable, max_by_trade_cap))
        return final_size

    def get_stop_take_prices(
        self,
        side: str,
        entry_price: float,
        atr: float | None = None,
        adx: float | None = None,
    ) -> tuple[float, float]:
        """
        ATR + ADX tabanlı dinamik stop-loss ve take-profit hesaplar.

        ADX momentum scaling:
          ADX < 20  → Zayıf trend, TP çarpanını küçült (erken çık)
          ADX 20-40 → Normal, standart TP
          ADX 40-60 → Güçlü trend, TP'yi uzat
          ADX > 60  → Çok güçlü, maksimum TP uzatma

        Bu sayede momentum güçlüyken karı uzun koşar,
        zayıf trendde erken alarak whipsaw zararını azaltır.
        """
        # ADX momentum çarpanı (TP için)
        if adx is not None and adx > 0:
            if adx < 20:
                adx_tp_mult = 0.75   # Zayıf trend → erkenden al
            elif adx < 40:
                adx_tp_mult = 1.0    # Normal
            elif adx < 60:
                adx_tp_mult = 1.4    # Güçlü trend → uzat
            else:
                adx_tp_mult = 1.7    # Çok güçlü → maksimum uzatma
        else:
            adx_tp_mult = 1.0

        if atr is not None and atr > 0:
            # ATR tabanlı dinamik seviyeler
            atr_multiplier_sl = 2.0
            atr_multiplier_tp = 4.0 * adx_tp_mult

            if side.upper() == "BUY":
                stop = entry_price - (atr * atr_multiplier_sl)
                take = entry_price + (atr * atr_multiplier_tp)
            else:
                stop = entry_price + (atr * atr_multiplier_sl)
                take = entry_price - (atr * atr_multiplier_tp)
        else:
            # Sabit yüzde (Fallback) + ADX scaling
            tp_pct = self.config.take_profit_pct * adx_tp_mult
            if side.upper() == "BUY":
                stop = entry_price * (1 - self.config.stop_loss_pct)
                take = entry_price * (1 + tp_pct)
            else:
                stop = entry_price * (1 + self.config.stop_loss_pct)
                take = entry_price * (1 - tp_pct)
        return stop, take

    def get_partial_tp_targets(self, side: str, entry_price: float) -> List[Dict[str, float]]:
        """Kademeli kar al seviyelerini hesaplar."""
        targets = []
        for level in self.config.partial_tp_levels:
            if side.upper() == "BUY":
                price = entry_price * (1 + level["target_pct"])
            else:
                price = entry_price * (1 - level["target_pct"])
            targets.append({"price": price, "close_pct": level["close_pct"]})
        return targets

    def correlation_size_multiplier(self, portfolio: Any, new_symbol: str) -> float:
        """
        Yeni işlem açılacak sembol, halihazırda açık pozisyon olan yüksek korelasyonlu bir
        sembolle aynı grupta ise pozisyon boyutunu `correlation_penalty` ile çarpar.

        Örnek: BTC/USDT açık pozisyon varken ETH/USDT için sinyal gelirse boyut %50 küçülür.
        Aynı anda hiç korelasyonlu açık pozisyon yoksa 1.0 döner (etki yok).
        """
        open_symbols = set(portfolio.positions.keys())
        for group in self.config.correlated_pairs:
            if new_symbol not in group:
                continue
            # Bu gruptaki diğer semboller arasında açık pozisyon var mı?
            others_open = [s for s in group if s != new_symbol and s in open_symbols]
            if others_open:
                self.logger.debug(
                    "Correlation penalty applied for %s (open: %s), mult=%.2f",
                    new_symbol, others_open, self.config.correlation_penalty,
                )
                return self.config.correlation_penalty
        return 1.0

    def _symbol_exposure(self, portfolio: Any, symbol: str, mark_price: float) -> float:
        pos = portfolio.positions.get(symbol)
        if not pos:
            return 0.0
        return pos.qty * mark_price

    def validate_trade(
        self,
        portfolio: Any,
        symbol: str,
        balance: float,
        current_balance: float,
        mark_price: float,
    ) -> tuple[bool, str]:
        if self.circuit_breaker_triggered or self.check_circuit_breaker(current_balance):
            return False, "Circuit breaker active"
        if portfolio.is_open(symbol):
            return False, f"Open position exists for {symbol}"
        if portfolio.open_count() >= self.config.max_open_trades:
            return False, "Max open trades reached"
        if balance <= 0:
            return False, "No available balance"
        symbol_exposure = self._symbol_exposure(portfolio, symbol, mark_price)
        if current_balance > 0 and symbol_exposure / current_balance > self.config.max_risk_per_symbol_pct:
            return False, f"Symbol risk limit exceeded for {symbol}"
        return True, "OK"

    def update_protective_levels(self, position: Any, current_price: float) -> tuple[float, float]:
        """Stop-loss ve take-profit seviyelerini dinamik olarak günceller (Trailing SL & TP)."""
        stop_loss = position.stop_loss
        take_profit = position.take_profit
        
        if position.side.upper() == "BUY":
            pnl_pct = (current_price - position.entry_price) / position.entry_price
            
            # 1. Başabaş noktasına çekme (Break-even)
            if pnl_pct >= self.config.break_even_trigger_pct:
                stop_loss = max(stop_loss, position.entry_price)
            
            # 2. Trailing Stop Loss
            trailing_sl_candidate = current_price * (1 - self.config.trailing_stop_pct)
            stop_loss = max(stop_loss, trailing_sl_candidate)
            
            # 3. Trailing Take Profit (Kârı Takip Et!)
            # Eğer fiyat hedef TP'nin üzerine çıktıysa, TP'yi de yukarı çek
            if current_price > take_profit:
                # Yeni TP, mevcut fiyatın biraz üzerinde (Trendi sürmek için)
                take_profit = current_price * (1 + (self.config.take_profit_pct * 0.2))
                self.logger.info(f"Trailing TP moved up to {take_profit:.2f} for {position.symbol}")

        else: # SELL/SHORT Pozisyonu
            pnl_pct = (position.entry_price - current_price) / position.entry_price
            if pnl_pct >= self.config.break_even_trigger_pct:
                stop_loss = min(stop_loss, position.entry_price)
            
            trailing_sl_candidate = current_price * (1 + self.config.trailing_stop_pct)
            stop_loss = min(stop_loss, trailing_sl_candidate)
            
            if current_price < take_profit:
                take_profit = current_price * (1 - (self.config.take_profit_pct * 0.2))
                self.logger.info(f"Trailing TP moved down to {take_profit:.2f} for {position.symbol}")
            
        return stop_loss, take_profit
