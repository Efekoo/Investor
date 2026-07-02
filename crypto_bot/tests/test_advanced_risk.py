import pytest
from crypto_bot.core.risk import RiskConfig, RiskManager

class MockLogger:
    def info(self, msg): pass
    def warning(self, msg): print(f"WARNING: {msg}")
    def error(self, msg): print(f"ERROR: {msg}")

@pytest.fixture
def risk_manager():
    config = RiskConfig(
        risk_per_trade=0.01,
        stop_loss_pct=0.05, # Stop mesafesi artırıldı
        take_profit_pct=0.10,
        max_open_trades=3,
        max_daily_loss_pct=0.05,
        max_weekly_loss_pct=0.10,
        use_kelly=True,
        kelly_fraction=0.1, # Daha küçük bir fraction
        max_trade_size_quote=1000000.0
    )
    return RiskManager(config, MockLogger())

def test_kelly_position_sizing(risk_manager):
    balance = 100000.0 # Bakiye artırıldı
    entry_price = 100.0
    stop_price = 95.0 # %5 risk mesafesi
    
    # %60 kazanma oranı ile Kelly hesaplaması
    size_high_win = risk_manager.calculate_position_size(balance, entry_price, stop_price, win_rate=0.6)
    
    # %40 kazanma oranı ile Kelly hesaplaması
    size_low_win = risk_manager.calculate_position_size(balance, entry_price, stop_price, win_rate=0.4)
    
    assert size_high_win > size_low_win
    assert size_low_win > 0
    print(f"Size High Win: {size_high_win}, Size Low Win: {size_low_win}")

def test_weekly_kill_switch(risk_manager):
    initial_balance = 10000.0
    risk_manager.start_week(initial_balance)
    
    # Haftalık %10 kayıp limiti var. Bakiyeyi 8500'e düşürelim (%15 kayıp)
    is_triggered = risk_manager.check_circuit_breaker(8500.0)
    
    assert is_triggered == True
    assert risk_manager.circuit_breaker_triggered == True

def test_partial_tp_targets(risk_manager):
    entry_price = 100.0
    targets = risk_manager.get_partial_tp_targets("BUY", entry_price)
    
    assert len(targets) == 2
    assert targets[0]["price"] == pytest.approx(101.5)
    assert targets[1]["price"] == pytest.approx(103.0)
    assert targets[0]["close_pct"] == 0.5
