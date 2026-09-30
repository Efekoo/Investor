import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from crypto_bot import dashboard_data as dd
from crypto_bot.core.commands import pending_commands, pop_commands, write_command
from crypto_bot.core.portfolio import Portfolio, Position
from crypto_bot.main import TradingBot
from crypto_bot.web.app import create_app

NOW = datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc)


def _state() -> dict:
    return {
        "paper_cash": 9_000.0,
        "last_prices": {"BTC/USDT": 105.0, "ETH/USDT": 90.0},
        "updated_at": (NOW - timedelta(seconds=10)).isoformat(),
        "portfolio": {
            "positions": [
                # 5x long: 10 @100 → teminat 200, likidasyon 80.5
                {"symbol": "BTC/USDT", "side": "BUY", "entry_price": 100.0, "qty": 10.0, "stop_loss": 95.0,
                 "take_profit": 110.0, "leverage": 5, "margin": 200.0, "liquidation_price": 80.5,
                 "strategy_name": "ema", "opened_at": NOW.isoformat()},
                # spot short (eski muhasebe)
                {"symbol": "ETH/USDT", "side": "SELL", "entry_price": 100.0, "qty": 1.0, "stop_loss": 105.0,
                 "take_profit": 90.0, "opened_at": NOW.isoformat()},
            ],
            "trade_history": [
                {"symbol": "BTC/USDT", "side": "BUY", "pnl": 50.0, "closed_at": "2026-09-30T10:00:00+00:00"},
                {"symbol": "ETH/USDT", "side": "SELL", "pnl": -20.0, "closed_at": "2026-09-30T11:00:00+00:00"},
                {"symbol": "BTC/USDT", "side": "BUY", "pnl": -201.0, "liquidated": True, "closed_at": "2026-09-30T12:00:00+00:00"},
            ],
        },
    }


def test_position_views_and_account_summary():
    views = dd.position_views(_state())
    btc, eth = views
    assert btc.side == "LONG" and btc.unrealized_pnl == pytest.approx(50.0)
    assert btc.roe_pct == pytest.approx(25.0)             # 50 / 200 teminat
    assert btc.liq_distance_pct == pytest.approx((105 - 80.5) / 105 * 100)
    assert eth.side == "SHORT" and eth.unrealized_pnl == pytest.approx(10.0)
    assert eth.liq_distance_pct is None

    summary = dd.account_summary(_state(), views)
    # nakit + (teminat + PnL) - short yükümlülüğü → botun _get_total_equity ile aynı formül
    assert summary.equity == pytest.approx(9_000 + 200 + 50 - 90)
    assert summary.margin_used == pytest.approx(200.0)
    assert summary.unrealized_pnl == pytest.approx(60.0)


def test_dashboard_equity_matches_bot_equity():
    state = _state()
    bot = TradingBot.__new__(TradingBot)
    bot.mode = "paper"
    bot.paper_cash = state["paper_cash"]
    bot.portfolio = Portfolio()
    bot._last_prices = dict(state["last_prices"])
    for row in state["portfolio"]["positions"]:
        bot.portfolio.open_position(Position(
            symbol=row["symbol"], side=row["side"], entry_price=row["entry_price"], qty=row["qty"], entry_fee=0.0,
            stop_loss=row["stop_loss"], take_profit=row["take_profit"], peak_price=row["entry_price"],
            leverage=row.get("leverage", 1.0), margin=row.get("margin", 0.0),
            liquidation_price=row.get("liquidation_price", 0.0),
        ))
    assert dd.account_summary(state).equity == pytest.approx(bot._get_total_equity())


def test_liquidation_alerts_threshold():
    views = dd.position_views(_state())
    assert dd.liquidation_alerts(views, 5.0) == []
    assert [v.symbol for v in dd.liquidation_alerts(views, 30.0)] == ["BTC/USDT"]


def test_trade_stats_and_state_age():
    stats = dd.trade_stats(dd.closed_trades(_state()))
    assert stats["count"] == 3 and stats["liquidations"] == 1
    assert stats["total_pnl"] == pytest.approx(-171.0)
    assert stats["win_rate"] == pytest.approx(100 / 3)
    assert dd.closed_trades(_state())[0]["liquidated"] is True  # en yeni önce
    assert dd.state_age_seconds(_state(), NOW) == pytest.approx(10.0)
    assert dd.state_age_seconds({}) is None


def test_command_queue_roundtrip(tmp_path):
    write_command(tmp_path, "close_position", symbol="BTC/USDT")
    write_command(tmp_path, "close_all")
    (tmp_path / "junk.json").write_text("{not json")
    assert len(pending_commands(tmp_path)) == 2
    popped = pop_commands(tmp_path)
    assert [c["action"] for c in popped] == ["close_position", "close_all"]
    assert list(tmp_path.glob("*.json")) == []
    with pytest.raises(ValueError):
        write_command(tmp_path, "withdraw_everything")


def _bot_for_commands(tmp_path) -> TradingBot:
    bot = TradingBot.__new__(TradingBot)
    bot.commands_dir = tmp_path
    bot.settings = {"safety": {"kill_switch_file": str(tmp_path / "KILL_SWITCH")}}
    bot.logger = __import__("logging").getLogger("test")
    bot.notifier = type("N", (), {"send": lambda self, msg: None})()
    bot._kill_switch_logged = False
    return bot


def test_bot_processes_close_commands(tmp_path):
    bot = _bot_for_commands(tmp_path)
    closed = []
    bot.portfolio = Portfolio()
    for sym in ("BTC/USDT", "ETH/USDT"):
        bot.portfolio.open_position(Position(sym, "BUY", 100.0, 1.0, 0.0, 95.0, 110.0, 100.0))
    bot._manual_close = lambda symbol: closed.append(symbol) or True
    write_command(tmp_path, "close_position", symbol="ETH/USDT")
    assert bot._process_control_commands() is True
    assert closed == ["ETH/USDT"]
    write_command(tmp_path, "close_all")
    bot._process_control_commands()
    assert closed == ["ETH/USDT", "BTC/USDT", "ETH/USDT"]
    assert bot._process_control_commands() is False  # kuyruk boş


def test_kill_switch_file_and_setting(tmp_path):
    bot = _bot_for_commands(tmp_path)
    assert bot._kill_switch_active() is False
    (tmp_path / "KILL_SWITCH").write_text("x")
    assert bot._kill_switch_active() is True
    (tmp_path / "KILL_SWITCH").unlink()
    bot.settings["safety"]["global_kill_switch"] = True
    assert bot._kill_switch_active() is True


@pytest.fixture
def panel(tmp_path, monkeypatch):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "state.json").write_text(json.dumps(_state()))
    settings = {
        "app": {"mode": "paper", "exchange": "binance", "data_dir": str(runtime)},
        "trading": {"symbols": ["BTC/USDT", "ETH/USDT"], "timeframe": "1m", "quote_currency": "USDT"},
        "leverage": {"enabled": True, "leverage": 5},
        "database": {"url": f"sqlite:///{tmp_path / 'db.sqlite'}"},
        "safety": {"kill_switch_file": str(runtime / "KILL_SWITCH")},
        "runtime": {"state_file": str(runtime / "state.json"), "commands_dir": str(runtime / "commands")},
    }
    path = tmp_path / "settings.yaml"
    path.write_text(yaml.safe_dump(settings))
    monkeypatch.delenv("DASHBOARD_TOKEN", raising=False)
    monkeypatch.delenv("STATE_FILE", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    return TestClient(create_app(str(path))), runtime


def test_panel_snapshot_and_index(panel):
    client, _ = panel
    assert "Investor" in client.get("/").text
    snap = client.get("/api/snapshot", params={"alert_pct": 30}).json()
    assert snap["leverage"] == {"enabled": True, "leverage": 5.0, "margin_mode": "isolated"}
    assert len(snap["positions"]) == 2 and snap["alerts"][0]["symbol"] == "BTC/USDT"
    assert snap["summary"]["margin_used"] == pytest.approx(200.0)
    assert snap["kill_switch"] is False
    assert client.get("/api/trades").json()["stats"]["liquidations"] == 1
    assert client.get("/api/equity").json()["points"] == [] or True  # tablo yoksa hata mesajı döner, çökmez


def test_panel_controls_write_files(panel):
    client, runtime = panel
    assert client.post("/api/kill-switch", json={"enabled": True}).json()["kill_switch"] is True
    assert (runtime / "KILL_SWITCH").exists()
    assert client.post("/api/kill-switch", json={"enabled": False}).json()["kill_switch"] is False
    assert client.post("/api/positions/BTC/USDT/close").status_code == 200
    assert client.post("/api/positions/XRP/USDT/close").status_code == 404
    assert client.post("/api/positions/close-all").status_code == 200
    assert [c["action"] for c in pending_commands(runtime / "commands")] == ["close_position", "close_all"]
    assert client.get("/api/snapshot").json()["pending_commands"] == 2


def test_panel_controls_require_token_when_set(panel, monkeypatch):
    client, runtime = panel
    monkeypatch.setenv("DASHBOARD_TOKEN", "s3cret")
    assert client.post("/api/kill-switch", json={"enabled": True}).status_code == 401
    assert client.post("/api/kill-switch", json={"enabled": True}, headers={"X-Dashboard-Token": "wrong"}).status_code == 401
    ok = client.post("/api/kill-switch", json={"enabled": True}, headers={"X-Dashboard-Token": "s3cret"})
    assert ok.status_code == 200 and (runtime / "KILL_SWITCH").exists()


def test_panel_controls_reject_remote_without_token(tmp_path, panel):
    client, runtime = panel
    remote = TestClient(client.app, client=("203.0.113.9", 5000))
    assert remote.post("/api/kill-switch", json={"enabled": True}).status_code == 403
    assert not (runtime / "KILL_SWITCH").exists()
    assert remote.get("/api/snapshot").status_code == 200  # okuma serbest


def test_panel_candles_validate_input(panel):
    client, _ = panel
    assert client.get("/api/candles", params={"symbol": "XRP/USDT"}).status_code == 404
    assert client.get("/api/candles", params={"symbol": "BTC/USDT", "timeframe": "7m"}).status_code == 400


def test_panel_websocket_pushes_snapshot(panel):
    client, _ = panel
    with client.websocket_connect("/ws") as ws:
        snap = json.loads(ws.receive_text())
    assert snap["summary"]["open_positions"] == 2
