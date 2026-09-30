from __future__ import annotations

"""Trading paneli: FastAPI backend + statik tek sayfa (TradingView Lightweight Charts).

Bot ile paylaşılan tek şey runtime/ dizini ve veritabanıdır:
  - state.json      → pozisyonlar, nakit, son fiyatlar, kapanmış işlemler (okuma)
  - KILL_SWITCH     → varsa bot yeni pozisyon açmaz (oluştur/sil)
  - commands/*.json → pozisyon kapatma komutları (yaz; bot ~1 sn içinde işler)

Güvenlik: kontrol uçları (POST) DASHBOARD_TOKEN ortam değişkeni ayarlıysa
X-Dashboard-Token başlığı ister; ayarlı değilse yalnızca localhost'tan kabul edilir.
"""

import asyncio
import hmac
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml
from fastapi import Depends, FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import create_engine, text

from crypto_bot import dashboard_data as dd
from crypto_bot.core.commands import pending_commands, write_command
from crypto_bot.core.leverage import LeverageConfig

STATIC_DIR = Path(__file__).parent / "static"
LOOPBACK = {"127.0.0.1", "::1", "localhost", "testclient"}
TIMEFRAMES = {"1m", "5m", "15m", "1h", "4h", "1d"}


class KillSwitchBody(BaseModel):
    enabled: bool


class Settings:
    """settings.yaml'dan dashboard'un ihtiyaç duyduğu yolları ve ayarları okur."""

    def __init__(self, path: str | None = None) -> None:
        self.path = Path(path or os.getenv("SETTINGS_FILE", "crypto_bot/config/settings.yaml"))
        self.raw: dict[str, Any] = yaml.safe_load(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}
        runtime = self.raw.get("runtime", {}) or {}
        safety = self.raw.get("safety", {}) or {}
        data_dir = Path((self.raw.get("app", {}) or {}).get("data_dir", "runtime"))
        self.state_file = Path(os.getenv("STATE_FILE", runtime.get("state_file", data_dir / "state.json")))
        self.kill_switch_file = Path(safety.get("kill_switch_file", data_dir / "KILL_SWITCH"))
        self.commands_dir = Path(runtime.get("commands_dir", data_dir / "commands"))
        self.db_url = os.getenv("DATABASE_URL", (self.raw.get("database", {}) or {}).get("url", ""))
        self.leverage = LeverageConfig.from_settings(self.raw)
        self.mode = str((self.raw.get("app", {}) or {}).get("mode", "paper")).lower()
        self.exchange = str((self.raw.get("app", {}) or {}).get("exchange", "binance"))
        self.symbols: list[str] = list((self.raw.get("trading", {}) or {}).get("symbols", []))
        self.timeframe = str((self.raw.get("trading", {}) or {}).get("timeframe", "1m"))


def read_state(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def build_snapshot(cfg: Settings, alert_threshold_pct: float = 5.0) -> dict[str, Any]:
    state = read_state(cfg.state_file)
    views = dd.position_views(state)
    summary = dd.account_summary(state, views)
    trades = dd.closed_trades(state)
    return {
        "mode": cfg.mode,
        "exchange": cfg.exchange,
        "symbols": cfg.symbols,
        "timeframe": cfg.timeframe,
        "leverage": {
            "enabled": cfg.leverage.enabled,
            "leverage": cfg.leverage.effective_leverage,
            "margin_mode": cfg.leverage.margin_mode,
        },
        "state_found": bool(state),
        "state_age_seconds": dd.state_age_seconds(state),
        "kill_switch": cfg.kill_switch_file.exists(),
        "pending_commands": len(pending_commands(cfg.commands_dir)),
        "summary": asdict(summary),
        "positions": [asdict(v) for v in views],
        "alerts": [asdict(v) for v in dd.liquidation_alerts(views, alert_threshold_pct)],
        "last_prices": state.get("last_prices", {}),
        "trade_stats": dd.trade_stats(trades),
    }


def create_app(settings_path: str | None = None) -> FastAPI:
    cfg = Settings(settings_path)
    app = FastAPI(title="Investor Panel", docs_url=None, redoc_url=None)
    app.state.cfg = cfg
    candle_cache: dict[tuple[str, str], tuple[float, list]] = {}
    exchange_holder: dict[str, Any] = {}
    engine_holder: dict[str, Any] = {}

    def require_control(request: Request) -> None:
        token = os.getenv("DASHBOARD_TOKEN", "")
        if token:
            given = request.headers.get("x-dashboard-token", "")
            if not hmac.compare_digest(given, token):
                raise HTTPException(401, "Geçersiz veya eksik panel token'ı")
            return
        host = request.client.host if request.client else ""
        if host not in LOOPBACK:
            raise HTTPException(403, "Kontroller yalnızca localhost'tan kullanılabilir (uzaktan erişim için DASHBOARD_TOKEN ayarlayın)")

    def get_exchange():
        if "client" not in exchange_holder:
            import logging

            from crypto_bot.core.exchange import ExchangeClient, ExchangeConfig

            exchange_holder["client"] = ExchangeClient(
                ExchangeConfig(
                    name=cfg.exchange, api_key="", api_secret="", mode="paper",
                    market_type=cfg.leverage.market_type if cfg.leverage.enabled else "spot",
                    settle_currency=cfg.leverage.settle_currency, max_retries=2,
                ),
                logging.getLogger("dashboard"),
            )
        return exchange_holder["client"]

    def get_engine():
        if "engine" not in engine_holder:
            url = cfg.db_url
            if url.startswith("sqlite:///"):
                Path(url.replace("sqlite:///", "", 1)).parent.mkdir(parents=True, exist_ok=True)
            engine_holder["engine"] = create_engine(url)
        return engine_holder["engine"]

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/snapshot")
    def snapshot(alert_pct: float = Query(5.0, ge=0.1, le=100)) -> dict[str, Any]:
        return build_snapshot(cfg, alert_pct)

    @app.get("/api/trades")
    def trades(limit: int = Query(100, ge=1, le=500)) -> dict[str, Any]:
        rows = dd.closed_trades(read_state(cfg.state_file))
        return {"trades": rows[:limit], "stats": dd.trade_stats(rows)}

    @app.get("/api/equity")
    def equity(max_points: int = Query(1500, ge=10, le=5000)) -> dict[str, Any]:
        if not cfg.db_url:
            return {"points": [], "error": "Veritabanı ayarlı değil"}
        try:
            with get_engine().connect() as conn:
                rows = conn.execute(
                    text("SELECT timestamp, total_balance FROM balance_history ORDER BY timestamp ASC")
                ).all()
        except Exception as exc:
            return {"points": [], "error": f"Veritabanı okunamadı: {exc.__class__.__name__}"}
        step = max(1, len(rows) // max_points)
        points = []
        for ts, value in rows[::step]:
            if isinstance(ts, str):
                from datetime import datetime

                ts = datetime.fromisoformat(ts)
            points.append({"time": int(ts.timestamp()), "value": float(value)})
        # Lightweight Charts artan ve tekil zaman ister
        dedup: dict[int, float] = {p["time"]: p["value"] for p in points}
        return {"points": [{"time": t, "value": v} for t, v in sorted(dedup.items())]}

    @app.get("/api/candles")
    def candles(symbol: str, timeframe: str = "1m", limit: int = Query(300, ge=1, le=1000)) -> dict[str, Any]:
        if symbol not in cfg.symbols:
            raise HTTPException(404, "Bilinmeyen sembol")
        if timeframe not in TIMEFRAMES:
            raise HTTPException(400, "Geçersiz zaman dilimi")
        key = (symbol, timeframe)
        cached = candle_cache.get(key)
        if cached and time.time() - cached[0] < 5 and len(cached[1]) >= limit:
            rows = cached[1]
        else:
            try:
                rows = get_exchange().fetch_ohlcv(symbol, timeframe, limit=limit)
            except Exception as exc:
                raise HTTPException(502, f"Borsa verisi alınamadı: {exc}") from exc
            candle_cache[key] = (time.time(), rows)
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "candles": [
                {"time": int(r[0] // 1000), "open": r[1], "high": r[2], "low": r[3], "close": r[4], "volume": r[5]}
                for r in rows[-limit:]
            ],
        }

    @app.post("/api/kill-switch", dependencies=[Depends(require_control)])
    def kill_switch(body: KillSwitchBody) -> dict[str, Any]:
        path = cfg.kill_switch_file
        if body.enabled:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"panel {time.strftime('%Y-%m-%dT%H:%M:%S')}\n", encoding="utf-8")
        else:
            path.unlink(missing_ok=True)
        return {"kill_switch": path.exists()}

    @app.post("/api/positions/close-all", dependencies=[Depends(require_control)])
    def close_all() -> dict[str, Any]:
        return {"queued": write_command(cfg.commands_dir, "close_all")}

    @app.post("/api/positions/{symbol:path}/close", dependencies=[Depends(require_control)])
    def close_position(symbol: str) -> dict[str, Any]:
        if symbol not in cfg.symbols:
            raise HTTPException(404, "Bilinmeyen sembol")
        return {"queued": write_command(cfg.commands_dir, "close_position", symbol=symbol)}

    @app.get("/api/control-status")
    def control_status(request: Request) -> dict[str, Any]:
        host = request.client.host if request.client else ""
        return {"token_required": bool(os.getenv("DASHBOARD_TOKEN")), "local": host in LOOPBACK}

    @app.websocket("/ws")
    async def ws(websocket: WebSocket) -> None:
        await websocket.accept()
        alert_pct = float(websocket.query_params.get("alert_pct", 5.0))
        last_payload = ""
        try:
            while True:
                payload = json.dumps(build_snapshot(cfg, alert_pct), default=str)
                if payload != last_payload:
                    await websocket.send_text(payload)
                    last_payload = payload
                await asyncio.sleep(1)
        except (WebSocketDisconnect, RuntimeError):
            return

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


app = create_app()
