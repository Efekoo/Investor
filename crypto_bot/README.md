# Crypto Bot

Production-oriented modular cryptocurrency trading bot with:
- Exchange layer (`ccxt`) with market metadata caching, precision-aware order validation, and order book support
- Data manager (OHLCV -> pandas, cache, multi-symbol/timeframe support, websocket streaming fallback)
- Pluggable strategy system (EMA, RSI, EMA+RSI+Volume)
- Risk module (position sizing, SL/TP, trailing stop, break-even, per-symbol limits, daily circuit breaker)
- Execution engine (paper/live, dynamic slippage, maker/taker fees, partial-fill simulation, retries)
- Portfolio tracker
- SQLite persistence (orders, trades, balances, logs) + crash state recovery
- Backtest engine with fees/slippage/delay simulation
- Telegram notifications + commands (`/status`, `/balance`, `/positions`)
- Docker deployment
- Unit/integration-oriented tests

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
cp .env.example .env
python -m crypto_bot.main
```

Set `app.mode` in `config/settings.yaml` to:
- `paper` for simulated execution
- `live` for real orders
- `backtest` for offline simulation

Live trading safety gate:
- keep `LIVE_TRADING_CONFIRM=NO` by default
- set `LIVE_TRADING_CONFIRM=YES` only when you intentionally want live execution
- bot will stop if kill switch file (`runtime/KILL_SWITCH`) exists

## Security notes
- Keep API keys in `.env` only.
- Configure exchange API keys with trading-only permissions and withdrawals disabled.
- Use exchange-side IP whitelist in production.
