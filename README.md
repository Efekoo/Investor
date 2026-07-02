# Investor — Algorithmic Trading Bot

[![CI](https://github.com/Efekoo/Investor/actions/workflows/ci.yml/badge.svg)](https://github.com/Efekoo/Investor/actions/workflows/ci.yml)

A modular, production-oriented cryptocurrency trading bot in **Python**. Runs in **paper**, **live**, or **backtest** mode with a strict safety-gate design: real orders require two explicit confirmations and can be halted instantly with a kill-switch file.

## Architecture

```
crypto_bot/
├── core/            # Exchange layer (ccxt), data manager, execution engine,
│                    # risk module, portfolio tracker, regime detection,
│                    # funding-rate & sentiment inputs, performance metrics
├── strategies/      # 8 pluggable strategies on a common base class:
│                    # EMA, RSI, EMA+RSI+Volume, Bollinger mean-reversion,
│                    # breakout, grid, SuperTrend, VWAP
├── backtest/        # Backtest engine with fee / slippage / delay simulation
│                    # + Optuna-based parameter optimization
├── database/        # SQLite persistence: orders, trades, balances, logs,
│                    # crash-state recovery
├── config/          # YAML settings (mode, symbols, timeframes, risk limits)
└── tests/           # 19 pytest tests
```

**Key design points**

- **Exchange layer** over `ccxt` with market-metadata caching and precision-aware order validation
- **Risk module**: position sizing, stop-loss / take-profit, trailing stop, break-even moves, per-symbol limits, and a daily circuit breaker
- **Execution engine**: paper & live modes with dynamic slippage, maker/taker fees, partial-fill simulation, and retries
- **Telegram integration**: notifications plus `/status`, `/balance`, `/positions` commands

## Safety model

Live trading is opt-in at three levels — the default build cannot place a real order:

1. `app.mode` in `config/settings.yaml` must be set to `live`
2. `LIVE_TRADING_CONFIRM=YES` and `LIVE_RUNTIME_CONFIRM` must both be set in `.env`
3. If a `runtime/KILL_SWITCH` file exists, the bot stops immediately

Use exchange API keys with **trading-only permissions and withdrawals disabled**, and an exchange-side IP whitelist in production.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env          # add exchange API keys (paper mode works without)
python -m crypto_bot.main
```

Or with Docker:

```bash
docker compose up --build
```

## Tests

```bash
pytest crypto_bot/tests
```

19 tests covering strategies, risk rules, backtest accounting, and the optimizer — run on every push via GitHub Actions.

## Disclaimer

Educational project. Nothing here is financial advice; use live mode at your own risk.
