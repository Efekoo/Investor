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
├── backtest/        # Backtest engine with fee / slippage / delay simulation,
│                    # intra-candle SL/TP, risk-based sizing, historical data
│                    # loader + Optuna optimization & walk-forward analysis
├── database/        # SQLite persistence: orders, trades, balances, logs,
│                    # crash-state recovery
├── web/             # Trading panel: FastAPI + TradingView Lightweight Charts
├── config/          # YAML settings (mode, symbols, timeframes, risk limits, leverage)
└── tests/           # 28 pytest tests
```

**Key design points**

- **Exchange layer** over `ccxt` with market-metadata caching and precision-aware order validation
- **Risk module**: position sizing, stop-loss / take-profit, trailing stop, break-even moves, per-symbol limits, and a daily circuit breaker
- **Execution engine**: paper & live modes with dynamic slippage, maker/taker fees, partial-fill simulation, and retries
- **Leverage (optional)**: USDT-M perpetual futures with isolated/cross margin; stop-loss is pulled inside the liquidation price, exits are `reduceOnly`, paper mode simulates margin and liquidation
- **Trading panel**: live candles with entry / SL / TP / liquidation lines, margin usage, liquidation-distance alerts, one-click position close and kill switch
- **Telegram integration**: notifications plus `/status`, `/balance`, `/positions` commands

## Safety model

Live trading is opt-in at three levels — the default build cannot place a real order:

1. `app.mode` in `config/settings.yaml` must be set to `live`
2. `LIVE_TRADING_CONFIRM=YES` and `LIVE_RUNTIME_CONFIRM` must both be set in `.env`
3. If a `runtime/KILL_SWITCH` file exists (or the panel's kill switch is on), the bot opens no new positions; open positions keep being managed by their SL/TP

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

## Trading panel

```bash
python -m crypto_bot.web        # http://127.0.0.1:8501
```

The panel reads the bot's `runtime/state.json` and database, and talks back to the bot through `runtime/KILL_SWITCH` and `runtime/commands/` (processed within ~1 s). Controls work from localhost only unless `DASHBOARD_TOKEN` is set in `.env`; with Docker Compose it is served on `127.0.0.1:8502`.

## Strategy validation (do this before going live)

Run every strategy against real historical data and compare with buy & hold — after realistic fees, slippage, and stop-losses:

```bash
# Download 1 year of Binance 1h data (cached in runtime/data/) and compare all strategies
python -m crypto_bot.backtest.run --symbols BTC/USDT,ETH/USDT,SOL/USDT --timeframe 1h --days 365

# Offline smoke test without network
python -m crypto_bot.backtest.run --synthetic

# Walk-forward analysis: optimize on train folds, verify on unseen test folds
python -m crypto_bot.backtest.run --symbols BTC/USDT --timeframe 1h --days 365 --walk-forward supertrend
```

The report (CSV + HTML in `runtime/reports/`) labels each strategy x symbol pair `PROMISING`, `PROFITABLE_BUT_LAGS_BH`, or `NOT_VIABLE` based on return vs. buy & hold, profit factor, and max drawdown. Only promote strategies to paper/live that stay `PROMISING` out-of-sample.

## Tests

```bash
pytest crypto_bot/tests
```

Tests covering strategies, risk rules, backtest accounting, stop-loss/sizing simulation, multi-timeframe lookahead safety, the optimizer, leverage/liquidation accounting, and the panel API — run on every push via GitHub Actions.

## Disclaimer

Educational project. Nothing here is financial advice; use live mode at your own risk.
