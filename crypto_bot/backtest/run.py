from __future__ import annotations

"""Backtest CLI — validate strategies on real history before going live.

Examples:
    # Full comparison on 1 year of Binance 1h data (downloads & caches):
    python -m crypto_bot.backtest.run --symbols BTC/USDT,ETH/USDT --timeframe 1h --days 365

    # Offline smoke test with synthetic data:
    python -m crypto_bot.backtest.run --synthetic

    # Walk-forward analysis for one strategy (out-of-sample proof):
    python -m crypto_bot.backtest.run --symbols BTC/USDT --timeframe 1h --days 365 \
        --walk-forward supertrend
"""

import argparse
import sys
from pathlib import Path

import yaml

from crypto_bot.backtest.data_loader import generate_synthetic_ohlcv, load_or_download
from crypto_bot.backtest.report import STRATEGY_REGISTRY, ReportConfig, build_comparison_report, save_report

DEFAULT_SETTINGS = Path(__file__).resolve().parents[1] / "config" / "settings.yaml"

# Walk-forward parameter search spaces (optuna format: (low, high, type))
WFA_PARAM_RANGES: dict[str, dict] = {
    "ema": {"fast_period": (5, 30, "int"), "slow_period": (20, 80, "int")},
    "rsi": {"period": (7, 28, "int"), "oversold": (20, 40, "int"), "overbought": (60, 80, "int")},
    "supertrend": {"atr_period": (7, 21, "int"), "multiplier": (1.5, 4.5, "float")},
    "bollinger_mean_reversion": {"period": (10, 30, "int"), "std_dev": (1.0, 3.0, "float")},
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Strategy validation backtests")
    p.add_argument("--exchange", default="binance")
    p.add_argument("--symbols", default="BTC/USDT,ETH/USDT,SOL/USDT")
    p.add_argument("--timeframe", default="1h")
    p.add_argument("--days", type=int, default=365)
    p.add_argument("--strategies", default="all", help="'all' or comma-separated names")
    p.add_argument("--balance", type=float, default=10_000)
    p.add_argument("--fee", type=float, default=0.001, help="taker fee (0.001 = Binance spot)")
    p.add_argument("--slippage", type=float, default=0.0005)
    p.add_argument("--risk", type=float, default=0.02, help="risk per trade (0 = all-in)")
    p.add_argument("--sl", type=float, default=0.02, help="stop-loss pct (0 = disabled)")
    p.add_argument("--tp", type=float, default=0.04, help="take-profit pct (0 = disabled)")
    p.add_argument("--trailing", type=float, default=0.0, help="trailing stop pct (0 = disabled)")
    p.add_argument("--settings", default=str(DEFAULT_SETTINGS))
    p.add_argument("--out", default="runtime/reports")
    p.add_argument("--cache-dir", default="runtime/data")
    p.add_argument("--synthetic", action="store_true", help="use synthetic data (no network)")
    p.add_argument("--force-refresh", action="store_true")
    p.add_argument("--regime-filter", action="store_true",
                   help="only allow long entries in bull regime (price>EMA200, EMA50>EMA200, EMA200 rising)")
    p.add_argument("--compare-regime", action="store_true",
                   help="run everything twice (filter off/on) and print a side-by-side comparison")
    p.add_argument("--walk-forward", metavar="STRATEGY", default=None,
                   help=f"run WFA for one strategy: {', '.join(WFA_PARAM_RANGES)}")
    p.add_argument("--wfa-folds", type=int, default=4)
    p.add_argument("--wfa-trials", type=int, default=30)
    return p.parse_args(argv)


def load_strategy_params(settings_path: str) -> dict[str, dict]:
    try:
        with open(settings_path, "r", encoding="utf-8") as f:
            settings = yaml.safe_load(f)
        return {k.lower(): v for k, v in settings["strategy"]["params"].items()}
    except (OSError, KeyError):
        return {}


def get_data(args: argparse.Namespace) -> dict:
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    data_by_symbol = {}
    if args.synthetic:
        for i, symbol in enumerate(symbols):
            data_by_symbol[symbol] = generate_synthetic_ohlcv(
                n=3000, timeframe=args.timeframe, seed=42 + i
            )
        print(f"Using synthetic data for {len(symbols)} symbols (offline mode)")
        return data_by_symbol

    for symbol in symbols:
        print(f"Loading {symbol} {args.timeframe} ({args.days}d) from {args.exchange}...")
        df = load_or_download(
            args.exchange, symbol, args.timeframe, days=args.days,
            cache_dir=args.cache_dir, force_refresh=args.force_refresh,
        )
        if df.empty:
            print(f"  WARNING: no data for {symbol}, skipping")
            continue
        print(f"  {len(df)} candles ({df['timestamp'].iloc[0]} -> {df['timestamp'].iloc[-1]})")
        data_by_symbol[symbol] = df
    return data_by_symbol


def run_walk_forward(args: argparse.Namespace, data_by_symbol: dict, params: dict) -> None:
    import optuna

    from crypto_bot.backtest.optimizer import StrategyOptimizer

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    name = args.walk_forward.lower()
    cls = STRATEGY_REGISTRY.get(name)
    ranges = WFA_PARAM_RANGES.get(name)
    if cls is None or ranges is None:
        print(f"Walk-forward not supported for '{name}'. Options: {', '.join(WFA_PARAM_RANGES)}")
        sys.exit(1)

    MIN_OOS_TRADES = 10

    for symbol, data in data_by_symbol.items():
        print(f"\n=== Walk-forward: {name} on {symbol} ({args.wfa_folds} folds) ===")
        optimizer = StrategyOptimizer(cls, data, args.balance, metric="sharpe_ratio")
        results = optimizer.walk_forward_analysis(
            ranges, n_folds=args.wfa_folds, n_trials=args.wfa_trials, use_optuna=True
        )
        for r in results:
            print(
                f"  Fold {r['fold']}: test_return={r.get('test_return', 0.0):+.2%} "
                f"(B&H {r.get('test_buy_hold', 0.0):+.2%}) trades={r['test_trades']} "
                f"dd={r['test_drawdown']:.2%} params={r['best_params']}"
            )

        # --- honest OOS aggregation ---
        total_trades = sum(r["test_trades"] for r in results)
        returns = [r.get("test_return", 0.0) for r in results]
        compound = 1.0
        for x in returns:
            compound *= 1 + x
        compound -= 1
        positive_folds = sum(1 for x in returns if x > 0)

        # parameter stability: normalized spread of each numeric param across folds
        stability_notes = []
        keys = {k for r in results for k in r["best_params"]}
        for k in sorted(keys):
            vals = [r["best_params"][k] for r in results if k in r["best_params"]
                    and isinstance(r["best_params"][k], (int, float))]
            if len(vals) >= 2:
                mean = sum(vals) / len(vals)
                spread = (max(vals) - min(vals)) / max(abs(mean), 1e-9)
                stability_notes.append(f"{k}: {min(vals)}–{max(vals)} (spread {spread:.0%})")

        print(f"\n  --- OOS summary for {symbol} ---")
        print(f"  Total OOS trades: {total_trades}")
        print(f"  Compound OOS return: {compound:+.2%} | positive folds: {positive_folds}/{len(results)}")
        for note in stability_notes:
            print(f"  Param range across folds -> {note}")

        if total_trades < MIN_OOS_TRADES:
            print(f"  VERDICT: YETERSIZ VERI — {total_trades} OOS islem ile hukum verilemez "
                  f"(en az {MIN_OOS_TRADES} gerekir). Daha uzun --days veya daha az fold deneyin.")
        elif compound > 0 and positive_folds >= max(2, int(0.6 * len(results))):
            print("  VERDICT: UMUT VERICI — OOS karli ve fold'lar tutarli. "
                  "Parametre araligi da darsa paper trading'e alinabilir.")
        else:
            print("  VERDICT: KANIT YOK — OOS getiri zayif veya fold'lar tutarsiz (muhtemel overfit).")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    data_by_symbol = get_data(args)
    if not data_by_symbol:
        print("No data available; aborting.")
        sys.exit(1)

    strategy_params = load_strategy_params(args.settings)

    if args.walk_forward:
        run_walk_forward(args, data_by_symbol, strategy_params)
        return

    names = list(STRATEGY_REGISTRY) if args.strategies == "all" else [
        s.strip().lower() for s in args.strategies.split(",")
    ]
    def make_cfg(regime: bool) -> ReportConfig:
        return ReportConfig(
            initial_balance=args.balance,
            taker_fee_pct=args.fee,
            slippage_pct=args.slippage,
            stop_loss_pct=args.sl or None,
            take_profit_pct=args.tp or None,
            trailing_stop_pct=args.trailing or None,
            risk_per_trade=args.risk or None,
            regime_filter=regime,
        )

    def print_report(report) -> None:
        cols = ["strategy", "symbol", "total_return_pct", "buy_hold_return_pct",
                "sharpe_ratio", "profit_factor", "max_drawdown", "total_trades", "verdict"]
        with_fmt = report[cols].copy()
        for c in ["total_return_pct", "buy_hold_return_pct", "max_drawdown"]:
            with_fmt[c] = (with_fmt[c] * 100).round(1)
        print(with_fmt.to_string(index=False))

    if args.compare_regime:
        print(f"\n[1/2] Filter OFF: {len(names)} strategies x {len(data_by_symbol)} symbols...")
        base = build_comparison_report(data_by_symbol, strategy_params, names, args.timeframe, make_cfg(False))
        print(f"[2/2] Filter ON (bull regime only)...")
        filt = build_comparison_report(data_by_symbol, strategy_params, names, args.timeframe, make_cfg(True))
        merged = base.merge(
            filt, on=["strategy", "symbol"], suffixes=("_off", "_on")
        )[["strategy", "symbol", "total_return_pct_off", "total_return_pct_on",
           "max_drawdown_off", "max_drawdown_on", "total_trades_off", "total_trades_on",
           "profit_factor_off", "profit_factor_on"]]
        for c in [c for c in merged.columns if "pct" in c or "drawdown" in c]:
            merged[c] = (merged[c] * 100).round(1)
        merged["improvement"] = (merged["total_return_pct_on"] - merged["total_return_pct_off"]).round(1)
        print("\n=== Regime filter comparison (returns & drawdowns in %) ===")
        print(merged.sort_values("improvement", ascending=False).to_string(index=False))
        csv_path, html_path = save_report(merged, args.out, label="regime_comparison")
        n_better = int((merged["improvement"] > 0).sum())
        print(f"\nFilter improved {n_better}/{len(merged)} strategy-symbol pairs.")
        print(f"Report saved:\n  {csv_path}\n  {html_path}")
        return

    cfg = make_cfg(args.regime_filter)
    label = " (regime filter ON)" if args.regime_filter else ""
    print(f"\nRunning {len(names)} strategies x {len(data_by_symbol)} symbols{label}...")
    report = build_comparison_report(data_by_symbol, strategy_params, names, args.timeframe, cfg)
    if report.empty:
        print("No results produced.")
        sys.exit(1)

    csv_path, html_path = save_report(report, args.out)
    print_report(report)
    print(f"\nReport saved:\n  {csv_path}\n  {html_path}")


if __name__ == "__main__":
    main()
