from __future__ import annotations

"""Strategy comparison report: run every strategy over every symbol and
compare against buy & hold, so you can see which strategies actually earn
after fees and slippage — before risking real money."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Type

import pandas as pd

from crypto_bot.backtest.data_loader import resample_ohlcv, timeframe_to_ms
from crypto_bot.backtest.engine import BacktestEngine
from crypto_bot.backtest.filters import bull_regime_filter
from crypto_bot.strategies.base_strategy import Strategy
from crypto_bot.strategies.bollinger_mean_reversion_strategy import BollingerMeanReversionStrategy
from crypto_bot.strategies.breakout_strategy import VolatilityBreakoutStrategy
from crypto_bot.strategies.ema_rsi_volume_strategy import EmaRsiVolumeStrategy
from crypto_bot.strategies.ema_strategy import EMACrossoverStrategy
from crypto_bot.strategies.grid_strategy import GridStrategy
from crypto_bot.strategies.rsi_strategy import RSIStrategy
from crypto_bot.strategies.supertrend_strategy import SupertrendStrategy
from crypto_bot.strategies.vwap_strategy import VWAPReversionStrategy

STRATEGY_REGISTRY: dict[str, Type[Strategy]] = {
    "ema": EMACrossoverStrategy,
    "rsi": RSIStrategy,
    "ema_rsi_volume": EmaRsiVolumeStrategy,
    "bollinger_mean_reversion": BollingerMeanReversionStrategy,
    "volatility_breakout": VolatilityBreakoutStrategy,
    "grid": GridStrategy,
    "vwap_reversion": VWAPReversionStrategy,
    "supertrend": SupertrendStrategy,
}


@dataclass
class ReportConfig:
    initial_balance: float = 10_000.0
    taker_fee_pct: float = 0.001       # Binance spot default (no BNB discount)
    slippage_pct: float = 0.0005
    execution_delay_candles: int = 1
    stop_loss_pct: float | None = 0.02
    take_profit_pct: float | None = 0.04
    trailing_stop_pct: float | None = None
    risk_per_trade: float | None = 0.02
    min_trades_for_verdict: int = 10
    regime_filter: bool = False
    leverage: float = 1.0


def run_strategy_backtest(
    strategy: Strategy,
    data: pd.DataFrame,
    timeframe: str,
    cfg: ReportConfig,
) -> dict[str, Any]:
    """Backtest a single strategy on one symbol's data (MTF-aware)."""
    ptf = str(strategy.params.get("primary_timeframe", timeframe))
    ctf = str(strategy.params.get("confirm_timeframe", ptf))

    htf_data = None
    if timeframe_to_ms(ctf) > timeframe_to_ms(timeframe):
        htf_data = resample_ohlcv(data, ctf)

    engine = BacktestEngine(
        strategy,
        initial_balance=cfg.initial_balance,
        taker_fee_pct=cfg.taker_fee_pct,
        slippage_pct=cfg.slippage_pct,
        execution_delay_candles=cfg.execution_delay_candles,
        stop_loss_pct=cfg.stop_loss_pct,
        take_profit_pct=cfg.take_profit_pct,
        trailing_stop_pct=cfg.trailing_stop_pct,
        risk_per_trade=cfg.risk_per_trade,
        timeframe=timeframe,
        leverage=cfg.leverage,
    )
    result = engine.run(
        data,
        htf_data=htf_data,
        primary_tf=ptf if htf_data is not None else None,
        confirm_tf=ctf if htf_data is not None else None,
        entry_filter=bull_regime_filter(data) if cfg.regime_filter else None,
    )
    return result.metrics


def _verdict(row: pd.Series, cfg: ReportConfig) -> str:
    if row["total_trades"] < cfg.min_trades_for_verdict:
        return "INSUFFICIENT_TRADES"
    profitable = row["total_return_pct"] > 0
    beats_bh = row["total_return_pct"] > row["buy_hold_return_pct"]
    robust = row["profit_factor"] >= 1.2 and row["max_drawdown"] > -0.30
    if profitable and beats_bh and robust:
        return "PROMISING"
    if profitable and robust:
        return "PROFITABLE_BUT_LAGS_BH"
    return "NOT_VIABLE"


def build_comparison_report(
    data_by_symbol: dict[str, pd.DataFrame],
    strategy_params: dict[str, dict],
    strategy_names: list[str],
    timeframe: str,
    cfg: ReportConfig | None = None,
) -> pd.DataFrame:
    """Run strategy x symbol matrix and return a tidy comparison table."""
    cfg = cfg or ReportConfig()
    rows = []
    for name in strategy_names:
        cls = STRATEGY_REGISTRY.get(name.lower())
        if cls is None:
            continue
        params = strategy_params.get(name.lower(), {})
        for symbol, data in data_by_symbol.items():
            strategy = cls(dict(params))
            metrics = run_strategy_backtest(strategy, data, timeframe, cfg)
            rows.append({"strategy": name, "symbol": symbol, **metrics})

    report = pd.DataFrame(rows)
    if report.empty:
        return report
    report["verdict"] = report.apply(lambda r: _verdict(r, cfg), axis=1)
    return report.sort_values(["strategy", "symbol"]).reset_index(drop=True)


def save_report(report: pd.DataFrame, out_dir: str | Path, label: str = "backtest_report") -> tuple[Path, Path]:
    """Persist the report as CSV + a readable HTML table."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"{label}_{stamp}.csv"
    html_path = out_dir / f"{label}_{stamp}.html"
    report.to_csv(csv_path, index=False)

    display = report.copy()
    for col in ["total_return_pct", "buy_hold_return_pct", "win_rate", "max_drawdown", "exposure_pct"]:
        if col in display.columns:
            display[col] = (display[col] * 100).round(2).astype(str) + "%"
    for col in ["final_equity", "fees_paid", "avg_trade_pnl", "sharpe_ratio", "profit_factor"]:
        if col in display.columns:
            display[col] = display[col].round(3)

    style = """
    <style>
      body { font-family: -apple-system, Segoe UI, sans-serif; margin: 24px; }
      table { border-collapse: collapse; width: 100%; font-size: 13px; }
      th, td { border: 1px solid #ddd; padding: 6px 10px; text-align: right; }
      th { background: #1f2937; color: #fff; position: sticky; top: 0; }
      td:first-child, td:nth-child(2) { text-align: left; font-weight: 600; }
      tr:nth-child(even) { background: #f7f7f7; }
      .PROMISING { background: #d1fae5 !important; }
      .NOT_VIABLE { background: #fee2e2 !important; }
    </style>"""
    rows_html = []
    for _, r in display.iterrows():
        cls = str(r.get("verdict", ""))
        cells = "".join(f"<td>{v}</td>" for v in r.values)
        rows_html.append(f'<tr class="{cls}">{cells}</tr>')
    header = "".join(f"<th>{c}</th>" for c in display.columns)
    html = (
        f"<html><head><meta charset='utf-8'>{style}</head><body>"
        f"<h2>Strategy Comparison Report — {stamp}</h2>"
        f"<p>Verdicts: <b>PROMISING</b> = profitable, beats buy&hold, PF&ge;1.2, DD&gt;-30% · "
        f"<b>NOT_VIABLE</b> = loses money or fails robustness · trades&lt;10 = insufficient data.</p>"
        f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(rows_html)}</tbody></table>"
        f"</body></html>"
    )
    html_path.write_text(html, encoding="utf-8")
    return csv_path, html_path
