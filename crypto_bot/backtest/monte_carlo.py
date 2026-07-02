from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from crypto_bot.backtest.metrics import max_drawdown, profit_factor, sharpe_ratio, win_rate


@dataclass
class MonteCarloResult:
    n_simulations: int
    # Her simülasyonun özet istatistikleri
    final_equity_dist: list[float]
    sharpe_dist: list[float]
    max_drawdown_dist: list[float]
    profit_factor_dist: list[float]
    win_rate_dist: list[float]
    # Yüzdelik istatistikler
    summary: dict[str, Any]


def run_monte_carlo(
    trades: pd.DataFrame,
    initial_balance: float = 10_000.0,
    n_simulations: int = 1_000,
    seed: int | None = 42,
) -> MonteCarloResult:
    """
    Gerçek backtest işlemlerinin sırasını karıştırarak N adet simülasyon çalıştırır.

    Amaç: Strateji gerçekten mi karlı, yoksa tarihsel sıraya mı bağımlı?
    Sağlam bir strateji tüm permütasyonlarda benzer sonuç vermeli.

    Girdi:
      trades: BacktestEngine.run() çıktısından SELL işlemleri (her satır bir kapalı işlem)
              Zorunlu sütun: "pnl"
      initial_balance: Başlangıç bakiyesi
      n_simulations: Kaç permütasyon çalıştırılacak

    Çıktı:
      MonteCarloResult — dağılım verileri + özet istatistikler
    """
    if trades.empty or "pnl" not in trades.columns:
        raise ValueError("trades DataFrame boş veya 'pnl' sütunu yok")

    pnl_values = trades["pnl"].tolist()
    n_trades = len(pnl_values)

    if n_trades < 2:
        raise ValueError(f"Monte Carlo için en az 2 işlem gerekli, {n_trades} var")

    rng = random.Random(seed)

    final_equities: list[float] = []
    sharpes: list[float] = []
    drawdowns: list[float] = []
    profit_factors: list[float] = []
    win_rates: list[float] = []

    for _ in range(n_simulations):
        shuffled = pnl_values[:]
        rng.shuffle(shuffled)

        # Kümülatif equity eğrisi
        equity = initial_balance
        equity_curve = [equity]
        for pnl in shuffled:
            equity += pnl
            equity_curve.append(equity)

        eq_series = pd.Series(equity_curve)
        ret_series = eq_series.pct_change().dropna()

        # Simülasyon metriklerini hesapla
        sim_trades_df = pd.DataFrame({"pnl": shuffled})
        sim_wr = float((sim_trades_df["pnl"] > 0).sum() / max(len(sim_trades_df), 1))
        sim_pf = profit_factor(sim_trades_df)
        sim_sharpe = sharpe_ratio(ret_series)
        sim_dd = max_drawdown(eq_series)

        final_equities.append(round(equity, 4))
        sharpes.append(round(sim_sharpe, 6))
        drawdowns.append(round(sim_dd, 6))
        profit_factors.append(round(sim_pf, 6))
        win_rates.append(round(sim_wr, 6))

    def _percentiles(data: list[float], label: str) -> dict[str, float]:
        arr = np.array(data)
        return {
            f"{label}_mean": round(float(arr.mean()), 4),
            f"{label}_std": round(float(arr.std()), 4),
            f"{label}_p5": round(float(np.percentile(arr, 5)), 4),
            f"{label}_p25": round(float(np.percentile(arr, 25)), 4),
            f"{label}_p50": round(float(np.percentile(arr, 50)), 4),
            f"{label}_p75": round(float(np.percentile(arr, 75)), 4),
            f"{label}_p95": round(float(np.percentile(arr, 95)), 4),
        }

    summary: dict[str, Any] = {
        "n_simulations": n_simulations,
        "n_trades": n_trades,
        "initial_balance": initial_balance,
        # Karlılık olasılığı: eşit başlangıçtan karlı biten simülasyon oranı
        "prob_profitable": round(float(sum(1 for e in final_equities if e > initial_balance) / n_simulations), 4),
        # Max drawdown 20%'yi aşma olasılığı
        "prob_drawdown_exceed_20pct": round(float(sum(1 for d in drawdowns if d < -0.20) / n_simulations), 4),
        **_percentiles(final_equities, "final_equity"),
        **_percentiles(sharpes, "sharpe"),
        **_percentiles(drawdowns, "max_drawdown"),
        **_percentiles(profit_factors, "profit_factor"),
        **_percentiles(win_rates, "win_rate"),
    }

    return MonteCarloResult(
        n_simulations=n_simulations,
        final_equity_dist=final_equities,
        sharpe_dist=sharpes,
        max_drawdown_dist=drawdowns,
        profit_factor_dist=profit_factors,
        win_rate_dist=win_rates,
        summary=summary,
    )


def print_monte_carlo_report(result: MonteCarloResult) -> None:
    """Monte Carlo sonuçlarını terminale özet olarak yazdırır."""
    s = result.summary
    print(f"\n{'='*55}")
    print(f"  Monte Carlo Robustness Raporu  ({s['n_simulations']} simülasyon)")
    print(f"{'='*55}")
    print(f"  İşlem sayısı        : {s['n_trades']}")
    print(f"  Başlangıç bakiye    : {s['initial_balance']:,.2f}")
    print(f"  Karlılık olasılığı  : {s['prob_profitable']*100:.1f}%")
    print(f"  DD>20% olasılığı    : {s['prob_drawdown_exceed_20pct']*100:.1f}%")
    print(f"\n  --- Final Equity ---")
    print(f"  Ortalama  : {s['final_equity_mean']:,.2f}")
    print(f"  P5  (kötü): {s['final_equity_p5']:,.2f}")
    print(f"  P50 (medyan): {s['final_equity_p50']:,.2f}")
    print(f"  P95 (iyi) : {s['final_equity_p95']:,.2f}")
    print(f"\n  --- Sharpe Ratio ---")
    print(f"  Ortalama  : {s['sharpe_mean']:.3f}")
    print(f"  P5        : {s['sharpe_p5']:.3f}")
    print(f"  P95       : {s['sharpe_p95']:.3f}")
    print(f"\n  --- Max Drawdown ---")
    print(f"  Ortalama  : {s['max_drawdown_mean']*100:.2f}%")
    print(f"  En Kötü P5: {s['max_drawdown_p5']*100:.2f}%")
    print(f"{'='*55}\n")
