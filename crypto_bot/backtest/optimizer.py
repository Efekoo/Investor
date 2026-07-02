from __future__ import annotations

import pandas as pd
import optuna
from typing import Any, Callable, Type, Dict
from crypto_bot.backtest.engine import BacktestEngine
from crypto_bot.backtest.monte_carlo import MonteCarloResult, run_monte_carlo
from crypto_bot.strategies.base_strategy import Strategy

class StrategyOptimizer:
    def __init__(
        self,
        strategy_class: Type[Strategy],
        data: pd.DataFrame,
        initial_balance: float = 10000.0,
        metric: str = "sharpe_ratio"
    ):
        self.strategy_class = strategy_class
        self.data = data
        self.initial_balance = initial_balance
        self.metric = metric

    def run_optuna_search(self, param_ranges: dict[str, tuple[Any, Any, str]], n_trials: int = 50) -> dict[str, Any]:
        """
        Optuna kullanarak Bayesian Optimizasyonu yapar.
        param_ranges: {'param_name': (min, max, type)}
        type: 'int', 'float', 'categorical'
        """
        def objective(trial):
            params = {}
            for name, (low, high, ptype) in param_ranges.items():
                if ptype == "int":
                    params[name] = trial.suggest_int(name, low, high)
                elif ptype == "float":
                    params[name] = trial.suggest_float(name, low, high)
                elif ptype == "categorical":
                    params[name] = trial.suggest_categorical(name, low) # low is a list here
            
            # EMA stratejisi için mantıksal kontrol (örneğin fast < slow)
            if "fast_period" in params and "slow_period" in params:
                if params["fast_period"] >= params["slow_period"]:
                    return -1.0 # Geçersiz kombinasyon için ceza puanı

            strategy = self.strategy_class(params)
            engine = BacktestEngine(strategy, self.initial_balance)
            result = engine.run(self.data)
            
            # Hedef metrik (varsayılan Sharpe Ratio)
            score = result.metrics.get(self.metric, 0.0)
            
            # Aşırı az işlem yapan stratejileri elemek için (overfitting engelleme)
            if result.metrics.get("total_trades", 0) < 5:
                return score * 0.1
                
            return score

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)

        return {
            "best_params": study.best_params,
            "best_metric": study.best_value,
            "study": study
        }

    def run_grid_search(self, param_grid: dict[str, list[Any]]) -> dict[str, Any]:
        """Basit bir Grid Search optimizasyonu."""
        import itertools
        
        keys = param_grid.keys()
        values = param_grid.values()
        combinations = list(itertools.product(*values))
        
        best_metric = -float("inf")
        best_params = {}
        results = []

        print(f"Running Grid Search with {len(combinations)} combinations...")

        for combo in combinations:
            params = dict(zip(keys, combo))
            strategy = self.strategy_class(params)
            engine = BacktestEngine(strategy, self.initial_balance)
            result = engine.run(self.data)
            
            current_metric = result.metrics.get(self.metric, 0.0)
            
            results.append({
                "params": params,
                "metric": current_metric,
                "total_trades": result.metrics["total_trades"]
            })

            if current_metric > best_metric:
                best_metric = current_metric
                best_params = params

        return {
            "best_params": best_params,
            "best_metric": best_metric,
            "all_results": sorted(results, key=lambda x: x["metric"], reverse=True)
        }

    def walk_forward_analysis(
        self, 
        param_ranges: dict[str, Any], 
        n_folds: int = 3,
        train_ratio: float = 0.7,
        use_optuna: bool = True,
        n_trials: int = 30
    ) -> list[dict[str, Any]]:
        """Walk-Forward Analysis (WFA) gerçekleştirir."""
        fold_size = len(self.data) // n_folds
        wf_results = []

        for i in range(n_folds):
            start_idx = i * fold_size
            end_idx = (i + 1) * fold_size
            fold_data = self.data.iloc[start_idx:end_idx]
            
            split_point = int(len(fold_data) * train_ratio)
            train_data = fold_data.iloc[:split_point]
            test_data = fold_data.iloc[split_point:]

            print(f"Fold {i+1}: Training on {len(train_data)} rows, Testing on {len(test_data)} rows")

            # Format tespiti: liste değerleri → grid search, tuple(3) → optuna
            first_val = next(iter(param_ranges.values()))
            is_grid_format = isinstance(first_val, list)

            # Training
            optimizer = StrategyOptimizer(self.strategy_class, train_data, self.initial_balance, self.metric)
            if use_optuna and not is_grid_format:
                train_result = optimizer.run_optuna_search(param_ranges, n_trials=n_trials)
            else:
                train_result = optimizer.run_grid_search(param_ranges)
            
            best_params = train_result["best_params"]

            # Validation: Test best params on out-of-sample data
            strategy = self.strategy_class(best_params)
            engine = BacktestEngine(strategy, self.initial_balance)
            test_result = engine.run(test_data)

            wf_results.append({
                "fold": i + 1,
                "best_params": best_params,
                "train_metric": train_result["best_metric"],
                "test_metric": test_result.metrics.get(self.metric, 0.0),
                "test_trades": test_result.metrics["total_trades"],
                "test_drawdown": test_result.metrics.get("max_drawdown", 0.0)
            })

        return wf_results

    def monte_carlo_robustness(
        self,
        n_simulations: int = 1_000,
        seed: int | None = 42,
    ) -> MonteCarloResult:
        """
        Tüm veriyi backtest eder ve sonuçlara Monte Carlo simülasyonu uygular.

        Kullanım:
          optimizer = StrategyOptimizer(MyStrategy, data, 10000)
          mc = optimizer.monte_carlo_robustness(n_simulations=1000)
          print(mc.summary["prob_profitable"])

        Yorumlama:
          prob_profitable > 0.75  → Strateji sağlam
          prob_drawdown_exceed_20pct < 0.10 → Drawdown riski kabul edilebilir
          max_drawdown_p5 < -0.30 → Kötü senaryoda %30'dan fazla düşebilir, dikkat
        """
        strategy = self.strategy_class({})  # Varsayılan parametrelerle çalıştır
        engine = BacktestEngine(strategy, self.initial_balance)
        result = engine.run(self.data)

        if result.trades.empty:
            raise ValueError("Backtest sıfır işlem üretti, Monte Carlo çalıştırılamaz")

        return run_monte_carlo(
            trades=result.trades,
            initial_balance=self.initial_balance,
            n_simulations=n_simulations,
            seed=seed,
        )
