"""hpo — Optuna 调 TSP 启发式超参（n_starts / max_passes / first_improve）。

目标：验证实例上 tour length（越小越好，与全局 score 语义一致）。
固定 sampler seed，结果可复现。
"""
from __future__ import annotations

from typing import Any, Dict

import optuna

from ..data.generators import gen_problem
from ..solvers.tsp_heuristic import tour_length
import numpy as np

# 固定验证实例：与 demo 数据集不同 seed，避免调参泄漏到评测
_VAL_SEED = 9042
_VAL_N = 40


def make_objective(val_seed: int = _VAL_SEED, val_n: int = _VAL_N):
    problem = gen_problem("tsp", val_n, val_seed)
    dist = np.asarray(problem.payload["dist"], dtype=np.int64)

    def objective(trial: optuna.Trial) -> float:
        n_starts = trial.suggest_int("n_starts", 1, 8)
        max_passes = trial.suggest_int("max_passes", 1, 40)
        first_improve = trial.suggest_categorical("first_improve", [True, False])
        best = float("inf")
        for s in range(min(n_starts, val_n)):
            from ..solvers.tsp_heuristic import _nearest_neighbor_start, _two_opt
            tour = _nearest_neighbor_start(dist, s)
            tour = _two_opt(tour, dist, max_passes, bool(first_improve))
            best = min(best, float(tour_length(tour, dist)))
        return best

    return objective


def tune(trials: int = 20, seed: int = 42,
         val_seed: int = _VAL_SEED, val_n: int = _VAL_N) -> Dict[str, Any]:
    """返回 best_params + best_value + study 统计。"""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler,
                                study_name="optiforge-tsp-hpo")
    study.optimize(make_objective(val_seed, val_n), n_trials=trials,
                   show_progress_bar=False)
    return {
        "best_params": dict(study.best_params),
        "best_value": float(study.best_value),
        "n_trials": len(study.trials),
        "direction": study.direction.name.lower(),
        "val_instance": {"kind": "tsp", "n": val_n, "seed": val_seed},
    }
