"""eval — 统一语义指标 + 参照解（brute force 小规模校验）。"""
from __future__ import annotations

from itertools import permutations
from math import inf
from typing import Optional

import numpy as np

from ..core.errors import EvalError
from ..core.types import Problem, Solution


def optimality_gap(solution: Solution, reference_score: float) -> float:
    """gap% = (score - ref_score) / |ref_score| * 100（score 越小越好，恒可比较）。

    ref_score 必须是同一问题同一语义下的分数（ Solution.score 口径）。
    """
    if solution.status in ("error", "infeasible"):
        return inf
    if reference_score == 0:
        raise EvalError("reference score is zero; gap undefined")
    return (solution.score - reference_score) / abs(reference_score) * 100.0


# ---------- 参照解（仅小规模，用于单测与基准校验） ----------
def brute_force(problem: Problem) -> Optional[float]:
    """小规模暴力最优值，返回自然语义 objective；规模超限返回 None。"""
    if problem.kind == "tsp" and problem.n <= 9:
        dist = np.asarray(problem.payload["dist"], dtype=np.int64)
        best = inf
        for perm in permutations(range(1, problem.n)):
            tour = (0,) + perm
            length = sum(int(dist[tour[i], tour[(i + 1) % problem.n]])
                         for i in range(problem.n))
            best = min(best, length)
        return float(best)
    if problem.kind == "knapsack" and problem.n <= 22:
        values = problem.payload["values"]
        weights = problem.payload["weights"]
        cap = problem.payload["capacity"]
        best = 0
        for mask in range(1 << problem.n):
            w = v = 0
            i = mask
            idx = 0
            while i:
                if i & 1:
                    w += weights[idx]
                    v += values[idx]
                i >>= 1
                idx += 1
            if w <= cap:
                best = max(best, v)
        return float(best)
    if problem.kind == "assignment" and problem.n <= 8:
        cost = np.asarray(problem.payload["cost"], dtype=np.int64)
        best = inf
        for perm in permutations(range(problem.n)):
            best = min(best, float(sum(int(cost[i, perm[i]]) for i in range(problem.n))))
        return float(best)
    return None
