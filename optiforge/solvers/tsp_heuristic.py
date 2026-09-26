"""TSP 启发式求解器（离线兜底）：最近邻多起点 + 2-opt 局部搜索。纯 numpy。

支持 HPO 参数（hpo.tune 注入）：
- n_starts:     最近邻起点数（1~8）
- max_passes:   2-opt 最大扫描轮数（1~40）
- first_improve: True=首个改进即交换，False=全扫描取最优交换
"""
from __future__ import annotations

import time

import numpy as np

from ..core.errors import SolveFailureError
from ..core.types import Problem, Solution


def tour_length(tour: list[int], dist: np.ndarray) -> int:
    total = 0
    n = len(tour)
    for i in range(n):
        total += int(dist[tour[i], tour[(i + 1) % n]])
    return total


def _nearest_neighbor_start(dist: np.ndarray, start: int) -> list[int]:
    n = dist.shape[0]
    visited = np.zeros(n, dtype=bool)
    tour = [start]
    visited[start] = True
    for _ in range(n - 1):
        cur = tour[-1]
        nxt = int(np.argmin(np.where(visited, np.inf, dist[cur])))
        tour.append(nxt)
        visited[nxt] = True
    return tour


def _two_opt(tour: list[int], dist: np.ndarray, max_passes: int,
             first_improve: bool) -> list[int]:
    n = len(tour)
    for _ in range(max_passes):
        improved = False
        for i in range(n - 1):
            a, b = tour[i], tour[(i + 1) % n]
            for j in range(i + 2, n):
                c, d = tour[j], tour[(j + 1) % n]
                if (i == 0 and j == n - 1):  # 同一条边
                    continue
                delta = (dist[a, c] + dist[b, d]) - (dist[a, b] + dist[c, d])
                if delta < -1e-9:
                    tour[i + 1: j + 1] = reversed(tour[i + 1: j + 1])
                    improved = True
                    if first_improve:
                        break
            if first_improve and improved:
                break
        if not improved:
            break
    return tour


def solve_tsp_heuristic(problem: Problem, timeout_s: float = 10.0,
                        n_starts: int = 4, max_passes: int = 20,
                        first_improve: bool = False) -> Solution:
    t0 = time.perf_counter()
    try:
        dist = np.asarray(problem.payload["dist"], dtype=np.int64)
        starts = list(range(min(n_starts, problem.n)))
        best_tour: list[int] | None = None
        best_len = float("inf")
        for s in starts:
            tour = _nearest_neighbor_start(dist, s)
            tour = _two_opt(tour, dist, max_passes, first_improve)
            length = tour_length(tour, dist)
            if length < best_len:
                best_len, best_tour = length, tour
        if best_tour is None:
            raise SolveFailureError("heuristic produced no tour")
        return Solution(
            problem_kind="tsp", solver_name="tsp_heuristic", backend="numpy",
            objective=float(best_len), status="heuristic",
            runtime_s=time.perf_counter() - t0,
            detail={"tour": best_tour},
        )
    except Exception as exc:  # noqa: BLE001 — solve 契约：失败转 error Solution
        return Solution(
            problem_kind="tsp", solver_name="tsp_heuristic", backend="numpy",
            objective=float("inf"), status="error",
            runtime_s=time.perf_counter() - t0, error=str(exc),
        )
