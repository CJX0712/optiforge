"""0-1 背包求解器：CP-SAT 精确解（SOTA）+ 精确 DP / 贪心（离线兜底）。

CP-SAT 与 DP 均为 exact（status=optimal），greedy 为 heuristic。
注意：n 与 capacity 过大时 DP 自动让位（容量上限保护），由注册器排优先级。
"""
from __future__ import annotations

import time

from ..core.types import Problem, Solution

_DP_CAPACITY_LIMIT = 400_000  # n*capacity 超过此值时 DP 拒绝求解（内存保护）


# ---------- CP-SAT (ortools) ----------
try:
    from ortools.sat.python import cp_model
    ORTOOLS_CPSAT_OK = True
    _CPSAT_IMPORT_ERR = ""
except Exception as _exc:  # noqa: BLE001
    ORTOOLS_CPSAT_OK = False
    _CPSAT_IMPORT_ERR = str(_exc)


def available_ortools_cpsat() -> bool:
    return ORTOOLS_CPSAT_OK


def solve_knapsack_ortools(problem: Problem, timeout_s: float = 10.0) -> Solution:
    if not ORTOOLS_CPSAT_OK:
        return Solution(
            problem_kind="knapsack", solver_name="knapsack_ortools", backend="ortools",
            objective=float("-inf"), status="error", runtime_s=0.0,
            error=f"ortools unavailable: {_CPSAT_IMPORT_ERR}",
        )
    t0 = time.perf_counter()
    try:
        values = problem.payload["values"]
        weights = problem.payload["weights"]
        capacity = problem.payload["capacity"]
        n = problem.n
        model = cp_model.CpModel()
        x = [model.NewBoolVar(f"x{i}") for i in range(n)]
        model.Add(sum(weights[i] * x[i] for i in range(n)) <= capacity)
        model.Maximize(sum(values[i] * x[i] for i in range(n)))
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = max(0.5, timeout_s)
        status_code = solver.Solve(model)
        if status_code not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return Solution(
                problem_kind="knapsack", solver_name="knapsack_ortools", backend="ortools",
                objective=float("-inf"), status="error",
                runtime_s=time.perf_counter() - t0, error=f"status={status_code}",
            )
        selected = [i for i in range(n) if solver.Value(x[i])]
        status = "optimal" if status_code == cp_model.OPTIMAL else "feasible"
        return Solution(
            problem_kind="knapsack", solver_name="knapsack_ortools", backend="ortools",
            objective=float(solver.ObjectiveValue()), status=status,
            runtime_s=time.perf_counter() - t0,
            detail={"selected": selected,
                    "total_weight": sum(weights[i] for i in selected)},
        )
    except Exception as exc:  # noqa: BLE001
        return Solution(
            problem_kind="knapsack", solver_name="knapsack_ortools", backend="ortools",
            objective=float("-inf"), status="error",
            runtime_s=time.perf_counter() - t0, error=str(exc),
        )


# ---------- 精确 DP（兜底，仍为 optimal） ----------
def solve_knapsack_dp(problem: Problem, timeout_s: float = 10.0) -> Solution:
    t0 = time.perf_counter()
    try:
        values = problem.payload["values"]
        weights = problem.payload["weights"]
        capacity = int(problem.payload["capacity"])
        n = problem.n
        if n * capacity > _DP_CAPACITY_LIMIT:
            return Solution(
                problem_kind="knapsack", solver_name="knapsack_dp", backend="numpy",
                objective=float("-inf"), status="error", runtime_s=0.0,
                error=f"n*capacity={n * capacity} exceeds DP limit {_DP_CAPACITY_LIMIT}",
            )
        # 一维滚动数组；take 恢复选品
        dp = [0] * (capacity + 1)
        take = [[False] * (capacity + 1) for _ in range(n)]
        for i in range(n):
            w, v = weights[i], values[i]
            for c in range(capacity, w - 1, -1):
                cand = dp[c - w] + v
                if cand > dp[c]:
                    dp[c] = cand
                    take[i][c] = True
        # 回溯
        selected: list[int] = []
        c = capacity
        for i in range(n - 1, -1, -1):
            if take[i][c]:
                selected.append(i)
                c -= weights[i]
        selected.reverse()
        return Solution(
            problem_kind="knapsack", solver_name="knapsack_dp", backend="numpy",
            objective=float(dp[capacity]), status="optimal",
            runtime_s=time.perf_counter() - t0,
            detail={"selected": selected,
                    "total_weight": sum(weights[i] for i in selected)},
        )
    except Exception as exc:  # noqa: BLE001
        return Solution(
            problem_kind="knapsack", solver_name="knapsack_dp", backend="numpy",
            objective=float("-inf"), status="error",
            runtime_s=time.perf_counter() - t0, error=str(exc),
        )


# ---------- 贪心（快基线） ----------
def solve_knapsack_greedy(problem: Problem, timeout_s: float = 10.0) -> Solution:
    t0 = time.perf_counter()
    try:
        values = problem.payload["values"]
        weights = problem.payload["weights"]
        capacity = problem.payload["capacity"]
        order = sorted(range(problem.n),
                       key=lambda i: values[i] / weights[i], reverse=True)
        selected: list[int] = []
        total_w = 0
        total_v = 0
        for i in order:
            if total_w + weights[i] <= capacity:
                selected.append(i)
                total_w += weights[i]
                total_v += values[i]
        return Solution(
            problem_kind="knapsack", solver_name="knapsack_greedy", backend="numpy",
            objective=float(total_v), status="heuristic",
            runtime_s=time.perf_counter() - t0,
            detail={"selected": selected, "total_weight": total_w},
        )
    except Exception as exc:  # noqa: BLE001
        return Solution(
            problem_kind="knapsack", solver_name="knapsack_greedy", backend="numpy",
            objective=float("-inf"), status="error",
            runtime_s=time.perf_counter() - t0, error=str(exc),
        )
