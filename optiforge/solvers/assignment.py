"""分配问题求解器：CP-SAT（SOTA 精确）+ scipy Hungarian（兜底，亦为精确）。"""
from __future__ import annotations

import time

import numpy as np

from ..core.types import Problem, Solution


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


def solve_assignment_ortools(problem: Problem, timeout_s: float = 10.0) -> Solution:
    if not ORTOOLS_CPSAT_OK:
        return Solution(
            problem_kind="assignment", solver_name="assignment_ortools", backend="ortools",
            objective=float("inf"), status="error", runtime_s=0.0,
            error=f"ortools unavailable: {_CPSAT_IMPORT_ERR}",
        )
    t0 = time.perf_counter()
    try:
        cost = problem.payload["cost"]
        n = problem.n
        model = cp_model.CpModel()
        x = [[model.NewBoolVar(f"x_{i}_{j}") for j in range(n)] for i in range(n)]
        for i in range(n):
            model.AddExactlyOne(x[i])
        for j in range(n):
            model.AddExactlyOne([x[i][j] for i in range(n)])
        model.Minimize(sum(int(cost[i][j]) * x[i][j] for i in range(n) for j in range(n)))
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = max(0.5, timeout_s)
        solver.parameters.num_workers = 8
        status_code = solver.Solve(model)
        if status_code not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return Solution(
                problem_kind="assignment", solver_name="assignment_ortools", backend="ortools",
                objective=float("inf"), status="error",
                runtime_s=time.perf_counter() - t0, error=f"status={status_code}",
            )
        col_ind = [next(j for j in range(n) if solver.Value(x[i][j])) for i in range(n)]
        status = "optimal" if status_code == cp_model.OPTIMAL else "feasible"
        return Solution(
            problem_kind="assignment", solver_name="assignment_ortools", backend="ortools",
            objective=float(solver.ObjectiveValue()), status=status,
            runtime_s=time.perf_counter() - t0,
            detail={"col_ind": col_ind},
        )
    except Exception as exc:  # noqa: BLE001
        return Solution(
            problem_kind="assignment", solver_name="assignment_ortools", backend="ortools",
            objective=float("inf"), status="error",
            runtime_s=time.perf_counter() - t0, error=str(exc),
        )


# ---------- scipy Hungarian（兜底，仍为精确最优） ----------
def solve_assignment_scipy(problem: Problem, timeout_s: float = 10.0) -> Solution:
    from scipy.optimize import linear_sum_assignment as _sk_lsa  # 改名防遮蔽

    t0 = time.perf_counter()
    try:
        cost = np.asarray(problem.payload["cost"], dtype=np.int64)
        row_ind, col_ind = _sk_lsa(cost)
        objective = float(cost[row_ind, col_ind].sum())
        return Solution(
            problem_kind="assignment", solver_name="assignment_scipy", backend="scipy",
            objective=objective, status="optimal",
            runtime_s=time.perf_counter() - t0,
            detail={"col_ind": [int(c) for c in col_ind]},
        )
    except Exception as exc:  # noqa: BLE001
        return Solution(
            problem_kind="assignment", solver_name="assignment_scipy", backend="scipy",
            objective=float("inf"), status="error",
            runtime_s=time.perf_counter() - t0, error=str(exc),
        )
