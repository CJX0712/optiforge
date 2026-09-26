"""求解器注册表 — backend 探测 + 按偏好挑选。

backend 语义：
- auto:     ortools 可用则 ortools+兜底都注册；不可用则只用兜底（零下载可跑）
- ortools:  只注册 ortools 系（不可用则返回空）
- fallback: 只注册 numpy / scipy 兜底系
"""
from __future__ import annotations

from typing import Callable

from ..core.errors import BackendUnavailableError
from .assignment import (available_ortools_cpsat as _cpsat_ok,
                         solve_assignment_ortools, solve_assignment_scipy)
from .knapsack import solve_knapsack_dp, solve_knapsack_greedy, solve_knapsack_ortools
from .tsp_heuristic import solve_tsp_heuristic
from .tsp_ortools import (available_ortools_routing as _routing_ok,
                          solve_tsp_ortools)

SolverFn = Callable[..., "Solution"]  # fn(problem, timeout_s=..., **hpo_params)

SPEC: dict[str, list[tuple[str, SolverFn, str]]] = {
    "tsp": [
        ("tsp_ortools", solve_tsp_ortools, "ortools"),
        ("tsp_heuristic", solve_tsp_heuristic, "numpy"),
    ],
    "knapsack": [
        ("knapsack_ortools", solve_knapsack_ortools, "ortools"),
        ("knapsack_dp", solve_knapsack_dp, "numpy"),
        ("knapsack_greedy", solve_knapsack_greedy, "numpy"),
    ],
    "assignment": [
        ("assignment_ortools", solve_assignment_ortools, "ortools"),
        ("assignment_scipy", solve_assignment_scipy, "scipy"),
    ],
}

_BACKEND_OK = {"ortools": None, "numpy": True, "scipy": True}


def available_ortools() -> bool:
    """ortools 两个子模块（routing / cp-sat）任一可用即视为可用。"""
    return bool(_routing_ok() or _cpsat_ok())


def get_solvers(kind: str, backend: str = "auto") -> list[tuple[str, SolverFn]]:
    """返回该问题类型下可用的 (solver_name, fn) 列表。"""
    if kind not in SPEC:
        raise BackendUnavailableError(f"unknown problem kind: {kind}")
    if backend not in ("auto", "ortools", "fallback"):
        raise BackendUnavailableError(f"unknown backend: {backend}")
    out: list[tuple[str, SolverFn]] = []
    for name, fn, b in SPEC[kind]:
        if backend == "ortools" and b != "ortools":
            continue
        if backend == "fallback" and b == "ortools":
            continue
        if b == "ortools" and not available_ortools():
            continue
        out.append((name, fn))
    return out


def list_solvers() -> dict[str, list[str]]:
    """诊断用：所有问题类型 × auto 模式下可用 solver 名。"""
    return {kind: [n for n, _ in get_solvers(kind, "auto")] for kind in SPEC}
