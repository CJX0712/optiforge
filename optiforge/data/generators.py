"""OptiForge.data — 合成问题实例生成 + JSON 文件载入。

生成器确定性：同 seed 同参数 → 完全相同实例（单测覆盖）。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..core.errors import ProblemLoadError, ProblemValidationError
from ..core.types import Problem

_KIND_SENSE = {"tsp": "min", "knapsack": "max", "assignment": "min"}


def _dist_matrix(coords: np.ndarray) -> np.ndarray:
    diff = coords[:, None, :] - coords[None, :, :]
    d = np.sqrt((diff ** 2).sum(axis=-1))
    return np.round(d * 1000).astype(int)  # 整数化，OR-Tools 需要整型弧代价


def gen_tsp(n: int, seed: int) -> Problem:
    """随机欧氏 TSP。n>=4。"""
    if n < 4:
        raise ProblemValidationError(f"tsp requires n>=4, got {n}")
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0.0, 100.0, size=(n, 2))
    return Problem(
        kind="tsp", sense="min", n=n, seed=seed,
        payload={"coords": coords.tolist(), "dist": _dist_matrix(coords).tolist()},
        name=f"tsp_n{n}_s{seed}",
    )


def gen_knapsack(n: int, seed: int) -> Problem:
    """强相关 0-1 背包（value ≈ weight + noise），有难度梯度。n>=5。"""
    if n < 5:
        raise ProblemValidationError(f"knapsack requires n>=5, got {n}")
    rng = np.random.default_rng(seed)
    weights = rng.integers(5, 100, size=n)
    values = weights + rng.integers(-3, 8, size=n)
    values = np.maximum(values, 1)
    capacity = int(0.5 * weights.sum())
    return Problem(
        kind="knapsack", sense="max", n=n, seed=seed,
        payload={
            "values": values.astype(int).tolist(),
            "weights": weights.astype(int).tolist(),
            "capacity": capacity,
        },
        name=f"knapsack_n{n}_s{seed}",
    )


def gen_assignment(n: int, seed: int) -> Problem:
    """随机方阵最小化分配。n>=3。"""
    if n < 3:
        raise ProblemValidationError(f"assignment requires n>=3, got {n}")
    rng = np.random.default_rng(seed)
    cost = rng.integers(1, 200, size=(n, n)).astype(int)
    return Problem(
        kind="assignment", sense="min", n=n, seed=seed,
        payload={"cost": cost.tolist()},
        name=f"assignment_n{n}_s{seed}",
    )


_GENERATORS = {"tsp": gen_tsp, "knapsack": gen_knapsack, "assignment": gen_assignment}


def validate_problem(problem: Problem) -> None:
    """结构自洽校验，失败抛 ProblemValidationError。"""
    p = problem.payload
    try:
        if problem.kind == "tsp":
            dist = np.asarray(p["dist"], dtype=int)
            if dist.shape != (problem.n, problem.n):
                raise ValueError("dist must be n x n")
            if (dist < 0).any():
                raise ValueError("dist must be non-negative")
            if not np.allclose(dist, dist.T):
                raise ValueError("dist must be symmetric")
        elif problem.kind == "knapsack":
            values = np.asarray(p["values"], dtype=int)
            weights = np.asarray(p["weights"], dtype=int)
            if values.shape != (problem.n,) or weights.shape != (problem.n,):
                raise ValueError("values/weights must be length n")
            if (weights <= 0).any() or (values <= 0).any():
                raise ValueError("values/weights must be positive")
            if p["capacity"] <= 0:
                raise ValueError("capacity must be positive")
        else:
            cost = np.asarray(p["cost"], dtype=int)
            if cost.shape != (problem.n, problem.n):
                raise ValueError("cost must be n x n")
    except KeyError as exc:
        raise ProblemValidationError(f"missing payload key: {exc}") from exc
    except (ValueError, TypeError) as exc:
        raise ProblemValidationError(str(exc)) from exc


def gen_problem(kind: str, n: int, seed: int) -> Problem:
    if kind not in _GENERATORS:
        raise ProblemValidationError(f"unknown kind: {kind}")
    problem = _GENERATORS[kind](n, seed)
    validate_problem(problem)
    return problem


def save_problem(problem: Problem, path: str | Path) -> None:
    data = {
        "kind": problem.kind, "sense": problem.sense, "n": problem.n,
        "seed": problem.seed, "name": problem.name, "payload": problem.payload,
    }
    Path(path).write_text(json.dumps(data), encoding="utf-8")


def load_problem(path: str | Path) -> Problem:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        problem = Problem(
            kind=data["kind"], sense=data["sense"], n=data["n"],
            seed=data["seed"], payload=data["payload"], name=data.get("name", ""),
        )
    except (OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
        raise ProblemLoadError(f"cannot load problem from {path}: {exc}") from exc
    validate_problem(problem)
    return problem
