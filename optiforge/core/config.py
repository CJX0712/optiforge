"""OptiForge config — ENV_XXX_* 覆盖 + 默认值。

环境变量（全部可选）：
- OPTIFORGE_SEED       全局随机种子（默认 42）
- OPTIFORGE_BACKEND    auto | ortools | fallback（默认 auto）
- OPTIFORGE_TIMEOUT_S  单次求解超时秒（默认 10）
- OPTIFORGE_TRIALS     HPO trial 数（默认 20）
- OPTIFORGE_OUTPUT_DIR 基准输出目录（默认 ./benchmark_out）
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from .errors import EnvConfigError


def _get_int(name: str, default: int, lo: int, hi: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        val = int(raw)
    except ValueError as exc:
        raise EnvConfigError(f"{name} must be int, got {raw!r}") from exc
    if not (lo <= val <= hi):
        raise EnvConfigError(f"{name} out of range [{lo}, {hi}]: {val}")
    return val


def _get_float(name: str, default: float, lo: float, hi: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        val = float(raw)
    except ValueError as exc:
        raise EnvConfigError(f"{name} must be float, got {raw!r}") from exc
    if not (lo <= val <= hi):
        raise EnvConfigError(f"{name} out of range [{lo}, {hi}]: {val}")
    return val


@dataclass(frozen=True)
class Config:
    seed: int = 42
    backend: str = "auto"          # auto | ortools | fallback
    timeout_s: float = 10.0
    hpo_trials: int = 20
    output_dir: str = "benchmark_out"

    def __post_init__(self) -> None:
        if self.backend not in ("auto", "ortools", "fallback"):
            raise EnvConfigError(f"backend must be auto|ortools|fallback, got {self.backend!r}")


def get_config() -> Config:
    """读取环境变量构建 Config（未设置则用默认值）。"""
    backend = os.environ.get("OPTIFORGE_BACKEND", "auto") or "auto"
    return Config(
        seed=_get_int("OPTIFORGE_SEED", 42, 0, 2**31 - 1),
        backend=backend,
        timeout_s=_get_float("OPTIFORGE_TIMEOUT_S", 10.0, 0.1, 3600.0),
        hpo_trials=_get_int("OPTIFORGE_TRIALS", 20, 1, 1000),
        output_dir=os.environ.get("OPTIFORGE_OUTPUT_DIR", "benchmark_out") or "benchmark_out",
    )
