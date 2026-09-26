"""OptiForge interfaces — Protocol 契约，solver 只依赖此处."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from .types import Problem, Solution


@runtime_checkable
class Solver(Protocol):
    """所有求解器契约：supports() + solve()。

    solve() 不抛异常（内部失败转为 status="error" 的 Solution），
    保证 benchmark 循环健壮。
    """

    name: str
    backend: str

    def supports(self, problem: Problem) -> bool: ...

    def solve(self, problem: Problem, timeout_s: float = 10.0) -> Solution: ...
