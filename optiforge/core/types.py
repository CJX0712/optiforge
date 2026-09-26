"""OptiForge core types.

统一评测语义：
- Solution.score 恒为「越小越好」：min 问题 score == objective，max 问题 score == -objective。
- Solution.status: optimal | feasible | heuristic | infeasible | error
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

SENSE_MIN = "min"
SENSE_MAX = "max"

VALID_STATUS = ("optimal", "feasible", "heuristic", "infeasible", "error")


@dataclass(frozen=True)
class Problem:
    """一个组合优化问题实例。

    payload 由 data 生成器保证自洽：
    - tsp:        {"coords": list[(x, y)], "dist": list[list[int]]}
    - knapsack:   {"values": list[int], "weights": list[int], "capacity": int}
    - assignment: {"cost": list[list[int]]}  (方阵)
    """

    kind: str                 # tsp | knapsack | assignment
    sense: str                # min | max（目标自然语义）
    n: int
    seed: int
    payload: Dict[str, Any] = field(default_factory=dict)
    name: str = ""

    def __post_init__(self) -> None:
        if self.kind not in ("tsp", "knapsack", "assignment"):
            raise ValueError(f"unknown problem kind: {self.kind}")
        if self.sense not in (SENSE_MIN, SENSE_MAX):
            raise ValueError(f"unknown problem sense: {self.sense}")


@dataclass(frozen=True)
class Solution:
    """求解结果。objective 为问题自然语义目标值；score 统一为越小越好。"""

    problem_kind: str
    solver_name: str
    backend: str                       # ortools | scipy | numpy ...
    objective: float
    status: str
    runtime_s: float
    detail: Dict[str, Any] = field(default_factory=dict)   # tour / selected / col_ind 等
    error: Optional[str] = None

    def __post_init__(self) -> None:
        if self.status not in VALID_STATUS:
            raise ValueError(f"invalid status: {self.status}")

    @property
    def sense(self) -> str:
        return SENSE_MAX if self.problem_kind == "knapsack" else SENSE_MIN

    @property
    def score(self) -> float:
        """统一语义：越小越好。max 问题取负。"""
        return self.objective if self.sense == SENSE_MIN else -self.objective

    def is_better_than(self, other: "Solution") -> bool:
        if self.status in ("infeasible", "error"):
            return False
        if other.status in ("infeasible", "error"):
            return True
        return self.score < other.score
