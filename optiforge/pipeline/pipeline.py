"""pipeline — OptiPipeline.run() 单问题最优解 + benchmark() 跨实例跨求解器评测。"""
from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..core.errors import PipelineError
from ..core.types import Problem, Solution
from ..eval.metrics import brute_force, optimality_gap
from ..solvers.registry import get_solvers


class OptiPipeline:
    """组合优化端到端管线。"""

    def __init__(self, backend: str = "auto", timeout_s: float = 10.0,
                 hpo_params: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        if backend not in ("auto", "ortools", "fallback"):
            raise PipelineError(f"unknown backend: {backend}")
        self.backend = backend
        self.timeout_s = timeout_s
        # hpo_params: {problem_kind: {param: value}} 注入对应兜底求解器
        self.hpo_params = hpo_params or {}

    def run(self, problem: Problem) -> Solution:
        """跑该问题全部可用求解器，返回 score 最优者。"""
        solvers = get_solvers(problem.kind, self.backend)
        if not solvers:
            raise PipelineError(f"no solver available for {problem.kind} "
                                f"(backend={self.backend})")
        best: Optional[Solution] = None
        for name, fn in solvers:
            kwargs: Dict[str, Any] = {}
            if name == "tsp_heuristic" and problem.kind in self.hpo_params:
                kwargs.update(self.hpo_params[problem.kind])
            sol = fn(problem, timeout_s=self.timeout_s, **kwargs)
            if best is None or sol.is_better_than(best):
                best = sol
        if best is None:
            raise PipelineError(f"all solvers failed for {problem.name}")
        return best

    def benchmark(self, problems: List[Problem],
                  with_reference: bool = True) -> Dict[str, Any]:
        """跨问题 × 跨求解器全量评测，输出行 + 最优解 + 参照 gap。"""
        rows: List[Dict[str, Any]] = []
        best_by_problem: Dict[str, Dict[str, Any]] = {}
        for problem in problems:
            solvers = get_solvers(problem.kind, self.backend)
            ref_obj = brute_force(problem) if with_reference else None
            ref_score = None
            if ref_obj is not None:
                ref_score = ref_obj if problem.sense == "min" else -ref_obj
            best_score = float("inf")
            best_name = ""
            entries: List[Dict[str, Any]] = []
            for name, fn in solvers:
                kwargs: Dict[str, Any] = {}
                if name == "tsp_heuristic" and problem.kind in self.hpo_params:
                    kwargs.update(self.hpo_params[problem.kind])
                sol = fn(problem, timeout_s=self.timeout_s, **kwargs)
                valid = sol.status not in ("error", "infeasible")
                if valid and sol.score < best_score:
                    best_score = sol.score
                    best_name = name
                gap = (optimality_gap(sol, ref_score)
                       if ref_score is not None and valid else None)
                entries.append({
                    "problem": problem.name, "kind": problem.kind, "n": problem.n,
                    "seed": problem.seed, "solver": name, "backend": sol.backend,
                    "objective": (sol.objective
                                  if sol.objective not in (float("inf"), float("-inf"))
                                  else None),
                    "score": sol.score if valid else None,
                    "status": sol.status, "gap_pct": gap,
                    "runtime_s": sol.runtime_s, "error": sol.error,
                })
            # 无暴力参照时：以 best-known 为参照补相对 gap（标注 reference）
            reference_used = "exact_bruteforce" if ref_score is not None else ""
            if ref_score is None:
                for e in entries:
                    if e["score"] is not None and best_score not in (float("inf"),):
                        e["gap_pct"] = (e["score"] - best_score) / abs(best_score) * 100.0
                reference_used = "best_known" if entries else ""
            for e in entries:
                e["reference"] = reference_used
            rows.extend(entries)
            best_by_problem[problem.name] = {"solver": best_name, "score": best_score}
        return {
            "meta": {
                "backend": self.backend, "timeout_s": self.timeout_s,
                "hpo_params": self.hpo_params, "n_problems": len(problems),
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            "rows": rows,
            "best_by_problem": best_by_problem,
        }


def save_benchmark(result: Dict[str, Any], path: str | Path) -> None:
    Path(path).write_text(json.dumps(result, indent=2, ensure_ascii=False),
                          encoding="utf-8")


def format_table(rows: List[Dict[str, Any]]) -> str:
    """固定宽度表头对齐（避开中英文混排错位坑）。"""
    headers = ["problem", "solver", "backend", "objective", "status", "gap_pct", "time_s"]
    widths = [22, 20, 9, 12, 10, 9, 8]
    lines = [" ".join(h.ljust(w) for h, w in zip(headers, widths)),
             " ".join("-" * w for w in widths)]
    for r in rows:
        obj = "-" if r["objective"] is None else f"{r['objective']:.1f}"
        gap = "-" if r["gap_pct"] is None else f"{r['gap_pct']:.2f}%"
        lines.append(" ".join([
            r["problem"][:21].ljust(widths[0]),
            r["solver"][:19].ljust(widths[1]),
            r["backend"][:8].ljust(widths[2]),
            obj[:11].ljust(widths[3]),
            r["status"][:9].ljust(widths[4]),
            gap[:8].ljust(widths[5]),
            f"{r['runtime_s']:.3f}".ljust(widths[6]),
        ]))
    return "\n".join(lines)
