"""OptiForge CLI — argparse 入口。

用法：
  python -m optiforge.cli --problem all --size 40 --seed 42 --backend auto --output benchmark.json
  python -m optiforge.cli --problem tsp --hpo --hpo-trials 20
"""
from __future__ import annotations

import argparse
import sys

from .core.config import get_config
from .core.errors import OptiForgeError
from .data.generators import gen_problem
from .hpo.tune import tune
from .pipeline.pipeline import OptiPipeline, format_table, save_benchmark
from .solvers.registry import list_solvers


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="optiforge",
                                description="OptiForge: combinatorial optimization "
                                            "(TSP / knapsack / assignment) benchmark suite")
    p.add_argument("--problem", choices=["tsp", "knapsack", "assignment", "all"],
                   default="all")
    p.add_argument("--size", type=int, default=40, help="problem size n")
    p.add_argument("--seed", type=int, default=None, help="random seed (default ENV/42)")
    p.add_argument("--backend", choices=["auto", "ortools", "fallback"], default=None)
    p.add_argument("--timeout", type=float, default=None, help="per-solve timeout seconds")
    p.add_argument("--hpo", action="store_true", help="run Optuna HPO for TSP heuristic first")
    p.add_argument("--hpo-trials", type=int, default=None)
    p.add_argument("--output", default="benchmark.json")
    p.add_argument("--list-solvers", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = get_config()
    if args.list_solvers:
        for kind, names in list_solvers().items():
            print(f"{kind:<12}: {', '.join(names)}")
        return 0

    try:
        hpo_params = None
        if args.hpo:
            trials = args.hpo_trials or cfg.hpo_trials
            print(f"[hpo] tuning TSP heuristic, trials={trials}, seed={args.seed or cfg.seed}")
            res = tune(trials=trials, seed=args.seed or cfg.seed)
            print(f"[hpo] best={res['best_params']} value={res['best_value']:.1f}")
            hpo_params = {"tsp": res["best_params"]}

        kinds = ["tsp", "knapsack", "assignment"] if args.problem == "all" else [args.problem]
        problems = [gen_problem(k, args.size, args.seed or cfg.seed) for k in kinds]
        pipeline = OptiPipeline(
            backend=args.backend or cfg.backend,
            timeout_s=args.timeout or cfg.timeout_s,
            hpo_params=hpo_params,
        )
        result = pipeline.benchmark(problems)
        print(format_table(result["rows"]))
        save_benchmark(result, args.output)
        print(f"\n[ok] benchmark saved -> {args.output}")
        return 0
    except OptiForgeError as exc:
        print(f"[fail] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
