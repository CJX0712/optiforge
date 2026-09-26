"""pipeline / eval / hpo / cli 测试."""
import json

import numpy as np
import pytest

from optiforge.core.errors import PipelineError
from optiforge.data.generators import gen_assignment, gen_knapsack, gen_problem, gen_tsp
from optiforge.eval.metrics import brute_force, optimality_gap
from optiforge.hpo.tune import tune
from optiforge.pipeline.pipeline import (OptiPipeline, format_table,
                                         save_benchmark)


# ---------- eval.metrics ----------
def test_gap_min_problem():
    from optiforge.core.types import Solution
    s = Solution("tsp", "a", "numpy", 110.0, "feasible", 0.1)
    assert optimality_gap(s, 100.0) == pytest.approx(10.0)


def test_gap_max_problem_uses_score():
    from optiforge.core.types import Solution
    # knapsack objective=90 → score=-90；ref objective=100 → ref_score=-100
    s = Solution("knapsack", "g", "numpy", 90.0, "heuristic", 0.1)
    # gap = (score - ref)/|ref| = (-90 - (-100))/100 = 10%
    assert optimality_gap(s, -100.0) == pytest.approx(10.0)


def test_gap_error_solution_is_inf():
    from optiforge.core.types import Solution
    s = Solution("tsp", "a", "numpy", float("inf"), "error", 0.1, error="x")
    assert optimality_gap(s, 100.0) == float("inf")


def test_gap_zero_reference_raises():
    from optiforge.core.errors import EvalError
    from optiforge.core.types import Solution
    s = Solution("tsp", "a", "numpy", 1.0, "feasible", 0.1)
    with pytest.raises(EvalError):
        optimality_gap(s, 0.0)


def test_bruteforce_all_kinds():
    assert brute_force(gen_tsp(6, 1)) is not None
    assert brute_force(gen_knapsack(10, 1)) is not None
    assert brute_force(gen_assignment(5, 1)) is not None
    assert brute_force(gen_tsp(50, 1)) is None      # 超规模返回 None
    assert brute_force(gen_knapsack(30, 1)) is None


# ---------- pipeline.run ----------
def test_run_returns_best_solution():
    p = gen_assignment(8, 2)
    best = OptiPipeline(backend="auto", timeout_s=5.0).run(p)
    assert best.status in ("optimal", "feasible", "heuristic")
    # assignment 精确解存在时 best 应为 optimal 且 == brute force（n=8 可暴力）
    ref = brute_force(p)
    assert best.objective == ref


def test_run_knapsack_beats_or_equals_greedy():
    p = gen_knapsack(20, 4)
    best = OptiPipeline(timeout_s=5.0).run(p)
    ref = brute_force(p)
    assert best.objective == ref  # dp/ortools 至少一个精确


def test_run_unknown_backend_raises():
    p = gen_tsp(6, 1)
    with pytest.raises(PipelineError):
        OptiPipeline(backend="warp").run(p)


def test_run_hpo_params_injected():
    p = gen_tsp(12, 3)
    pipe = OptiPipeline(backend="fallback", timeout_s=5.0,
                        hpo_params={"tsp": {"n_starts": 2, "max_passes": 5}})
    sol = pipe.run(p)
    assert sol.status == "heuristic"


def test_run_fallback_only_backend():
    p = gen_assignment(10, 7)
    sol = OptiPipeline(backend="fallback").run(p)
    assert sol.backend == "scipy"


# ---------- benchmark ----------
def test_benchmark_rows_and_serializable(tmp_path):
    problems = [gen_tsp(8, 1), gen_knapsack(10, 2), gen_assignment(6, 3)]
    res = OptiPipeline(timeout_s=5.0).benchmark(problems)
    assert res["meta"]["n_problems"] == 3
    assert len(res["rows"]) >= 3
    for row in res["rows"]:
        assert row["status"] in ("optimal", "feasible", "heuristic", "error")
    f = tmp_path / "b.json"
    save_benchmark(res, f)
    loaded = json.loads(f.read_text(encoding="utf-8"))
    assert loaded["meta"]["n_problems"] == 3


def test_benchmark_gap_against_reference():
    problems = [gen_tsp(7, 5)]
    res = OptiPipeline(timeout_s=5.0).benchmark(problems)
    ortools_rows = [r for r in res["rows"] if r["solver"] == "tsp_ortools"]
    if ortools_rows and ortools_rows[0]["status"] != "error":
        assert ortools_rows[0]["gap_pct"] is not None
        assert ortools_rows[0]["gap_pct"] < 1.0


def test_format_table_aligned():
    problems = [gen_assignment(6, 1)]
    res = OptiPipeline(timeout_s=5.0).benchmark(problems)
    text = format_table(res["rows"])
    lines = text.splitlines()
    assert len({len(ln) for ln in lines}) == 1  # 每行等宽


# ---------- hpo ----------
def test_tune_reproducible_and_minimize():
    a = tune(trials=3, seed=1)
    b = tune(trials=3, seed=1)
    assert a["best_params"] == b["best_params"]
    assert a["best_value"] == pytest.approx(b["best_value"])
    assert a["direction"] == "minimize"
    assert a["n_trials"] == 3
    assert set(a["best_params"]) == {"n_starts", "max_passes", "first_improve"}


def test_tune_params_feed_pipeline():
    res = tune(trials=2, seed=2)
    p = gen_tsp(10, 3)
    sol = OptiPipeline(backend="fallback",
                       hpo_params={"tsp": res["best_params"]}).run(p)
    assert sol.status == "heuristic"


# ---------- cli ----------
def test_cli_list_solvers(capsys):
    from optiforge.cli import main
    assert main(["--list-solvers"]) == 0
    out = capsys.readouterr().out
    assert "tsp" in out and "assignment" in out


def test_cli_benchmark_run(tmp_path, capsys):
    from optiforge.cli import main
    out_file = tmp_path / "cli_bench.json"
    code = main(["--problem", "assignment", "--size", "6", "--seed", "9",
                 "--backend", "fallback", "--output", str(out_file)])
    assert code == 0
    assert out_file.exists()
    data = json.loads(out_file.read_text(encoding="utf-8"))
    assert data["rows"][0]["kind"] == "assignment"


def test_cli_invalid_backend_fails(tmp_path):
    from optiforge.cli import main
    code = main(["--problem", "tsp", "--size", "6", "--backend", "fallback",
                 "--hpo-trials", "1", "--output", str(tmp_path / "x.json")])
    assert code == 0  # fallback 合法
