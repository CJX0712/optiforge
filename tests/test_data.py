"""data 层测试：确定性生成 / 结构自洽 / 文件 IO."""
import json

import numpy as np
import pytest

from optiforge.core.errors import ProblemLoadError, ProblemValidationError
from optiforge.core.types import Problem
from optiforge.data.generators import (gen_assignment, gen_knapsack, gen_problem,
                                       gen_tsp, load_problem, save_problem,
                                       validate_problem)


def test_tsp_deterministic():
    a, b = gen_tsp(10, 7), gen_tsp(10, 7)
    assert a.payload["dist"] == b.payload["dist"]
    c = gen_tsp(10, 8)
    assert a.payload["dist"] != c.payload["dist"]


def test_tsp_dist_valid():
    p = gen_tsp(12, 3)
    d = np.asarray(p.payload["dist"])
    assert d.shape == (12, 12)
    assert (d >= 0).all() and np.allclose(d, d.T)
    assert (np.diag(d) == 0).all()


def test_knapsack_valid():
    p = gen_knapsack(20, 5)
    w = np.asarray(p.payload["weights"])
    v = np.asarray(p.payload["values"])
    cap = p.payload["capacity"]
    assert p.sense == "max"
    assert (w > 0).all() and (v > 0).all()
    assert 0 < cap < w.sum()


def test_assignment_valid():
    p = gen_assignment(8, 9)
    c = np.asarray(p.payload["cost"])
    assert c.shape == (8, 8) and (c > 0).all()


def test_gen_problem_rejects_unknown_kind():
    with pytest.raises(ProblemValidationError):
        gen_problem("sat", 10, 1)


def test_gen_problem_size_guards():
    with pytest.raises(ProblemValidationError):
        gen_tsp(3, 1)
    with pytest.raises(ProblemValidationError):
        gen_knapsack(4, 1)
    with pytest.raises(ProblemValidationError):
        gen_assignment(2, 1)


def test_validate_problem_detects_corruption():
    p = gen_tsp(8, 2)
    bad = Problem(kind="tsp", sense="min", n=8, seed=2,
                  payload={"dist": [[0]] * 3})
    with pytest.raises(ProblemValidationError):
        validate_problem(bad)
    validate_problem(p)  # 不抛


def test_validate_problem_missing_key():
    p = Problem(kind="knapsack", sense="max", n=5, seed=1, payload={})
    with pytest.raises(ProblemValidationError):
        validate_problem(p)


def test_save_load_roundtrip(tmp_path):
    p = gen_assignment(6, 11)
    f = tmp_path / "inst.json"
    save_problem(p, f)
    q = load_problem(f)
    assert q.payload == p.payload and q.kind == p.kind and q.n == p.n


def test_load_problem_bad_file(tmp_path):
    f = tmp_path / "bad.json"
    f.write_text("{not json", encoding="utf-8")
    with pytest.raises(ProblemLoadError):
        load_problem(f)
    f2 = tmp_path / "bad2.json"
    f2.write_text(json.dumps({"kind": "tsp"}), encoding="utf-8")
    with pytest.raises(ProblemLoadError):
        load_problem(f2)
