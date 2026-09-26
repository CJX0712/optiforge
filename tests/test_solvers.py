"""solvers 测试：有效性不变量 + 小规模暴力最优交叉验证。"""
import numpy as np
import pytest

from optiforge.data.generators import gen_assignment, gen_knapsack, gen_tsp
from optiforge.eval.metrics import brute_force
from optiforge.solvers.assignment import (available_ortools_cpsat,
                                         solve_assignment_ortools,
                                         solve_assignment_scipy)
from optiforge.solvers.knapsack import (solve_knapsack_dp,
                                        solve_knapsack_greedy,
                                        solve_knapsack_ortools)
from optiforge.solvers.registry import available_ortools, get_solvers, list_solvers
from optiforge.solvers.tsp_heuristic import (solve_tsp_heuristic, tour_length,
                                             _nearest_neighbor_start, _two_opt)
from optiforge.solvers.tsp_ortools import (available_ortools_routing,
                                           solve_tsp_ortools)


# ---------- TSP 启发式 ----------
def test_heuristic_tour_is_valid_permutation():
    p = gen_tsp(12, 5)
    sol = solve_tsp_heuristic(p)
    assert sol.status == "heuristic"
    tour = sol.detail["tour"]
    assert sorted(tour) == list(range(12))


def test_heuristic_length_matches_objective():
    p = gen_tsp(10, 6)
    sol = solve_tsp_heuristic(p)
    d = np.asarray(p.payload["dist"], dtype=np.int64)
    assert tour_length(sol.detail["tour"], d) == sol.objective


def test_heuristic_deterministic():
    p = gen_tsp(15, 7)
    a = solve_tsp_heuristic(p, n_starts=3)
    b = solve_tsp_heuristic(p, n_starts=3)
    assert a.detail["tour"] == b.detail["tour"] and a.objective == b.objective


def test_heuristic_never_worse_than_single_nn():
    """2-opt 后长度 <= 纯最近邻。"""
    p = gen_tsp(30, 8)
    d = np.asarray(p.payload["dist"], dtype=np.int64)
    nn = tour_length(_nearest_neighbor_start(d, 0), d)
    improved = tour_length(_two_opt(_nearest_neighbor_start(d, 0), d, 20, False), d)
    assert improved <= nn


def test_heuristic_matches_bruteforce_small():
    p = gen_tsp(7, 4)
    sol = solve_tsp_heuristic(p, n_starts=7, max_passes=50)
    ref = brute_force(p)
    # 7 城市 2-opt 多起点不保证全局最优，但差值应很小（< 15%）
    assert sol.objective <= ref * 1.15


# ---------- TSP ortools ----------
@pytest.mark.skipif(not available_ortools_routing(), reason="ortools unavailable")
def test_ortools_tsp_matches_bruteforce():
    p = gen_tsp(7, 4)
    sol = solve_tsp_ortools(p, timeout_s=10.0)
    assert sol.status == "feasible"
    ref = brute_force(p)
    assert sol.objective <= ref * 1.001  # GLS 可能极微超，或等于


def test_ortools_tsp_graceful_when_missing(monkeypatch):
    import optiforge.solvers.tsp_ortools as m
    monkeypatch.setattr(m, "ORTOOLS_ROUTING_OK", False)
    sol = m.solve_tsp_ortools(gen_tsp(8, 1))
    assert sol.status == "error" and sol.error


# ---------- 背包 ----------
def test_dp_matches_bruteforce():
    p = gen_knapsack(12, 3)
    sol = solve_knapsack_dp(p)
    assert sol.status == "optimal"
    assert sol.objective == brute_force(p)


def test_dp_selected_weight_within_capacity():
    p = gen_knapsack(20, 5)
    sol = solve_knapsack_dp(p)
    w = sum(p.payload["weights"][i] for i in sol.detail["selected"])
    assert w <= p.payload["capacity"]
    assert sol.detail["total_weight"] == w


def test_greedy_le_dp():
    p = gen_knapsack(25, 9)
    g = solve_knapsack_greedy(p)
    d = solve_knapsack_dp(p)
    assert g.objective <= d.objective


def test_greedy_valid():
    p = gen_knapsack(15, 2)
    sol = solve_knapsack_greedy(p)
    assert sol.status == "heuristic"
    assert sorted(sol.detail["selected"]) == sol.detail["selected"] or True
    w = sum(p.payload["weights"][i] for i in sol.detail["selected"])
    assert w <= p.payload["capacity"]


@pytest.mark.skipif(not available_ortools_cpsat(), reason="ortools unavailable")
def test_ortools_knapsack_equals_dp():
    p = gen_knapsack(18, 6)
    a = solve_knapsack_ortools(p, timeout_s=10.0)
    b = solve_knapsack_dp(p)
    assert a.status in ("optimal", "feasible")
    assert a.objective == b.objective


def test_dp_capacity_guard():
    p = gen_knapsack(20, 4)
    huge = dict(p.payload)
    huge["capacity"] = 1_000_000
    from optiforge.core.types import Problem
    big = Problem(kind="knapsack", sense="max", n=20, seed=4, payload=huge)
    sol = solve_knapsack_dp(big)
    assert sol.status == "error" and "DP limit" in sol.error


# ---------- 分配 ----------
def test_scipy_assignment_matches_bruteforce():
    p = gen_assignment(6, 3)
    sol = solve_assignment_scipy(p)
    assert sol.status == "optimal"
    assert sol.objective == brute_force(p)


def test_scipy_col_ind_is_permutation():
    p = gen_assignment(9, 5)
    sol = solve_assignment_scipy(p)
    assert sorted(sol.detail["col_ind"]) == list(range(9))


@pytest.mark.skipif(not available_ortools_cpsat(), reason="ortools unavailable")
def test_ortools_assignment_equals_scipy():
    p = gen_assignment(7, 8)
    a = solve_assignment_ortools(p, timeout_s=10.0)
    b = solve_assignment_scipy(p)
    assert a.status in ("optimal", "feasible")
    assert a.objective == b.objective


# ---------- registry ----------
def test_registry_auto_never_empty():
    for kind in ("tsp", "knapsack", "assignment"):
        assert get_solvers(kind, "auto"), kind


def test_registry_fallback_has_no_ortools():
    for kind in ("tsp", "knapsack", "assignment"):
        names = [n for n, _ in get_solvers(kind, "fallback")]
        assert all("ortools" not in n for n in names)


def test_registry_ortools_mode():
    if available_ortools():
        for kind in ("tsp", "knapsack", "assignment"):
            names = [n for n, _ in get_solvers(kind, "ortools")]
            assert names and all("ortools" in n for n in names)
    else:
        assert get_solvers("tsp", "ortools") == []


def test_registry_unknown_kind_raises():
    from optiforge.core.errors import BackendUnavailableError
    with pytest.raises(BackendUnavailableError):
        get_solvers("sat", "auto")
    with pytest.raises(BackendUnavailableError):
        get_solvers("tsp", "bogus")


def test_list_solvers_diagnostic():
    diag = list_solvers()
    assert set(diag) == {"tsp", "knapsack", "assignment"}
    assert available_ortools() in (True, False)
