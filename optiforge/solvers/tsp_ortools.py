"""TSP SOTA 后端：Google OR-Tools Routing（Guided Local Search）。"""
from __future__ import annotations

import time

from ..core.types import Problem, Solution

try:  # 可选依赖探测
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2
    ORTOOLS_ROUTING_OK = True
    _IMPORT_ERR = ""
except Exception as _exc:  # noqa: BLE001
    ORTOOLS_ROUTING_OK = False
    _IMPORT_ERR = str(_exc)


def available_ortools_routing() -> bool:
    return ORTOOLS_ROUTING_OK


def solve_tsp_ortools(problem: Problem, timeout_s: float = 10.0) -> Solution:
    if not ORTOOLS_ROUTING_OK:
        return Solution(
            problem_kind="tsp", solver_name="tsp_ortools", backend="ortools",
            objective=float("inf"), status="error", runtime_s=0.0,
            error=f"ortools unavailable: {_IMPORT_ERR}",
        )
    t0 = time.perf_counter()
    try:
        dist = problem.payload["dist"]
        n = problem.n
        manager = pywrapcp.RoutingIndexManager(n, 1, 0)
        routing = pywrapcp.RoutingModel(manager)

        def transit_cb(from_index: int, to_index: int) -> int:
            return int(dist[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)])

        transit = routing.RegisterTransitCallback(transit_cb)
        routing.SetArcCostEvaluatorOfAllVehicles(transit)

        params = pywrapcp.DefaultRoutingSearchParameters()
        params.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
        params.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
        params.time_limit.FromSeconds(max(1, int(timeout_s)))

        solution = routing.SolveWithParameters(params)
        if solution is None:
            return Solution(
                problem_kind="tsp", solver_name="tsp_ortools", backend="ortools",
                objective=float("inf"), status="error",
                runtime_s=time.perf_counter() - t0, error="no solution found",
            )
        index = routing.Start(0)
        tour: list[int] = []
        while not routing.IsEnd(index):
            tour.append(manager.IndexToNode(index))
            index = solution.Value(routing.NextVar(index))
        objective = float(solution.ObjectiveValue())
        return Solution(
            problem_kind="tsp", solver_name="tsp_ortools", backend="ortools",
            objective=objective, status="feasible",
            runtime_s=time.perf_counter() - t0,
            detail={"tour": tour},
        )
    except Exception as exc:  # noqa: BLE001
        return Solution(
            problem_kind="tsp", solver_name="tsp_ortools", backend="ortools",
            objective=float("inf"), status="error",
            runtime_s=time.perf_counter() - t0, error=str(exc),
        )
