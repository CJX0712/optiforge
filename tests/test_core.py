"""core 层测试：types / errors / config / interfaces."""
import os

import pytest

from optiforge.core.config import Config, get_config
from optiforge.core.errors import (BackendUnavailableError, ConfigError,
                                   EnvConfigError, OptiForgeError)
from optiforge.core.types import Problem, Solution


# ---------- types ----------
def test_problem_kind_validation():
    with pytest.raises(ValueError):
        Problem(kind="bogus", sense="min", n=5, seed=1)
    with pytest.raises(ValueError):
        Problem(kind="tsp", sense="sideways", n=5, seed=1)


def test_solution_score_semantics_min_and_max():
    s_min = Solution("tsp", "a", "numpy", objective=100.0, status="feasible",
                     runtime_s=0.1)
    s_max = Solution("knapsack", "b", "numpy", objective=250.0, status="optimal",
                     runtime_s=0.1)
    assert s_min.score == 100.0          # min: score == objective
    assert s_max.score == -250.0         # max: score == -objective


def test_solution_status_validation():
    with pytest.raises(ValueError):
        Solution("tsp", "a", "numpy", 1.0, "weird", 0.1)


def test_is_better_than_prefers_lower_score():
    good = Solution("tsp", "a", "numpy", 10.0, "feasible", 0.1)
    bad = Solution("tsp", "b", "numpy", 20.0, "feasible", 0.1)
    err = Solution("tsp", "c", "numpy", float("inf"), "error", 0.1)
    assert good.is_better_than(bad)
    assert not bad.is_better_than(good)
    assert good.is_better_than(err)
    assert not err.is_better_than(good)


# ---------- errors ----------
def test_error_codes_segmented():
    assert ConfigError("x").code == "E100"
    assert EnvConfigError("x").code == "E110"
    assert BackendUnavailableError("x").code == "E310"


def test_error_is_optiforge_error():
    assert isinstance(OptiForgeError("x"), Exception)


# ---------- config ----------
def test_config_defaults(monkeypatch):
    for var in ("OPTIFORGE_SEED", "OPTIFORGE_BACKEND", "OPTIFORGE_TIMEOUT_S",
                "OPTIFORGE_TRIALS", "OPTIFORGE_OUTPUT_DIR"):
        monkeypatch.delenv(var, raising=False)
    cfg = get_config()
    assert cfg.seed == 42 and cfg.backend == "auto"
    assert cfg.timeout_s == 10.0 and cfg.hpo_trials == 20


def test_config_env_override(monkeypatch):
    monkeypatch.setenv("OPTIFORGE_SEED", "7")
    monkeypatch.setenv("OPTIFORGE_BACKEND", "fallback")
    monkeypatch.setenv("OPTIFORGE_TIMEOUT_S", "2.5")
    monkeypatch.setenv("OPTIFORGE_TRIALS", "5")
    monkeypatch.setenv("OPTIFORGE_OUTPUT_DIR", "out_x")
    cfg = get_config()
    assert (cfg.seed, cfg.backend, cfg.timeout_s, cfg.hpo_trials, cfg.output_dir) == \
        (7, "fallback", 2.5, 5, "out_x")


def test_config_invalid_values(monkeypatch):
    monkeypatch.setenv("OPTIFORGE_SEED", "abc")
    with pytest.raises(EnvConfigError):
        get_config()
    monkeypatch.setenv("OPTIFORGE_SEED", "42")
    monkeypatch.setenv("OPTIFORGE_BACKEND", "quantum")
    with pytest.raises(EnvConfigError):
        get_config()
    monkeypatch.setenv("OPTIFORGE_BACKEND", "auto")
    monkeypatch.setenv("OPTIFORGE_TIMEOUT_S", "0")
    with pytest.raises(EnvConfigError):
        get_config()


def test_config_dataclass_direct_invalid_backend():
    with pytest.raises(EnvConfigError):
        Config(backend="nope")
