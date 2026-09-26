"""OptiForge errors — 分段错误码 E100~E500."""
from __future__ import annotations


class OptiForgeError(Exception):
    """Base error. code 段: 1xx config / 2xx data / 3xx solver / 4xx eval / 5xx pipeline."""

    default_code = "E000"

    def __init__(self, message: str, code: str | None = None) -> None:
        self.code = code or self.default_code
        super().__init__(f"[{self.code}] {message}")


# ---- config E1xx ----
class ConfigError(OptiForgeError):
    default_code = "E100"


class EnvConfigError(ConfigError):
    default_code = "E110"


# ---- data E2xx ----
class DataError(OptiForgeError):
    default_code = "E200"


class ProblemLoadError(DataError):
    default_code = "E210"


class ProblemValidationError(DataError):
    default_code = "E220"


# ---- solver E3xx ----
class SolverError(OptiForgeError):
    default_code = "E300"


class BackendUnavailableError(SolverError):
    default_code = "E310"


class SolveFailureError(SolverError):
    default_code = "E320"


# ---- eval E4xx ----
class EvalError(OptiForgeError):
    default_code = "E400"


class ReferenceMissingError(EvalError):
    default_code = "E410"


# ---- pipeline E5xx ----
class PipelineError(OptiForgeError):
    default_code = "E500"
