"""EROT: accelerator-oriented entropy-regularized optimal transport."""

from .api import solve
from .types import SolveResult, SolverConfig

__all__ = ["SolveResult", "SolverConfig", "solve"]
__version__ = "0.1.0"
