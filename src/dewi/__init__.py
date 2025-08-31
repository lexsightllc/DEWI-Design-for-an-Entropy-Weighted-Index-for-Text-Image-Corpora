"""DEWI: entropy-weighted multimodal indexing."""

__version__ = "0.1.0"

from .scorer import DewiScorer
from .robust import RobustStats
from .types import Weights, Payload

__all__ = [
    "__version__",
    "DewiScorer",
    "RobustStats",
    "Weights",
    "Payload",
]
