"""DEWI: entropy-weighted multimodal indexing."""

__version__ = "0.1.0"

# Export only lightweight symbols
from .scorer import DewiScorer, RobustStats, Signals, Weights

__all__ = [
    "__version__",
    "DewiScorer",
    "RobustStats",
    "Signals",
    "Weights"
]
