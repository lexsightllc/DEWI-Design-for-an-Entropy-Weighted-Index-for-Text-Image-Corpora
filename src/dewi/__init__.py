"""DEWI: A Design for an Entropy-Weighted Index for Text+Image Corpora.

This module provides tools for building and querying an entropy-weighted index
that prioritizes useful surprise in multimodal (text+image) data.
"""

__version__ = "0.1.0"

# Core components
from dewi.signals import (
    TextEntropyEstimator,
    ImageEntropyEstimator,
    CrossModalDependency,
    RedundancyEstimator,
    NoiseEstimator,
)
from dewi.index import DewiIndex
from dewi.scoring import DEWIScorer

__all__ = [
    "TextEntropyEstimator",
    "ImageEntropyEstimator",
    "CrossModalDependency",
    "RedundancyEstimator",
    "NoiseEstimator",
    "DewiIndex",
    "DEWIScorer",
]
