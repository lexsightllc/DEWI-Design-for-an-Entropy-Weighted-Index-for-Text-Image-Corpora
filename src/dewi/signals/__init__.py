"""Signal computation for DEWI.

This module provides various signal estimators for text and image analysis,
including entropy estimation, cross-modal analysis, redundancy detection,
and noise estimation.
"""

from typing import Dict, Any, Optional, List, Tuple, Union, Callable
import numpy as np

try:  # Optional import; avoids hard dependency on torch
    from .text_entropy import TextEntropyEstimator  # pragma: no cover
except Exception:  # pragma: no cover
    TextEntropyEstimator = None  # type: ignore

try:
    from .image_entropy import ImageEntropyEstimator  # noqa: F401
except Exception:  # pragma: no cover
    ImageEntropyEstimator = None  # type: ignore

try:
    from .cross_modal import CrossModalDependency  # type: ignore  # pragma: no cover
except Exception:
    CrossModalDependency = None  # type: ignore

try:
    from .redundancy import RedundancyEstimator  # noqa: F401
except Exception:
    RedundancyEstimator = None  # type: ignore

try:
    from .noise import NoiseEstimator  # noqa: F401
except Exception:
    NoiseEstimator = None  # type: ignore

__all__ = [
    'TextEntropyEstimator',
    'ImageEntropyEstimator',
    'CrossModalDependency',
    'RedundancyEstimator',
    'NoiseEstimator',
]
