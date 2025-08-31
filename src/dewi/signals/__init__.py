"""Signal computation for DEWI.

This module provides various signal estimators for text and image analysis,
including entropy estimation, cross-modal analysis, redundancy detection,
and noise estimation.
"""

from typing import Dict, Any, Optional, List, Tuple, Union, Callable
import numpy as np

# Import the main signal classes from their respective modules
from .text_entropy import TextEntropyEstimator
from .image_entropy import ImageEntropyEstimator
from .cross_modal import CrossModalDependency, CrossModalEstimator
from .redundancy import RedundancyEstimator
from .noise import NoiseEstimator

__all__ = [
    'TextEntropyEstimator',
    'ImageEntropyEstimator',
    'CrossModalDependency',
    'CrossModalEstimator',
    'RedundancyEstimator',
    'NoiseEstimator'
]
