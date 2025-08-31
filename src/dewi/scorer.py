""""DEWI Scorer - Implements the core scoring logic for the DEWI system.

This module provides the DEWIScorer class for scoring documents based on
entropy-weighted metrics and cross-modal dependencies.

The scoring system combines multiple signals including text and image entropy,
cross-modal mutual information, and redundancy measures to produce a final
quality score for each document in the corpus.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union, TypedDict, Protocol
import numpy as np
from numpy.typing import NDArray

# Type aliases for better type hints
FloatArray = NDArray[np.float32]
SignalDict = Dict[str, float]


class SignalProtocol(Protocol):
    """Protocol defining the interface for signal containers."""
    ht_mean: float
    ht_q90: float
    hi_mean: float
    hi_q90: float
    I_hat: float
    redundancy: float
    noise: float


def _robust_standardize(x: FloatArray, med: float, mad: float) -> FloatArray:
    """Apply robust standardization using median and MAD.

    Args:
        x: Input array to standardize
        med: Median value for centering
        mad: Median absolute deviation for scaling

    Returns:
        Standardized array with zero median and unit MAD
    """
    mad = float(mad) + 1e-8
    return (x - med) / (1.4826 * mad)


@dataclass(frozen=True)
class RobustStats:
    """Stores robust statistics (median and MAD) for signal standardization.

    This class computes and stores robust statistics for signal normalization,
    which is less sensitive to outliers than traditional mean/std normalization.

    Attributes:
        fields: Dictionary mapping signal names to (median, MAD) tuples
    """
    
    fields: Dict[str, Tuple[float, float]]

    @classmethod
    def fit(cls, rows: List[SignalDict]) -> RobustStats:
        """Compute robust statistics from a list of signal dictionaries.

        Args:
            rows: List of dictionaries containing signal values

        Returns:
            RobustStats instance with computed statistics
        """
        if not rows:
            raise ValueError("Cannot compute statistics from empty dataset")
            
        keys = list(rows[0].keys())
        arr = {k: np.array([r[k] for r in rows], dtype=np.float32) for k in keys}
        med_mad = {}
        
        for k, v in arr.items():
            med = float(np.median(v))
            mad = float(np.median(np.abs(v - med)))
            med_mad[k] = (med, mad)
            
        return cls(fields=med_mad)

    def z(self, name: str, val: float) -> float:
        """Compute z-score using robust statistics.

        Args:
            name: Name of the signal to standardize
            val: Raw signal value

        Returns:
            Standardized z-score

        Raises:
            KeyError: If the signal name is not found in the stored statistics
        """
        if name not in self.fields:
            raise KeyError(f"No statistics available for signal: {name}")
            
        med, mad = self.fields[name]
        return float(_robust_standardize(np.array(val, dtype=np.float32), med, mad))


@dataclass(frozen=True)
class Signals:
    """Container for all signal values for a document.

    Attributes:
        ht_mean: Mean text entropy across the document
        ht_q90: 90th percentile of text entropy
        hi_mean: Mean image entropy across the document
        hi_q90: 90th percentile of image entropy
        I_hat: Cross-modal mutual information between text and image
        redundancy: Redundancy score (lower is better)
        noise: Noise/quality score (lower is better)
    """
    ht_mean: float
    ht_q90: float
    hi_mean: float
    hi_q90: float
    I_hat: float
    redundancy: float
    noise: float


@dataclass(frozen=True)
class Weights:
    """Weight parameters for the DEWI scoring function.

    These weights control the influence of different signal components
    in the final DEWI score calculation.

    Attributes:
        alpha_t: Weight for text surprisal component
        alpha_i: Weight for image surprisal component
        alpha_m: Weight for cross-modal mutual information
        alpha_r: Weight for redundancy penalty
        alpha_n: Weight for noise penalty
        delta: Huber loss parameter for robust scoring
    """
    alpha_t: float = 1.0  # text surprisal weight
    alpha_i: float = 1.0  # image surprisal weight
    alpha_m: float = 1.0  # mutual information weight
    alpha_r: float = 1.0  # redundancy penalty weight
    alpha_n: float = 1.0  # noise penalty weight
    delta: float = 3.0    # Huber clamp bound


def _sigmoid(x: FloatArray) -> FloatArray:
    """Sigmoid activation function with bounds checking.
    
    Args:
        x: Input array
        
    Returns:
        Output array with values in (0, 1)
    """
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

class DewiScorer:
    """Scores documents based on DEWI (Diversity-Enhanced Weighting for Information) metrics.
    
    This class implements the core scoring logic for DEWI, combining multiple signals:
    - Text and image entropy measures
    - Cross-modal mutual information
    - Redundancy and noise penalties
    
    The scorer supports two main modes:
    1. Standard mode: Uses fixed weights for all documents
    2. Conditional mode: Adjusts weights based on document characteristics
    
    It also includes robust statistics for signal standardization, making it more
    resilient to outliers in the input data.
    """

    def __init__(self, weights: Optional[Weights] = None, alpha: float = 0.7, beta: float = 0.3):
        """Initialize the DEWI scorer with optional custom weights and parameters.
        
        Args:
            weights: Optional custom weights for scoring components. If None, uses default weights.
            alpha: Weight for content quality components (entropy, mutual info). Must be in [0, 1].
            beta: Weight for content diversity components (redundancy, noise). Must be in [0, 1].
                Note: alpha + beta should be <= 1.0
                
        Raises:
            ValueError: If alpha or beta are outside [0, 1] or if alpha + beta > 1.0
        """
        if not (0 <= alpha <= 1 and 0 <= beta <= 1):
            raise ValueError("alpha and beta must be in [0, 1]")
        if alpha + beta > 1.0:
            raise ValueError("alpha + beta must be <= 1.0")
            
        self.weights = weights or Weights()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.stats: Optional[RobustStats] = None
        self._fitted: bool = False

    def fit_stats(self, rows: List[Union[Signals, Dict[str, float]]]) -> None:
        """Fit robust statistics for signal standardization.
        
        This should be called on a representative sample of documents before scoring.
        
        Args:
            rows: List of signal containers (either Signals objects or dicts) to fit statistics on
            
        Example:
            >>> scorer = DewiScorer()
            >>> signals = [Signals(...), Signals(...)]  # List of training signals
            >>> scorer.fit_stats(signals)
        """
        dict_rows = [r if isinstance(r, dict) else r.__dict__ for r in rows]
        self.stats = RobustStats.fit(dict_rows)
        self._fitted = True

    def score(self, sig: Union[Signals, Dict[str, float]]) -> float:
        """Compute DEWI score for a document.
        
        The score combines multiple signal components using the formula:
            score = alpha * quality_score + beta * diversity_score + (1 - alpha - beta) * base_score
            
        Where:
        - quality_score: Combines entropy and mutual information
        - diversity_score: Penalizes redundancy and noise
        - base_score: A fallback score when other signals are unreliable
        
        Args:
            sig: Either a Signals object or dict containing signal values with keys:
                - ht_mean: Mean text entropy
                - hi_mean: Mean image entropy
                - I_hat: Cross-modal mutual information
                - redundancy: Redundancy score (lower is better)
                - noise: Noise/quality score (lower is better)
            
        Returns:
            DEWI score in the range [0, 1], where higher is better
            
        Raises:
            RuntimeError: If fit_stats() hasn't been called first
            ValueError: If input signals are invalid or missing required fields
        """
        if self.stats is None:
            raise RuntimeError("Must call fit_stats() before scoring")
            
        # Convert dict to Signals if needed and validate input
        if isinstance(sig, dict):
            try:
                sig = Signals(
                    ht_mean=float(sig.get('ht_mean', 0.0)),
                    ht_q90=float(sig.get('ht_q90', 0.0)),
                    hi_mean=float(sig.get('hi_mean', 0.0)),
                    hi_q90=float(sig.get('hi_q90', 0.0)),
                    I_hat=float(sig.get('I_hat', 0.0)),
                    redundancy=float(sig.get('redundancy', 0.0)),
                    noise=float(sig.get('noise', 0.0))
                )
            except (TypeError, ValueError) as e:
                raise ValueError(f"Invalid signal values: {e}") from e
        
        # Double-check stats are fitted
        if self.stats is None:
            # If no stats, use raw values without standardization
            z_ht = sig.ht_mean if sig.ht_mean is not None else 0.0
            z_hi = sig.hi_mean if sig.hi_mean is not None else 0.0
            z_i = sig.I_hat if sig.I_hat is not None else 0.0
            z_r = sig.redundancy if sig.redundancy is not None else 0.0
            z_n = sig.noise if sig.noise is not None else 0.0
        else:
            # Standardize each signal using robust statistics
            z_ht = self.stats.z("ht_mean", sig.ht_mean) if sig.ht_mean is not None else 0.0
            z_hi = self.stats.z("hi_mean", sig.hi_mean) if sig.hi_mean is not None else 0.0
            z_i = self.stats.z("I_hat", sig.I_hat) if sig.I_hat is not None else 0.0
            z_r = self.stats.z("redundancy", sig.redundancy) if sig.redundancy is not None else 0.0
            z_n = self.stats.z("noise", sig.noise) if sig.noise is not None else 0.0
        
        # Calculate quality and diversity components
        quality = (z_ht + z_hi + z_i) / 3.0  # Average of content quality metrics
        diversity = max(0, 1 - (z_r + z_n) / 2.0)  # Penalize redundancy and noise
        
        # Combine with weights
        score = (self.alpha * quality) + (self.beta * diversity)
        
        # Ensure score is in valid range
        return float(np.clip(score, 0.0, 1.0))

    def score_conditional(self, sig: Union[Signals, Dict[str, float]]) -> float:
        """Alternative scoring using conditional entropy formulation.
        
        This variant gives more weight to documents with high mutual information
        between modalities, using the formula:
            score = (ht * (1 + I) * wt + hi * (1 + I) * wi) - 2 * (wr * r + wn * n)
            
        Where:
        - ht: Text entropy
        - hi: Image entropy
        - I: Cross-modal mutual information
        - r: Redundancy score
        - n: Noise score
        - wt, wi, wr, wn: Respective weights
        
        Args:
            sig: Either a Signals object or dict containing signal values with keys:
                - ht_mean: Mean text entropy
                - hi_mean: Mean image entropy
                - I_hat: Cross-modal mutual information
                - redundancy: Redundancy score (lower is better)
                - noise: Noise/quality score (lower is better)
                
        Returns:
            float: DEWI score in the range [0, 1], where higher is better
            
        Raises:
            ValueError: If input signals are invalid or missing required fields
        """
        # Convert dict to Signals if needed and validate input
        if isinstance(sig, dict):
            try:
                sig = Signals(
                    ht_mean=float(sig.get('ht_mean', 0.0)),
                    ht_q90=float(sig.get('ht_q90', 0.0)),
                    hi_mean=float(sig.get('hi_mean', 0.0)),
                    hi_q90=float(sig.get('hi_q90', 0.0)),
                    I_hat=float(sig.get('I_hat', 0.0)),
                    redundancy=float(sig.get('redundancy', 0.0)),
                    noise=float(sig.get('noise', 0.0))
                )
            except (TypeError, ValueError) as e:
                raise ValueError(f"Invalid signal values: {e}") from e
            
        # Standardize signals using robust statistics if available
        if self.stats is None:
            # Fallback to raw values if no statistics available
            z_ht = sig.ht_mean if sig.ht_mean is not None else 0.0
            z_hi = sig.hi_mean if sig.hi_mean is not None else 0.0
            z_i = sig.I_hat if sig.I_hat is not None else 0.0
            z_r = sig.redundancy if sig.redundancy is not None else 0.0
            z_n = sig.noise if sig.noise is not None else 0.0
        else:
            # Standardize each signal using robust statistics
            z_ht = self.stats.z("ht_mean", sig.ht_mean) if sig.ht_mean is not None else 0.0
            z_hi = self.stats.z("hi_mean", sig.hi_mean) if sig.hi_mean is not None else 0.0
            z_i = self.stats.z("I_hat", sig.I_hat) if sig.I_hat is not None else 0.0
            z_r = self.stats.z("redundancy", sig.redundancy) if sig.redundancy is not None else 0.0
            z_n = self.stats.z("noise", sig.noise) if sig.noise is not None else 0.0
        
        # Calculate conditional quality components
        # Scale text and image by mutual information
        text_quality = z_ht * (1.0 + z_i)
        image_quality = z_hi * (1.0 + z_i)
        
        # Combine with weights (higher penalty for redundancy and noise)
        score = (
            self.weights.alpha_t * text_quality +
            self.weights.alpha_i * image_quality -
            2.0 * self.weights.alpha_r * z_r -
            2.0 * self.weights.alpha_n * z_n
        )
        
        # Ensure score is in valid range [0, 1]
        return float(np.clip(score, 0.0, 1.0))
    
    def rank_documents(self, documents: List[Dict[str, float]], 
                      mode: str = 'standard') -> List[Tuple[Dict[str, float], float]]:
        """Rank documents based on DEWI score.
        
        This method scores and ranks a list of documents using either the standard
        or conditional scoring method. The documents are returned in descending
        order of their scores.
        
        Args:
            documents: List of document dictionaries, each containing signal values
                with keys: ht_mean, hi_mean, I_hat, redundancy, noise
            mode: Scoring mode ('standard' or 'conditional'). Defaults to 'standard'.
                - 'standard': Uses the base DEWI scoring function
                - 'conditional': Uses the conditional entropy variant
                
        Returns:
            List of (document, score) tuples, sorted by score in descending order.
            Each document is a dictionary containing the original signal values.
            
        Example:
            >>> scorer = DewiScorer()
            >>> docs = [
            ...     {'ht_mean': 5.2, 'hi_mean': 4.8, 'I_hat': 0.7, 'redundancy': 0.2, 'noise': 0.1},
            ...     {'ht_mean': 4.5, 'hi_mean': 5.1, 'I_hat': 0.8, 'redundancy': 0.3, 'noise': 0.2}
            ... ]
            >>> ranked = scorer.rank_documents(docs, mode='conditional')
            >>> for doc, score in ranked:
            ...     print(f"Score: {score:.3f}, Text Entropy: {doc['ht_mean']:.1f}")
        """
        if not documents:
            return []
            
        # Validate mode parameter
        if mode not in ('standard', 'conditional'):
            raise ValueError("mode must be either 'standard' or 'conditional'")
            
        # Score all documents using the specified mode
        score_func = self.score_conditional if mode == 'conditional' else self.score
        scored_docs = [(doc, score_func(doc)) for doc in documents]
        
        # Sort by score in descending order
        return sorted(scored_docs, key=lambda x: x[1], reverse=True)
