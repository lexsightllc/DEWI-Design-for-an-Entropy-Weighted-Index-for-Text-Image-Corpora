"""DEWI Scorer - Implements the core scoring logic for the DEWI system."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

# === Robust statistics helpers ===

def _robust_standardize(x: np.ndarray, med: float, mad: float) -> np.ndarray:
    """Robust standardization using median and MAD."""
    mad = float(mad) + 1e-8
    return (x - med) / (1.4826 * mad)

@dataclass
class RobustStats:
    """Stores robust statistics (median and MAD) for signal standardization."""
    
    # median, MAD for each field
    fields: Dict[str, Tuple[float, float]]

    @classmethod
    def fit(cls, rows: List[Dict[str, float]]) -> "RobustStats":
        """Compute robust statistics from a list of signal dictionaries."""
        keys = list(rows[0].keys())
        arr = {k: np.array([r[k] for r in rows], dtype=np.float32) for k in keys}
        med_mad = {}
        for k, v in arr.items():
            med = float(np.median(v))
            mad = float(np.median(np.abs(v - med)))
            med_mad[k] = (med, mad)
        return cls(fields=med_mad)

    def z(self, name: str, val: float) -> float:
        """Compute z-score using robust statistics."""
        med, mad = self.fields[name]
        return float(_robust_standardize(np.array(val, dtype=np.float32), med, mad))

@dataclass
class Signals:
    """Container for all signal values for a document."""
    ht_mean: float  # Mean text entropy
    ht_q90: float   # 90th percentile text entropy
    hi_mean: float  # Mean image entropy
    hi_q90: float   # 90th percentile image entropy
    I_hat: float    # Cross-modal mutual information
    redundancy: float  # Redundancy score
    noise: float    # Noise/quality score

@dataclass
class Weights:
    """Weight parameters for the DEWI scoring function."""
    alpha_t: float = 1.0  # text surprisal weight
    alpha_i: float = 1.0  # image surprisal weight
    alpha_m: float = 1.0  # mutual information weight
    alpha_r: float = 1.0  # redundancy penalty weight
    alpha_n: float = 1.0  # noise penalty weight
    delta: float = 3.0    # Huber clamp bound

def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid activation function."""
    return 1.0 / (1.0 + np.exp(-x))

class DewiScorer:
    """Entropy-weighted document scorer.
    
    Combines standardized text/image surprisals, cross-modal MI proxy, and penalties
    into a bounded score in (0,1). Stores robust stats for standardization.
    """

    def __init__(self, weights: Optional[Weights] = None):
        """Initialize the scorer with optional custom weights."""
        self.weights = weights or Weights()
        self.stats: Optional[RobustStats] = None

    def fit_stats(self, rows: List[Signals]) -> None:
        """Fit robust statistics for signal standardization."""
        # Convert to list of dicts for RobustStats
        dict_rows = [r.__dict__ for r in rows]
        self.stats = RobustStats.fit(dict_rows)

    def score(self, sig: Signals) -> float:
        """Compute DEWI score for a document."""
        assert self.stats is not None, "Call fit_stats() first to compute medians/MADs."
        s = self.stats
        
        # Aggregate surprisal terms
        Ht = 0.5 * (s.z("ht_mean", sig.ht_mean) + s.z("ht_q90", sig.ht_q90))
        Hi = 0.5 * (s.z("hi_mean", sig.hi_mean) + s.z("hi_q90", sig.hi_q90))
        I = s.z("I_hat", sig.I_hat)
        R = s.z("redundancy", sig.redundancy)
        N = s.z("noise", sig.noise)

        # Apply weights and compute utility score
        w = self.weights
        U = w.alpha_t * Ht + w.alpha_i * Hi - w.alpha_m * I - w.alpha_r * R - w.alpha_n * N
        
        # Apply Huber-like clamping and sigmoid
        U = np.clip(U, -w.delta, w.delta)
        return float(_sigmoid(U))

    def score_conditional(self, sig: Signals) -> float:
        """Alternative scoring using conditional entropy formulation."""
        assert self.stats is not None, "Call fit_stats() first."
        s = self.stats
        
        # Conditional entropy terms
        HtI = 0.5 * (
            s.z("ht_mean", sig.ht_mean) + s.z("ht_q90", sig.ht_q90)
        ) - s.z("I_hat", sig.I_hat)
        
        HiI = 0.5 * (
            s.z("hi_mean", sig.hi_mean) + s.z("hi_q90", sig.hi_q90)
        ) - s.z("I_hat", sig.I_hat)
        
        # Penalties
        R = s.z("redundancy", sig.redundancy)
        N = s.z("noise", sig.noise)
        
        # Apply weights and compute utility score
        w = self.weights
        U = w.alpha_t * HtI + w.alpha_i * HiI - w.alpha_r * R - w.alpha_n * N
        
        # Apply Huber-like clamping and sigmoid
        U = np.clip(U, -w.delta, w.delta)
        return float(_sigmoid(U))
