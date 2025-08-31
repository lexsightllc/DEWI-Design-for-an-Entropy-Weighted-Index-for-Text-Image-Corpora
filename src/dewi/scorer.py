from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .types import Weights


@dataclass
class RobustStats:
    """Stores median and MAD for robust standardization."""

    medians: Dict[str, float]
    mads: Dict[str, float]

    @classmethod
    def fit(cls, rows: List[Dict[str, float]]) -> "RobustStats":
        keys = rows[0].keys()
        arr = {k: np.asarray([r[k] for r in rows], dtype=np.float32) for k in keys}
        med = {k: float(np.median(v)) for k, v in arr.items()}
        mad = {
            k: float(np.median(np.abs(v - med[k]))) or 1e-8 for k, v in arr.items()
        }
        return cls(medians=med, mads=mad)

    def z(self, name: str, val: float) -> float:
        med = self.medians[name]
        mad = self.mads[name]
        return float((val - med) / (1.4826 * mad))


class DewiScorer:
    """Robust DEWI scorer supporting standard and conditional modes."""

    def __init__(self, weights: Optional[Weights] = None, delta: float = 3.0):
        self.weights = weights or Weights()
        self.weights.delta = delta
        self.stats: Optional[RobustStats] = None

    def fit_stats(self, rows: List[Dict[str, float]]) -> None:
        """Fit robust statistics from signal dictionaries."""
        self.stats = RobustStats.fit(rows)

    def is_fitted(self) -> bool:
        return self.stats is not None

    def _components(self, sig: Dict[str, float]) -> Dict[str, float]:
        assert self.stats is not None, "Call fit_stats() before scoring."
        s = self.stats
        return {
            "Ht": 0.5 * (s.z("ht_mean", sig["ht_mean"]) + s.z("ht_q90", sig["ht_q90"])),
            "Hi": 0.5 * (s.z("hi_mean", sig["hi_mean"]) + s.z("hi_q90", sig["hi_q90"])),
            "I": s.z("I_hat", sig["I_hat"]),
            "R": s.z("redundancy", sig["redundancy"]),
            "N": s.z("noise", sig["noise"]),
        }

    @staticmethod
    def _sigmoid(x: float) -> float:
        return float(1.0 / (1.0 + np.exp(-x)))

    def score(self, sig: Dict[str, float]) -> float:
        comps = self._components(sig)
        w = self.weights
        U = (
            w.alpha_t * comps["Ht"]
            + w.alpha_i * comps["Hi"]
            - w.alpha_m * comps["I"]
            - w.alpha_r * comps["R"]
            - w.alpha_n * comps["N"]
        )
        U = float(np.clip(U, -w.delta, w.delta))
        return self._sigmoid(U)

    def score_conditional(self, sig: Dict[str, float]) -> float:
        comps = self._components(sig)
        w = self.weights
        HtI = comps["Ht"] - comps["I"]
        HiI = comps["Hi"] - comps["I"]
        U = (
            w.alpha_t * HtI
            + w.alpha_i * HiI
            - w.alpha_r * comps["R"]
            - w.alpha_n * comps["N"]
        )
        U = float(np.clip(U, -w.delta, w.delta))
        return self._sigmoid(U)
