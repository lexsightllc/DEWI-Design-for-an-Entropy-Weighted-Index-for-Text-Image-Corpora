from typing import Optional, Sequence

from .types import Weights, Payload
from .robust import RobustStats


class DewiScorer:
    """Scores documents based on DEWI metrics, using a single Weights source."""

    def __init__(self, weights: Optional[Weights] = None):
        self.weights = weights or Weights()
        self.stats: Optional[RobustStats] = None
        self._fitted: bool = False

    def fit(self, payloads: Sequence[Payload]) -> None:
        self.stats = RobustStats.from_payloads(payloads)
        self._fitted = True

    def score(self, p: Payload) -> float:
        w = self.weights
        s = (w.alpha_t * p.ht_mean) + (w.alpha_i * p.hi_mean) + (w.alpha_r * (1.0 - p.redundancy)) - (w.alpha_n * p.noise)
        return float(s)

    def score_conditional(self, p: Payload) -> float:
        w = self.weights
        entropy_term = (w.alpha_t * p.ht_q90) + (w.alpha_i * p.hi_q90)
        diversity_term = w.alpha_r * (1.0 - p.redundancy)
        return float(entropy_term + diversity_term - w.alpha_n * p.noise)
