from dataclasses import dataclass
from typing import Dict, Tuple, Sequence
import numpy as np

from .types import Payload


def _robust_standardize(x, med, mad):
    mad = float(mad) + 1e-8
    return (x - med) / (1.4826 * mad)


@dataclass(frozen=True)
class RobustStats:
    fields: Dict[str, Tuple[float, float]]

    @classmethod
    def from_payloads(cls, payloads: Sequence[Payload]) -> "RobustStats":
        if not payloads:
            raise ValueError("Cannot compute statistics from empty dataset")
        keys = ["ht_mean", "hi_mean", "redundancy", "noise"]
        arr = {k: np.array([getattr(p, k) for p in payloads], dtype=np.float32) for k in keys}
        fields: Dict[str, Tuple[float, float]] = {}
        for k, v in arr.items():
            med = float(np.median(v))
            mad = float(np.median(np.abs(v - med)))
            fields[k] = (med, mad)
        return cls(fields=fields)

    def z(self, name: str, val: float) -> float:
        med, mad = self.fields[name]
        return float(_robust_standardize(np.array(val, dtype=np.float32), med, mad))
