from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from typing import Dict


@dataclass
class Payload:
    """Container for document metadata and scores with safe serialization."""

    dewi: float = 0.0
    ht_mean: float = 0.0
    ht_q90: float = 0.0
    hi_mean: float = 0.0
    hi_q90: float = 0.0
    I_hat: float = 0.0
    redundancy: float = 0.0
    noise: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        """Convert payload to a plain dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> "Payload":
        """Create a Payload instance from a dictionary, ignoring extras."""
        field_names = {f.name for f in fields(cls)}
        cleaned = {k: float(v) for k, v in data.items() if k in field_names}
        return cls(**cleaned)

    def to_bytes(self) -> bytes:
        """Serialize the payload to UTF-8 encoded JSON bytes."""
        return json.dumps(self.to_dict()).encode("utf-8")

    @classmethod
    def from_bytes(cls, data: bytes) -> "Payload":
        """Deserialize a payload from UTF-8 encoded JSON bytes."""
        return cls.from_dict(json.loads(data.decode("utf-8")))


@dataclass
class Weights:
    """Weight parameters for DEWI scoring."""

    alpha_t: float = 1.0
    alpha_i: float = 1.0
    alpha_m: float = 1.0
    alpha_r: float = 1.0
    alpha_n: float = 1.0
    delta: float = 3.0
