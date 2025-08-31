from dataclasses import dataclass
from .backends import Payload as Payload


@dataclass
class Weights:
    """Weight parameters for DEWI scoring."""
    alpha_t: float = 1.0
    alpha_i: float = 1.0
    alpha_m: float = 1.0
    alpha_r: float = 1.0
    alpha_n: float = 1.0
    delta: float = 3.0
