"""Local weight computation for DEWI."""

import numpy as np

def local_weights_from_surprisal(s: np.ndarray) -> np.ndarray:
    """Convert per-token or per-patch surprisals to positive weights.
    
    Applies robust standardization and a softplus activation to ensure
    positive weights that highlight surprising regions.
    
    Args:
        s: Array of surprisal values (lower is more expected).
        
    Returns:
        Array of positive weights with the same shape as input.
    """
    s = np.asarray(s, dtype=np.float32)
    
    # Robust standardization using median and MAD
    med = np.median(s)
    mad = np.median(np.abs(s - med)) + 1e-8  # Add small epsilon to avoid division by zero
    z = (s - med) / (1.4826 * mad)  # 1.4826 makes MAD consistent with std for normal dist
    
    # Clip to prevent extreme values and apply softplus
    z = np.clip(z, -5, 5)
    return np.log1p(np.exp(z))  # log1p(exp(x)) is more numerically stable than softplus
