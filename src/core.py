"""Core Δa2 computations."""
import numpy as np
from typing import Iterable

def incremental_covariance(iterator: Iterable[np.ndarray]):
    mean = None
    M2 = None
    n = 0
    for x in iterator:
        x = np.asarray(x, dtype=float).ravel()
        n += 1
        if mean is None:
            mean = x.copy()
            M2 = np.zeros((x.size, x.size), dtype=float)
            continue
        delta = x - mean
        mean += delta / n
        M2 += np.outer(delta, x - mean)
    if n < 2:
        if mean is None:
            return np.eye(1) * 1e-6
        return np.eye(mean.size) * 1e-6
    return M2 / (n - 1)


def compute_a2_from_matrix(h: np.ndarray, regularizer: float = 1e-6) -> float:
    h = np.asarray(h, dtype=float)
    if h.ndim == 1:
        h = h.reshape(1, -1)
    seq_len, dim = h.shape
    if seq_len >= 2 and seq_len <= 5000:
        cov = np.cov(h.T)
    else:
        cov = incremental_covariance(h)
    cov = cov + np.eye(cov.shape[0]) * regularizer
    R = float(np.trace(cov @ cov) / float(cov.shape[0]))
    return R
  
