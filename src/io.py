"""I/O helpers and synthetic activation generation."""
import numpy as np

def synthetic_activations(seq_len=10, dim=128, seed=None, n_samples=1):
    rng = np.random.default_rng(seed)
    for _ in range(n_samples):
        yield rng.normal(loc=0.0, scale=0.5, size=(seq_len, dim)) + 1.0

def inject_additive(h, concept_vec, strength=1.0):
    return h + (strength * concept_vec)[None, :]
  
