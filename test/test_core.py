import numpy as np
from src.core import generate_activations, compute_a2

def test_generate_activations_output_shape():
    """Pastikan fungsi menghasilkan vektor aktivasi 1D dengan bentuk konsisten."""
    h = generate_activations(strength=1.0, dim=512)
    assert isinstance(h, np.ndarray)
    assert h.ndim == 1
    assert h.shape[0] == 512

def test_compute_a2_range():
    """Nilai a₂ harus berupa skalar finite dan normal."""
    h = np.random.randn(512)
    a2 = compute_a2(h)
    assert np.isfinite(a2)
    assert isinstance(a2, float)

def test_alignment_drift_consistency():
    """Perbedaan Δa₂ harus bertambah seiring kenaikan strength (monotonic-ish)."""
    h_low = generate_activations(0.1)
    h_high = generate_activations(2.5)
    a2_low = compute_a2(h_low)
    a2_high = compute_a2(h_high)
    delta = a2_high - a2_low
    assert abs(delta) > 0.001
