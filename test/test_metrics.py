from src.metrics_analysis import run_synthetic

def test_synthetic_runs():
    rows = run_synthetic(trials=5, seed=1)
    assert len(rows) == 5
