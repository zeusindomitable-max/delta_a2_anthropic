import pandas as pd
from src.introspection import run_introspection_experiment

def test_introspection_output_dataframe():
    """Fungsi introspeksi harus mengembalikan DataFrame dengan kolom utama."""
    df = run_introspection_experiment(
        model_name="meta-llama/Llama-2-7b-hf",
        concept="all_caps",
        strength=0.5,
        num_trials=3
    )
    assert isinstance(df, pd.DataFrame)
    for col in ["Trial", "Δa₂", "Self_Report_Detection"]:
        assert col in df.columns

def test_introspection_variability():
    """Δa₂ tidak boleh konstan (uji dinamika eksperimental)."""
    df = run_introspection_experiment(
        model_name="meta-llama/Llama-2-7b-hf",
        concept="mirror_test",
        strength=1.0,
        num_trials=5
    )
    assert df["Δa₂"].std() > 0.0
