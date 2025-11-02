# Δa₂ Alignment Toolkit — Anthropic‑Ready Research Release (v1.0)

**Author:** Zeus Indomitable  
**License:** MIT  
**Status:** Research Prototype — Anthropic‑oriented packaging

## TL;DR
Δa₂ is a reproducible research toolkit that operationalizes a scalar proxy derived from activation covariance curvature (a₂ = tr(Cov(h)^2)/d) and evaluates Δa₂ dynamics under controlled perturbations. This release is packaged for alignment research teams: modular code, LaTeX whitepaper, reproducible experiments, Colab-ready notebook, and safe opt-in adapters for model-backed introspection.

## Highlights for reviewers
- Clear hypothesis: Δa₂ changes correlate with self-report / introspective detection of injected concepts.
- Reproducible synthetic pipeline (fast) + opt-in model-backed adapters (requires local HF model & `--allow-api`).
- Whitepaper (paper/delta_a2_whitepaper.tex) with methodology, evaluation plan, and ethical considerations.
- Notebook and scripts designed to be executed in a sandboxed research environment.

## Quickstart (synthetic experiments)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.metrics_analysis --mode synthetic --trials 100 --out results/demo_results.json
python -m src.metrics_analysis --mode plot --in results/demo_results.json --out results/figure_da2_correlation.png
```

## Running model-backed introspection (OPT‑IN)
**Warning:** only run in isolated container/VM with sufficient RAM/GPU. Use `--allow-api` to enable loading `transformers` + `torch`:
```bash
python -m src.metrics_analysis --mode model_backed --model <HF_MODEL> --allow-api --trials 20
```

## Contents
- `src/` — core code, metrics analysis, CLI
- `paper/` — LaTeX whitepaper and bib
- `notebooks/` — Colab-ready demo notebook
- `experiments/` — example runs and analysis summary
- `results/` — placeholder synthetic results + figures
- `docs/` — methodology & evaluation checklist
- `README_PITCH.md` — 1‑page outreach pitch for research partners

## Contact & Reproducibility
Open an Issue for reproducibility requests. For acquisition or collaboration inquiries, use README_PITCH.md.
