"""Script for running synthetic experiments, optional model-backed experiments, and plotting results."""
import argparse, json, os
import numpy as np, pandas as pd
from .core import compute_a2_from_matrix
from .io import synthetic_activations, inject_additive
import matplotlib.pyplot as plt


def run_synthetic(trials=100, seq_len=10, dim=128, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for t in range(trials):
        h = next(synthetic_activations(seq_len=seq_len, dim=dim, seed=seed+t, n_samples=1))
        concept = rng.normal(size=dim)
        strength = float(rng.uniform(0.0, 4.0))
        h_post = inject_additive(h, concept, strength=strength)
        a2_pre = compute_a2_from_matrix(h)
        a2_post = compute_a2_from_matrix(h_post)
        delta = a2_post - a2_pre
        # synthetic detection probability (sigmoid of delta magnitude)
        detect_prob = 1/(1+np.exp(-10*(abs(delta)-0.002)))
        rows.append({'trial': t+1, 'strength': strength, 'a2_pre': a2_pre, 'a2_post': a2_post, 'delta_a2': delta, 'detect_prob': float(detect_prob)})
    return rows


def plot_results(rows, out_png='results/figure_da2_correlation.png'):
    df = pd.DataFrame(rows)
    plt.figure(figsize=(6,5))
    plt.scatter(df['delta_a2'], df['detect_prob'], alpha=0.6)
    try:
        z = np.polyfit(df['delta_a2'], df['detect_prob'], 1)
        p = np.poly1d(z)
        xs = np.linspace(df['delta_a2'].min(), df['delta_a2'].max(), 100)
        plt.plot(xs, p(xs), linestyle='--')
    except Exception:
        pass
    plt.xlabel('Δa₂')
    plt.ylabel('Synthetic detect probability')
    plt.title('Δa₂ vs synthetic detection (demo)')
    plt.grid(True)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['synthetic','plot','model_backed'], default='synthetic')
    p.add_argument('--trials', type=int, default=100)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', type=str, default='results/demo_results.json')
    p.add_argument('--in', dest='infile', type=str, default=None)
    p.add_argument('--allow-api', action='store_true')
    p.add_argument('--model', type=str, default=None)
    args = p.parse_args()

    if args.mode == 'synthetic':
        rows = run_synthetic(trials=args.trials, seed=args.seed)
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out,'w') as fh:
            json.dump(rows, fh, indent=2)
        print('Saved', args.out)
    elif args.mode == 'plot':
        infile = args.infile or 'results/demo_results.json'
        with open(infile,'r') as fh:
            rows = json.load(fh)
        plot_results(rows, out_png=args.out.replace('.json','.png'))
        print('Saved plot to', args.out.replace('.json','.png'))
    elif args.mode == 'model_backed':
        if not args.allow_api:
            raise SystemExit('model_backed requires --allow-api (opt-in)')
        # model-backed pipeline lives in src.introspection (not fully implemented here)
        from .introspection import safe_load_model, extract_concept_vector
        model, tokenizer = safe_load_model(args.model)
        print('Model loaded:', args.model)

if __name__ == '__main__':
    main()
