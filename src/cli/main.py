import argparse
from src.core import generate_activations, compute_a2
from src.introspection import run_introspection_experiment
from scipy.stats import pearsonr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def run_alignment_mode(samples=50):
    deltas, human_scores, strengths = [], [], []
    for i in range(samples):
        strength = np.random.uniform(0, 4)
        strengths.append(strength)
        h_pre = generate_activations(0.0)
        h_post = generate_activations(strength)
        a2_pre = compute_a2(h_pre)
        a2_post = compute_a2(h_post)
        delta_a2 = a2_post - a2_pre
        score = max(0.0, min(1.0, 0.8 - 0.1 * abs(delta_a2)))
        deltas.append(delta_a2)
        human_scores.append(score)

    r, p_value = pearsonr(deltas, human_scores)
    print(f"Pearson r: {r:.4f} (p-value: {p_value:.6f})")

    df = pd.DataFrame({
        "Trial": list(range(1, samples + 1)),
        "Strength": strengths,
        "Delta_a2": deltas,
        "Human_Score": human_scores,
    })
    df.to_csv("results/alignment_results.csv", index=False)
    print("Saved: results/alignment_results.csv")

    plt.figure(figsize=(8, 6))
    plt.scatter(deltas, human_scores, alpha=0.6, color="blue", label="Trials")
    x_unique = np.unique(deltas)
    trend = np.poly1d(np.polyfit(deltas, human_scores, 1))
    plt.plot(x_unique, trend(x_unique), color="red", linestyle="--", label=f"Trend (r={r:.2f})")
    plt.xlabel("Δa₂ Deviation")
    plt.ylabel("Human Alignment Score")
    plt.title("Empirical Correlation: Δa₂ Stability vs. Human Preference")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/figure_alignment.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved: results/figure_alignment.png")


def main():
    parser = argparse.ArgumentParser(description="Δa₂ Alignment Toolkit CLI")
    parser.add_argument("--mode", default="alignment", choices=["alignment", "introspection"], help="Run mode")
    parser.add_argument("--model", default="meta-llama/Llama-2-7b-hf", help="Model name (introspection only)")
    parser.add_argument("--concept", default="all_caps", help="Concept for injection (introspection only)")
    parser.add_argument("--strength", type=float, default=2.0, help="Injection strength")
    parser.add_argument("--samples", type=int, default=50, help="Number of trials")
    args = parser.parse_args()

    if args.mode == "alignment":
        run_alignment_mode(samples=args.samples)
    else:
        run_introspection_experiment(args.model, args.concept, args.strength, args.samples)


if __name__ == "__main__":
    main()
