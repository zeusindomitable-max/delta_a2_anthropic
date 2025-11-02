from src.metrics_analysis import run_synthetic
import json, os
rows = run_synthetic(trials=100, seed=0)
os.makedirs('results', exist_ok=True)
with open('results/demo_results.json','w') as fh:
    json.dump(rows, fh, indent=2)
print('Saved results/demo_results.json')
