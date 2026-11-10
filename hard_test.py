"""Forecast-MAE-FDE Top 1/2/3/4/5/10% comparison using the original text metric definition."""
import argparse
import math
from pathlib import Path

def read_metrics(directory):
    result = {}
    for file in sorted(Path(directory).glob("*.txt")):
        lines = file.read_text().splitlines()
        ade = float(lines[0].split(":", 1)[1]); fde = float(lines[1].split(":", 1)[1])
        if not math.isfinite(ade) or not math.isfinite(fde):
            raise ValueError("Non-finite metric in " + str(file))
        result[file.stem] = (ade, fde)
    if not result:
        raise ValueError("No metrics found in " + str(directory))
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default="results/forecast_mae")
    parser.add_argument("--mdp", default="results/mdp")
    args = parser.parse_args()
    baseline, mdp = read_metrics(args.baseline), read_metrics(args.mdp)
    if set(baseline) != set(mdp):
        raise ValueError("Baseline and MDP sample IDs must match exactly")
    ordered = sorted(baseline, key=lambda key: (-baseline[key][1], key))
    rows = []
    for percentage in [1, 2, 3, 4, 5, 10]:
        n = max(1, int(len(ordered) * percentage / 100))
        selected = ordered[:n]
        a = [sum(baseline[key][i] for key in selected) / n for i in [0, 1]]
        b = [sum(mdp[key][i] for key in selected) / n for i in [0, 1]]
        rows.append({"top_percent": percentage, "samples": n, "forecast_mae_ADE": a[0],
                     "forecast_mae_FDE": a[1], "MDP_ADE": b[0], "MDP_FDE": b[1]})
        print(f"Top {percentage}% | {n} | Forecast-MAE {a[0]:.4f}/{a[1]:.4f} | MDP {b[0]:.4f}/{b[1]:.4f}")

if __name__ == "__main__":
    main()
