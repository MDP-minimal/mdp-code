"""Streaming prediction results with the original evaluate.py text protocol."""
import csv
import json
from pathlib import Path

import numpy as np
import torch


class PredictionResults:
    """Write one model's validation results, without Lightning dependencies."""

    def __init__(self, config):
        models = {"MDP": "mdp", "forecast": "forecast_mae"}
        model_name = config["model_name"]
        if model_name not in models:
            raise ValueError("PredictionResults supports only MDP and forecast")
        self.model = models[model_name]
        self.past_len = int(config["past_len"])
        if self.past_len < 1:
            raise ValueError("past_len must be positive")
        self.output_dir = Path(config["output_dir"]).expanduser().resolve()
        h5_dir = config.get("h5_dir")
        self.h5_directory = None if h5_dir is None else str(Path(h5_dir).expanduser().resolve())
        self._seen = set()
        self._totals = np.zeros(3, dtype=np.float64)
        self._summary = None
        self._stream = None
        metrics_path = self.output_dir / "metrics.csv"
        if metrics_path.exists():
            raise FileExistsError("Use a new output_dir; metrics.csv already exists")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Exclusive creation also protects against races and existing symlinks.
        self._stream = metrics_path.open("x", encoding="utf-8", newline="")
        self._fields = ["scenario_id", "center_objects_id", "ade", "fde",
                        "minADE6", "minFDE6", "miss_rate"]
        self._writer = csv.DictWriter(self._stream, fieldnames=self._fields)
        self._writer.writeheader()

    @torch.no_grad()
    def add(self, batch, prediction, metrics):
        """Append a GPU validation batch and its per-sample UniTraj metrics."""
        if self._stream is None or self._stream.closed:
            raise RuntimeError("PredictionResults is closed")
        inputs = batch["input_dict"]
        trajectories = prediction["predicted_trajectory"].detach()
        if trajectories.ndim != 4 or tuple(trajectories.shape[1:]) != (6, 60, 2):
            raise ValueError("predicted_trajectory must have shape (B, 6, 60, 2)")
        count = trajectories.shape[0]
        scenarios = list(inputs["scenario_id"])
        centers = inputs["center_objects_id"]
        if len(scenarios) != count or len(centers) != count:
            raise ValueError("Scenario and center IDs must match the prediction batch")
        pending = set()
        for scenario in scenarios:
            if (not isinstance(scenario, str) or not scenario
                    or scenario in (".", "..") or "/" in scenario
                    or "\\" in scenario or "\x00" in scenario
                    or Path(scenario).name != scenario):
                raise ValueError("Invalid scenario ID for a filename")
            if scenario in self._seen or scenario in pending:
                raise ValueError("Duplicate scenario_id; this text protocol requires a unique prediction ID")
            pending.add(scenario)
            if (self.output_dir / (scenario + ".txt")).exists():
                raise FileExistsError("Scenario result already exists: " + scenario)

        target = (inputs["obj_trajs_future_state"][:, 0, :, :2]
                  - inputs["obj_trajs"][:, 0, self.past_len - 1, :2].unsqueeze(1))
        invalid = (~inputs["obj_trajs_mask"][:, 0, self.past_len - 1].unsqueeze(-1)
                   | ~inputs["obj_trajs_future_mask"][:, 0, :])
        target = torch.where(invalid.unsqueeze(-1), torch.zeros_like(target), target)
        distance = torch.linalg.vector_norm(trajectories - target[:, None], dim=-1)
        best = distance.sum(-1).argmin(-1)
        selected = distance[torch.arange(count, device=distance.device), best]
        legacy_ade, legacy_fde = selected.mean(-1), selected[:, -1]
        metric_values = []
        for name in ("minADE6", "minFDE6", "miss_rate"):
            value = metrics[name].detach()
            if tuple(value.shape) != (count,):
                raise ValueError(name + " must contain one value per sample")
            metric_values.append(value.to(device=distance.device, dtype=torch.float32))
        # Match the old evaluate torch.stack float32 array before float64 sums.
        values = torch.stack([legacy_ade.float(), legacy_fde.float(), *metric_values], dim=-1).cpu().numpy()
        for index, row in enumerate(values):
            scenario = scenarios[index]
            with (self.output_dir / (scenario + ".txt")).open("x", encoding="utf-8") as stream:
                stream.write(f"ADE: {row[0]:.4f}\nFDE: {row[1]:.4f}\n")
            center = centers[index]
            if isinstance(center, (torch.Tensor, np.generic)):
                center = center.item()
            self._writer.writerow(dict(zip(self._fields, [
                scenario, center, f"{row[0]:.4f}", f"{row[1]:.4f}",
                float(row[2]), float(row[3]), float(row[4])
            ])))
            self._seen.add(scenario)
        self._totals += values[:, 2:5].astype(np.float64).sum(0)

    def finish(self):
        """Close the CSV, save summary.json, and return its contents."""
        if self._summary is not None:
            return self._summary
        if not self._seen:
            self.close()
            raise ValueError("Cannot summarize prediction results without samples")
        if self._stream is None or self._stream.closed:
            raise RuntimeError("PredictionResults was closed without finishing")
        self.close()
        samples = len(self._seen)
        summary = {
            "model": self.model,
            "samples": samples,
            "minADE6": float(self._totals[0] / samples),
            "minFDE6": float(self._totals[1] / samples),
            "miss_rate": float(self._totals[2] / samples),
            "text_protocol": "unmasked minADE-selected mode ADE/FDE, rounded to four decimals",
            "h5_directory": self.h5_directory,
        }
        (self.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        self._summary = summary
        return summary

    def close(self):
        """Close only the CSV stream; do not create a summary."""
        if self._stream is not None and not self._stream.closed:
            self._stream.close()
