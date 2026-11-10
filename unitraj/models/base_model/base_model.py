"""UniTraj Lightning wrapper support, without external training-label processing."""
from pathlib import Path
import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
from unitraj.datasets.format import FeatureConverter

class BaseModel(pl.LightningModule):
    def __init__(self, config=None):
        super().__init__()
        self.config = config if config is not None else OmegaConf.load(
            Path(__file__).resolve().parents[3] / "configs/data.yaml")
        self._prediction_results = None

    def convert_to_fmae_format(self, inputs):
        data = FeatureConverter(self.config, self.device).convert_to_fmae_format(inputs)
        if self.training:
            # Pass through existing training targets; do not generate or insert labels.
            keys = ("dynamic", "modal", "i_lane", "i_others", "i_target", "i_all", "head",
                    "CDF_dyn", "CDF_lane", "CDF_agent")
            for key in keys:
                if key in inputs:
                    value = inputs[key]
                    data[key] = value.to(self.device) if isinstance(value, torch.Tensor) else value
        return data

    def load_prediction_weights(self, path):
        from unitraj.models.checkpoint import read_checkpoint, load_network_state
        return load_network_state(self.net, read_checkpoint(path))

    def on_load_checkpoint(self, checkpoint):
        from unitraj.models.checkpoint import prediction_state_dict
        state = prediction_state_dict(self.net, checkpoint)
        checkpoint["state_dict"] = {"net." + key: value for key, value in state.items()}

    def on_validation_start(self):
        name = self.config.get("model_name")
        if self.config.get("save_results", False) and name in ("MDP", "mdp", "Tail-MDP", "forecast", "forecast_mae"):
            if self.trainer.world_size != 1:
                raise ValueError("Result export requires one GPU to avoid duplicated prediction records")
            from unitraj.utils.results import PredictionResults
            self._prediction_results = PredictionResults(self.config)

    def on_validation_end(self):
        if self._prediction_results is not None:
            self._prediction_results.finish()
            self._prediction_results = None

    def teardown(self, stage):
        if self._prediction_results is not None:
            self._prediction_results.close()
            self._prediction_results = None

    def compute_official_evaluation(self, batch, prediction):
        # The minimal package evaluates standard UniTraj metrics below, not SDK submissions.
        if any(self.config.get(key, False) for key in ("eval_nuscenes", "eval_waymo", "eval_argoverse2")):
            raise NotImplementedError("External SDK submission evaluation is not included in this minimal package")

    def log_info(self, batch, batch_idx, prediction, status="train"):
        inputs = batch["input_dict"]
        trajectory = prediction["predicted_trajectory"][..., :2]
        gt = inputs["center_gt_trajs"].to(trajectory.device)[..., :2]
        mask = inputs["center_gt_trajs_mask"].to(trajectory.device)
        distance = torch.linalg.vector_norm(trajectory - gt[:, None], dim=-1)
        valid = mask.sum(-1)
        if torch.any(valid == 0):
            raise ValueError("No valid future ground truth")
        ade = ((distance * mask[:, None]).sum(-1) / valid[:, None]).min(-1).values
        index = inputs["center_gt_final_valid_idx"].to(trajectory.device).long().view(-1, 1, 1)
        fde_all = distance.gather(-1, index.expand(-1, distance.shape[1], 1)).squeeze(-1)
        fde, best = fde_all.min(-1)
        probability = prediction["predicted_probability"]
        probability = probability[torch.arange(len(best), device=best.device), best]
        values = {"minADE6": ade, "minFDE6": fde, "miss_rate": (fde > 2.0).float(),
                  "brier_fde": fde + (1 - probability).square()}
        for name, value in values.items():
            self.log(status + "/" + name, value.mean(), on_step=False, on_epoch=True,
                     sync_dist=True, batch_size=len(gt))
        if status == "val" and self._prediction_results is not None:
            self._prediction_results.add(batch, prediction, values)
