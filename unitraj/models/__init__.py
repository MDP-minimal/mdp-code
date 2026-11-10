"""Only the two prediction models are registered for testing."""
import torch
from .mdp.model import ModelForecast as MDP
from .forecast_mae.model_forecast import ModelForecast as ForecastMAE

MODELS = {"mdp": MDP, "forecast_mae": ForecastMAE}

def load_model(name, checkpoint, device):
    from .checkpoint import read_checkpoint, architecture_from_checkpoint, load_network_state
    payload = read_checkpoint(checkpoint)
    model = MODELS[name](**architecture_from_checkpoint(payload))
    load_network_state(model, payload)
    return model.to(device).eval()


# Trainer construction remains lazy: standalone inference does not require Lightning.
def build_model(config):
    from .mdp.trainer_forecast_unitraj import TrainerMDP
    from .mdp.trainer_auto import Trainer_auto
    from .mdp.trainer_auto_inter import Trainer_auto_inter
    from .forecast_mae.trainer_forecast import TrainerForecast
    trainers = {"MDP": TrainerMDP, "mdp": TrainerMDP, "Tail-MDP": TrainerMDP,
                "forecast": TrainerForecast, "forecast_mae": TrainerForecast,
                "auto": Trainer_auto, "inter": Trainer_auto_inter, "auto_inter": Trainer_auto_inter}
    method = config.get("method", {})
    name = method.get("model_name", config.get("model_name", "MDP"))
    if name not in trainers:
        raise KeyError("Unknown model: " + str(name))
    return trainers[name](config=config)
