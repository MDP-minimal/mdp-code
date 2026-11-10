"""Load trusted original UniTraj Lightning checkpoints, without converting files."""
import torch

def read_checkpoint(path):
    # Original Lightning checkpoints contain trusted experiment metadata (OmegaConf etc.).
    # Never use this on an untrusted/downloaded pickle without verifying its source.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError("Expected an original UniTraj Lightning .ckpt with state_dict")
    return checkpoint

def architecture_from_checkpoint(checkpoint):
    metadata = checkpoint.get("hyper_parameters", {})
    names = {"embed_dim": "dim", "historical_steps": "historical_steps", "future_steps": "future_steps",
             "encoder_depth": "encoder_depth", "num_heads": "num_heads", "mlp_ratio": "mlp_ratio",
             "qkv_bias": "qkv_bias", "drop_path": "drop_path"}
    missing = [value for value in names.values() if value not in metadata]
    if missing:
        raise ValueError("Missing network architecture metadata: " + str(missing))
    return {name: metadata[key] for name, key in names.items()}

def prediction_state_dict(network, checkpoint):
    raw = checkpoint["state_dict"]
    if not raw or any(not key.startswith("net.") for key in raw):
        raise ValueError("Expected the original trainer's net.* parameter names")
    state = {key[4:]: value for key, value in raw.items()}
    return state


def load_network_state(network, checkpoint):
    return network.load_state_dict(prediction_state_dict(network, checkpoint), strict=False)
