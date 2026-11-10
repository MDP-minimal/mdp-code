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

# Modules that were deleted from the network but still exist in the original
# checkpoints. A checkpoint entry is dropped only when the current network does
# not have it *and* its name belongs to one of these; anything else is passed on
# and still loaded with strict=True, so a real mismatch keeps raising.
REMOVED_MODULES = (
    "linear_cli.",      # unused projection head
    "linear_modal.",    # unused projection head
    "modal_queue",      # unused contrastive queue
    "modal_label_queue",
    "modal_queue_ptr",
)


def prediction_state_dict(network, checkpoint):
    raw = checkpoint["state_dict"]
    if not raw or any(not key.startswith("net.") for key in raw):
        raise ValueError("Expected the original trainer's net.* parameter names")
    expected = set(network.state_dict().keys())
    state = {}
    for key, value in raw.items():
        name = key[4:]
        if name not in expected:
            if not name.startswith(REMOVED_MODULES):
                raise ValueError("Checkpoint has a parameter the network does not define: " + name)
            continue
        state[name] = value
    return state


def load_network_state(network, checkpoint):
    return network.load_state_dict(prediction_state_dict(network, checkpoint), strict=True)
