"""Run retained native UniTraj processing, without additional training labels."""
import argparse
from pathlib import Path
import random
import numpy as np
import torch
from omegaconf import OmegaConf
from hydra import compose, initialize_config_dir
from unitraj.datasets import build_dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--cache-root", default="cache")
    parser.add_argument("--model", choices=["mdp", "forecast_mae"], default="mdp")
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    root = Path(__file__).resolve().parent
    with initialize_config_dir(version_base=None, config_dir=str(root / "configs")):
        config = compose(config_name="data", overrides=["method=" + args.model])
    OmegaConf.set_struct(config, False)
    config = OmegaConf.merge(config, config.method)
    data = str(Path(args.data_dir).resolve())
    config.train_data_path = [data]
    config.val_data_path = [data]
    config.cache_path = str(Path(args.cache_root).resolve())
    config.preprocess_num_workers = args.workers
    config.h5_dir = None # This entry must preprocess raw data, not reuse H5_DIR from the environment.
    config.method.model_name = "MDP" if args.model == "mdp" else "forecast"
    random.seed(config.seed); np.random.seed(config.seed); torch.manual_seed(config.seed)
    import os
    os.chdir(root)
    dataset = build_dataset(config, val=args.split == "val")
    print("Processed samples:", len(dataset))
    print("H5 directory:", dataset.cache_path)

if __name__ == "__main__":
    main()
