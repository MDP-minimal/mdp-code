import pytorch_lightning as pl
import torch

torch.set_float32_matmul_precision('medium')
from torch.utils.data import DataLoader
from unitraj.models import build_model
from unitraj.datasets import build_dataset
from unitraj.utils.utils import set_seed
import hydra
from omegaconf import OmegaConf


@hydra.main(version_base=None, config_path="configs", config_name="data")
def evaluation(cfg):
    set_seed(cfg.seed)
    OmegaConf.set_struct(cfg, False)
    cfg = OmegaConf.merge(cfg, cfg.method)
    cfg['eval'] = True

    model = build_model(cfg)
    val_set = build_dataset(cfg, val=True)

    eval_batch_size = cfg.method['eval_batch_size']
    val_loader = DataLoader(
        val_set, batch_size=eval_batch_size, num_workers=cfg.load_num_workers,
        shuffle=False, drop_last=False, collate_fn=val_set.collate_fn)

    trainer = pl.Trainer(
        inference_mode=True,
        logger=False,
        devices=1 if cfg.debug else cfg.devices,
        accelerator="cpu" if cfg.debug else "gpu",
        enable_checkpointing=False,
    )

    # Original .ckpt is loaded by Lightning; the model hook handles removed training-only keys.
    trainer.validate(model=model, dataloaders=val_loader, ckpt_path=cfg.ckpt_path, weights_only=False)


if __name__ == '__main__':
    evaluation()
