import torch
import torch.nn as nn

from .utils.optim import WarmupCosLR

from .model_auto_inter import ModelForecast_Auto_Inter
from unitraj.models.base_model.base_model import BaseModel


class Trainer_auto_inter(BaseModel):
    def __init__(
        self,
        config = None,
        dim=128,
        historical_steps=50,
        future_steps=60,
        encoder_depth=4,
        num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop_path=0.2,
        pretrained_weights: str = None,
        autoencoder_checkpoint: str = None,
        lr: float = 1e-3,
        warmup_epochs: int = 10,
        epochs: int = 60,
        weight_decay: float = 1e-4,
    ) -> None:
        super(Trainer_auto_inter, self).__init__(config=config)
        
        ## for UniTraj integration
        if config is not None:
            dim = config["dim"]
            historical_steps = config["past_len"]
            future_steps = config["future_len"]
            encoder_depth = config["num_encoder_layers"]
            num_heads = config["tx_num_heads"]
            mlp_ratio = config["mlp_ratio"]
            qkv_bias = config["qkv_bias"]
            drop_path = config["drop_path"]
            epochs = config["max_epochs"]
            warmup_epochs = config["warmup_epochs"]
            lr = config["learning_rate"]
            weight_decay = config["weight_decay"]
            pretrained_weights = config["pretrained_weights"]

        self.warmup_epochs = warmup_epochs
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.save_hyperparameters()

        self.net = ModelForecast_Auto_Inter(
            embed_dim=dim,
            encoder_depth=encoder_depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_path=drop_path,
            future_steps=future_steps,
            historical_steps=historical_steps,
            lane_steps=20
        )

        self._load_optional_checkpoint(pretrained_weights)
        self._load_optional_checkpoint(autoencoder_checkpoint)

    def _load_optional_checkpoint(self, checkpoint_path):
        """Load only an explicitly supplied checkpoint; never choose a default."""
        if checkpoint_path is None or checkpoint_path == "":
            return
        self.net.load_from_checkpoint(checkpoint_path)

    def forward(self, data):
        converted_data = self.convert_to_fmae_format(data['input_dict'].copy())
        return self.net(converted_data)

    def predict(self, data):
        with torch.no_grad():
            converted_data = self.convert_to_fmae_format(data['input_dict'].copy())
            out = self.net(converted_data)
        return out

    def cal_loss(self, out, data):

        loss = out["loss_all"]

        return {
            "loss": loss,
            "target_reg_loss": out["target_reg_loss"],
            "others_reg_loss": out["others_reg_loss"],
            "lane_reg_loss": out["lane_reg_loss"],
            "agent_entropy_loss": out["agent_entropy_loss"],
            "lane_entropy_loss": out["lane_entropy_loss"],
            "entropy_loss": out["lane_entropy_loss"] + out["agent_entropy_loss"]
        }

    def training_step(self, data, batch_idx):
        out = self(data)
        converted_data = self.convert_to_fmae_format(data['input_dict'].copy()) ##TODO: maybe only do it once, not here and in forward?
        #fmae
        losses = self.cal_loss(out, converted_data)

        self.log("train_loss", losses["loss"], on_epoch=True, batch_size=8, sync_dist=True)
        self.log("train_target_reg_loss", losses["target_reg_loss"], on_epoch=True, batch_size=8, sync_dist=True)       
        self.log("train_others_reg_loss", losses["others_reg_loss"], on_epoch=True, batch_size=8, sync_dist=True)
        self.log("train_lane_reg_loss", losses["lane_reg_loss"], on_epoch=True, batch_size=8, sync_dist=True)
        self.log("train_agent_entropy_loss", losses["agent_entropy_loss"], on_epoch=True, batch_size=8, sync_dist=True)
        self.log("train_lane_entropy_loss", losses["lane_entropy_loss"], on_epoch=True, batch_size=8, sync_dist=True)
        self.log("train_entropy_loss", losses["entropy_loss"], on_epoch=True, batch_size=8, sync_dist=True)

        return losses["loss"]

    def validation_step(self, data, batch_idx):
        out = self(data)
        converted_data = self.convert_to_fmae_format(data['input_dict'].copy()) 
        losses = self.cal_loss(out, converted_data)

        self.log("val_loss", losses["loss"], on_epoch=True, batch_size=8, sync_dist=True)
        self.log("val_target_reg_loss", losses["target_reg_loss"], on_epoch=True, batch_size=8, sync_dist=True)       
        self.log("val_others_reg_loss", losses["others_reg_loss"], on_epoch=True, batch_size=8, sync_dist=True)
        self.log("val_lane_reg_loss", losses["lane_reg_loss"], on_epoch=True, batch_size=8, sync_dist=True)
        self.log("val_agent_entropy_loss", losses["agent_entropy_loss"], on_epoch=True, batch_size=8, sync_dist=True)
        self.log("val_lane_entropy_loss", losses["lane_entropy_loss"], on_epoch=True, batch_size=8, sync_dist=True)
        self.log("val_entropy_loss", losses["entropy_loss"], on_epoch=True, batch_size=8, sync_dist=True)

    def test_step(self, data, batch_idx):
        return self.validation_step(data, batch_idx)

    def configure_optimizers(self):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (
            nn.Linear,
            nn.Conv1d,
            nn.Conv2d,
            nn.Conv3d,
            nn.MultiheadAttention,
            nn.LSTM,
            nn.GRU,
        )
        blacklist_weight_modules = (
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.SyncBatchNorm,
            nn.LayerNorm,
            nn.Embedding,
        )
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                full_param_name = (
                    "%s.%s" % (module_name, param_name) if module_name else param_name
                )
                if "bias" in param_name:
                    no_decay.add(full_param_name)
                elif "weight" in param_name:
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_param_name)
                elif not ("weight" in param_name or "bias" in param_name):
                    no_decay.add(full_param_name)
        param_dict = {
            param_name: param for param_name, param in self.named_parameters()
        }
        inter_params = decay & no_decay
        union_params = decay | no_decay
        # assert len(inter_params) == 0
        # assert len(param_dict.keys() - union_params) == 0

        optim_groups = [
            {
                "params": [
                    param_dict[param_name] for param_name in sorted(list(decay))
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    param_dict[param_name] for param_name in sorted(list(no_decay))
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(
            optim_groups, lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = WarmupCosLR(
            optimizer=optimizer,
            lr=self.lr,
            min_lr=1e-6,
            warmup_epochs=self.warmup_epochs,
            epochs=self.epochs,
        )
        return [optimizer], [scheduler]
