import torch
import torch.nn as nn

from .layers.agent_embedding import AgentEmbeddingLayer
from .layers.transformer_blocks import Block

from .layers.memory_module import MemModule
from .layers.entropy_loss import EntropyLossEncap


class ModelForecast_Auto(nn.Module):
    def __init__(
        self,
        embed_dim=128,
        encoder_depth=4,
        num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop_path=0.2,
        future_steps: int = 60,
        historical_steps: int = 50,
    ) -> None:
        super().__init__()


        self.future_steps = future_steps
        self.historical_steps = historical_steps
        # Encode the target-agent trajectory for reconstruction.
        self.hist_embed_target = AgentEmbeddingLayer(
            4, embed_dim // 4, drop_path_rate=drop_path
        )
        self.actor_type_embed_target = nn.Parameter(torch.Tensor(4, embed_dim))

        # Memory features.
        self.mem_rep = MemModule(mem_dim=64, fea_dim=embed_dim, shrink_thres = 0.02)
        self.target_entropy_loss = EntropyLossEncap()
        self.target_recover_loss = nn.MSELoss()

        # Retain the original element-wise reconstruction loss without writing files.
        self.dynamic_reocver_loss = nn.MSELoss(reduction='none')

        # Decode the target-agent trajectory reconstruction.
        dpr = [x.item() for x in torch.linspace(0, drop_path, 2)]
        self.decoder_target = nn.ModuleList(Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=dpr[i],
            )
            for i in range(2)
        )
        self.norm_target = nn.LayerNorm(embed_dim)

        self.target_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, self.historical_steps * 2),
        )

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.normal_(self.actor_type_embed_target, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def load_from_checkpoint(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        state_dict = {
            k[len("net.") :]: v for k, v in ckpt.items() if k.startswith("net.")
        }
        return self.load_state_dict(state_dict=state_dict, strict=False)

    def forward(self, data):

        #===================================================================
        # Encode features.
        hist_padding_mask = data["x_padding_mask"][:, :, :self.historical_steps][:, 0].unsqueeze(1)
        hist_key_padding_mask = data["x_key_padding_mask"][:, 0].unsqueeze(1)
        hist_feat = torch.cat(
            [
                data["x"][:, 0],
                data["x_velocity_diff"][..., None][:, 0],
                ~hist_padding_mask[..., None].squeeze(1),
            ],
            dim=-1,
        ).unsqueeze(1)

        B, N, L, D = hist_feat.shape
        hist_feat = hist_feat.view(B * N, L, D)
        hist_feat_key_padding = hist_key_padding_mask.view(B * N)
        # actor_feat = self.hist_embed(
        actor_feat = self.hist_embed_target(
            hist_feat[~hist_feat_key_padding].permute(0, 2, 1).contiguous()
        )
        actor_feat_tmp = torch.zeros(
            B * N, actor_feat.shape[-1], device=actor_feat.device
        )
        actor_feat_tmp[~hist_feat_key_padding] = actor_feat
        actor_feat = actor_feat_tmp.view(B * N, actor_feat.shape[-1])

        # actor_type_embed = self.actor_type_embed[data["x_attr"][..., 2][:, 0].long()]
        actor_type_embed = self.actor_type_embed_target[data["x_attr"][..., 2][:, 0].long()]
        actor_feat += actor_type_embed
        actor_feat = actor_feat.unsqueeze(1)

        # Map features through the memory module.
        res_mem = self.mem_rep(actor_feat.permute(0, 2, 1))
        actor_feat = res_mem['output'].permute(0, 2, 1)
        att = res_mem['att']

        # Decode features.
        for blk in self.decoder_target:
            actor_feat = blk(actor_feat, key_padding_mask=data["x_key_padding_mask"][:, 0].unsqueeze(1))
        actor_feat = self.norm_target(actor_feat)

        y_dynamic = self.target_mlp(actor_feat).view(B, self.historical_steps, 2)
        # Match target and prediction lengths if data['x'] includes future steps.
        dynamic_gt = data["x"][:, 0, :self.historical_steps] 

        #===================================================================
        dynamic_entropy_loss = self.target_entropy_loss(att)
        dynamic_reg_loss = self.target_recover_loss(y_dynamic, dynamic_gt)
        loss_dynamic = dynamic_reg_loss + 0.0002 * dynamic_entropy_loss
        

        return {
            "loss_dynamic": loss_dynamic,
            "dynamic_reg_loss": dynamic_reg_loss,
            "dynamic_entropy_loss": dynamic_entropy_loss,
        }
