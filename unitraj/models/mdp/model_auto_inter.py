import torch
import torch.nn as nn

from .layers.agent_embedding import AgentEmbeddingLayer
from .layers.lane_embedding import LaneEmbeddingLayer
from .layers.transformer_blocks import Block

from .layers.memory_module_inter import MemModule
from .layers.entropy_loss import EntropyLossEncap


class ModelForecast_Auto_Inter(nn.Module):
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
        lane_steps=20,
    ) -> None:
        super().__init__()

        # Encode scene features.
        self.hist_embed = AgentEmbeddingLayer(
            4, embed_dim // 4, drop_path_rate=drop_path
        )
        self.lane_embed = LaneEmbeddingLayer(3, embed_dim)

        self.pos_embed = nn.Sequential(
            nn.Linear(4, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path, encoder_depth)]
        self.blocks = nn.ModuleList(
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=dpr[i],
            )
            for i in range(encoder_depth)
        )
        self.norm = nn.LayerNorm(embed_dim)

        self.actor_type_embed = nn.Parameter(torch.Tensor(4, embed_dim))
        self.lane_type_embed = nn.Parameter(torch.Tensor(1, 1, embed_dim))

        self.future_steps = future_steps
        self.historical_steps = historical_steps
        self.lane_steps = lane_steps

        # Decode scene features.
        self.decoder_blocks = nn.ModuleList(
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=dpr[i],
            )
            for i in range(encoder_depth)
        )
        self.decoder_norm = nn.LayerNorm(embed_dim)

        self.agent_inter = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, self.historical_steps * 2),
        )

        self.lane_inter = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, self.lane_steps * 2),
        )

        # Memory features.
        self.mem_rep_road = MemModule(mem_dim=64, fea_dim=embed_dim, shrink_thres = 0.03)
        self.mem_rep_agent = MemModule(mem_dim=64, fea_dim=embed_dim, shrink_thres = 0.03)
        self.entropy_loss = EntropyLossEncap()
        self.recover_loss = nn.MSELoss()

        # Retain the original element-wise reconstruction loss without writing files.
        self.inter_recover_loss = nn.MSELoss(reduction='none')

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.normal_(self.actor_type_embed, std=0.02)
        nn.init.normal_(self.lane_type_embed, std=0.02)
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

        # Encode scene features.
        hist_padding_mask = data["x_padding_mask"][:, :, :self.historical_steps]
        hist_key_padding_mask = data["x_key_padding_mask"]
        hist_feat = torch.cat(
            [
                data["x"],
                data["x_velocity_diff"][..., None],
                ~hist_padding_mask[..., None],
            ],
            dim=-1,
        )

        B, N, L, D = hist_feat.shape
        hist_feat = hist_feat.view(B * N, L, D)
        hist_feat_key_padding = hist_key_padding_mask.view(B * N)
        actor_feat = self.hist_embed(
            hist_feat[~hist_feat_key_padding].permute(0, 2, 1).contiguous()
        )
        actor_feat_tmp = torch.zeros(
            B * N, actor_feat.shape[-1], device=actor_feat.device
        )
        actor_feat_tmp[~hist_feat_key_padding] = actor_feat
        actor_feat = actor_feat_tmp.view(B, N, actor_feat.shape[-1])

        lane_padding_mask = data["lane_padding_mask"]
        lane_normalized = data["lane_positions"] - data["lane_centers"].unsqueeze(-2)
        lane_normalized = torch.cat(
            [lane_normalized, ~lane_padding_mask[..., None]], dim=-1
        )
        B, M, L, D = lane_normalized.shape
        lane_feat = self.lane_embed(lane_normalized.view(-1, L, D).contiguous())
        lane_feat = lane_feat.view(B, M, -1)

        x_centers = torch.cat([data["x_centers"], data["lane_centers"]], dim=1)
        angles = torch.cat([data["x_angles"][:, :, self.historical_steps - 1], data["lane_angles"]], dim=1)
        x_angles = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
        pos_feat = torch.cat([x_centers, x_angles], dim=-1)
        pos_embed = self.pos_embed(pos_feat)

        actor_type_embed = self.actor_type_embed[data["x_attr"][..., 2].long()]
        lane_type_embed = self.lane_type_embed.repeat(B, M, 1)
        actor_feat += actor_type_embed
        lane_feat += lane_type_embed

        x_encoder = torch.cat([actor_feat, lane_feat], dim=1)
        key_padding_mask = torch.cat(
            [data["x_key_padding_mask"], data["lane_key_padding_mask"]], dim=1
        )

        x_encoder = x_encoder + pos_embed
        for blk in self.blocks:
            x_encoder = blk(x_encoder, key_padding_mask=key_padding_mask)
        x_encoder = self.norm(x_encoder)

        # Map features through the memory module.
        # Suppress padded-agent noise.
        actor_feat = x_encoder[:, :N].reshape(B * N, -1)[~hist_feat_key_padding]
        res_mem_agent = self.mem_rep_agent(actor_feat.unsqueeze(1).permute(0, 2, 1)) 
        actor_feat = res_mem_agent['output'].permute(0, 2, 1).squeeze(1)
        actor_att = res_mem_agent['att']

        actor_feat_tmp = torch.zeros(
            B * N, actor_feat.shape[-1], device=actor_feat.device)
        actor_feat_tmp[~hist_feat_key_padding] = actor_feat
        actor_feat = actor_feat_tmp.view(B, N, actor_feat.shape[-1])


        # Suppress padded-lane noise.
        lane_mask = ((~lane_padding_mask).sum(dim=2) > 0).view(B * M)
        lane_feat = x_encoder[:, N:].reshape(B * M, -1)[lane_mask]
        res_mem_road = self.mem_rep_road(lane_feat.unsqueeze(1).permute(0, 2, 1)) 
        lane_feat = res_mem_road['output'].permute(0, 2, 1).squeeze(1)  
        lane_att = res_mem_road['att'] 

        lane_feat_tmp = torch.zeros(
            B * M, lane_feat.shape[-1], device=lane_feat.device) 
        lane_feat_tmp[lane_mask] = lane_feat
        lane_feat = lane_feat_tmp.view(B, M, lane_feat.shape[-1])

        # Concatenate features.
        x_encoder = torch.cat([actor_feat, lane_feat], dim=1)

        # Decode features.
        for blk in self.decoder_blocks:
            x_encoder = blk(x_encoder, key_padding_mask=key_padding_mask)
        x_encoder = self.decoder_norm(x_encoder)

        y_target = self.agent_inter(x_encoder[:, 0]).view(B, self.historical_steps, 2)
        target_gt = data["x"][:, 0]

        y_others = self.agent_inter(x_encoder[:, 1:N]).view(B, N-1, self.historical_steps, 2)
        others_gt = data["x"][:, 1:]

        y_lane = self.lane_inter(x_encoder[:, N:]).view(B, M, self.lane_steps, 2)
        lane_gt = data["lane_positions"] - data["lane_centers"].unsqueeze(-2)

        #===================================================================
        agent_entropy_loss = self.entropy_loss(actor_att)
        target_reg_loss = self.recover_loss(y_target, target_gt)

        others_reg_mask = ~data["x_padding_mask"][:, 1:, :self.historical_steps]
        others_reg_loss = self.recover_loss(y_others[others_reg_mask], others_gt[others_reg_mask])

        lane_entropy_loss = self.entropy_loss(lane_att)
        lane_reg_mask = ~lane_padding_mask
        lane_reg_loss = self.recover_loss(y_lane[lane_reg_mask], lane_gt[lane_reg_mask])

        loss_all = target_reg_loss + others_reg_loss + 0.03 * lane_reg_loss + 0.0002 * (agent_entropy_loss + lane_entropy_loss)


        return {
            "loss_all": loss_all,
            "target_reg_loss": target_reg_loss,
            "others_reg_loss": others_reg_loss,
            "lane_reg_loss": lane_reg_loss,
            "agent_entropy_loss": agent_entropy_loss,
            "lane_entropy_loss": lane_entropy_loss
        }
