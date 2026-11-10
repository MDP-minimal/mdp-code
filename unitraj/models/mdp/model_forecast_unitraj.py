import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers.agent_embedding import AgentEmbeddingLayer
from .layers.lane_embedding import LaneEmbeddingLayer
from .layers.transformer_blocks import Block, LongTailPrompt, PromptBlock


class SceneDifficultyClassifier(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.LayerNorm(dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(dim // 2, dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(dim // 4, 3) 
        )

    def forward(self, x_encoder):
        target_feat = x_encoder[:, 0]
        logits = self.classifier(target_feat)
        return logits



class ModelForecast(nn.Module):
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

        
        self.multi_modal_query_embedding = nn.Embedding(6, embed_dim)
        self.register_buffer('modal', torch.arange(6).long())

        
        self.lt_prompt = LongTailPrompt(
            emb_dim=embed_dim,
            n_classes=3,           
            pool_size_per_class=10,   
            prompt_length=40,          
            key_dim=embed_dim,        
            ortho_mu=0.1              
        )

        
        self.cdf_thr_simple = 0.9485
        self.cdf_thr_complex = 1.2606
        self.difficulty_classifier = SceneDifficultyClassifier(embed_dim)

        
        dpr = [x.item() for x in torch.linspace(0, drop_path, 4)]

        self.cross_block_mode = nn.ModuleList(
            PromptBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=dpr[i],
            )
            for i in range(3)
        )
        
        self.self_block_mode = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=dpr[3],
                )
        

        
        self.future_others = nn.Sequential(
            nn.Linear(embed_dim, 256), nn.ReLU(), nn.Linear(256, future_steps * 2)
        )
        self.target_future = nn.Sequential(
            nn.Linear(embed_dim, 256), nn.ReLU(), nn.Linear(256, future_steps * 2)
        )
        self.pi = nn.Sequential(
            nn.Linear(embed_dim, 256), nn.ReLU(), nn.Linear(256, 1)
        )
        

        
        # Original contrastive projection heads.
        self.linear_cla = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.ReLU(),
                nn.Linear(embed_dim // 2, embed_dim)
            )
        
        self.linear_clm = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.ReLU(),
                nn.Linear(embed_dim // 2, embed_dim)
            )
        
        self.linear_clia = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.ReLU(),
                nn.Linear(embed_dim // 2, embed_dim)
            )
        
        self.linear_clil = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.ReLU(),
                nn.Linear(embed_dim // 2, embed_dim)
            )
        
        # =========================================================================
        
        
        # Retain the original queues for training and full checkpoint compatibility.
        queue_size = 512
        
        
        self.register_buffer('clm_feat_queue', torch.zeros(queue_size, embed_dim))
        self.register_buffer('clm_label_queue', torch.full((queue_size,), -1.0, dtype=torch.float))
        self.register_buffer('clm_queue_ptr', torch.zeros(1, dtype=torch.long))

        
        self.register_buffer('feat_queue', torch.zeros(queue_size, embed_dim))
        self.register_buffer('label_queue', torch.full((queue_size,), -1, dtype=torch.long))
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))

        
        self.register_buffer('clia_feat_queue', torch.zeros(queue_size, embed_dim))
        self.register_buffer('clia_label_queue', torch.full((queue_size,), -1.0, dtype=torch.float))
        self.register_buffer('clia_queue_ptr', torch.zeros(1, dtype=torch.long))

        
        self.register_buffer('clil_feat_queue', torch.zeros(queue_size, embed_dim))
        self.register_buffer('clil_label_queue', torch.full((queue_size,), -1.0, dtype=torch.float))
        self.register_buffer('clil_queue_ptr', torch.zeros(1, dtype=torch.long))

        # =========================================================================

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
        angles = torch.cat([data["x_angles"][:, :, self.historical_steps-1], data["lane_angles"]], dim=1)
        x_angles = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
        pos_feat = torch.cat([x_centers, x_angles], dim=-1)
        pos_embed = self.pos_embed(pos_feat)

        actor_type_embed = self.actor_type_embed[data["x_attr"][..., 2].long()]
        lane_type_embed = self.lane_type_embed.repeat(B, M, 1)
        actor_feat += actor_type_embed
        lane_feat += lane_type_embed

        # ==============================================================
        
        # CLM: motion contrastive objective.
        if self.training == False:
            clm_loss = 0
        else:
            motion_feat = actor_feat[:, 0]
            motion_feat = self.linear_clm(motion_feat)
            motion_feat = F.normalize(motion_feat, dim=-1)

            
            motion_class = data["dynamic"] 
            t = 0.05
            eps = 1e-8
            
            
            queue_size = self.clm_feat_queue.shape[0]
            ptr = int(self.clm_queue_ptr.item())
            batch_size = motion_feat.shape[0]

            
            valid_size = ptr
            
            if valid_size > 0:
                all_feats = torch.cat([motion_feat, self.clm_feat_queue[:valid_size].detach()], dim=0)
                all_labels = torch.cat([motion_class, self.clm_label_queue[:valid_size].detach()], dim=0)
            else:
                all_feats = motion_feat
                all_labels = motion_class

            total_size = all_labels.shape[0]

            
            sim_matrix = F.cosine_similarity(motion_feat.unsqueeze(1), all_feats.unsqueeze(0), dim=-1) / t
            sim_exp = torch.exp(sim_matrix)

            
            anchor_labels = motion_class.unsqueeze(1) 
            key_labels = all_labels.unsqueeze(0)      
            
            abs_diff = torch.abs(anchor_labels - key_labels)
            
            self_mask = torch.zeros_like(abs_diff, dtype=torch.bool)
            self_mask_diag = torch.eye(batch_size, dtype=torch.bool, device=motion_feat.device)
            self_mask[:, :batch_size] = self_mask_diag
            
            abs_diff_no_self = abs_diff.masked_fill(self_mask, float('inf'))
            
            total_samples_to_consider = total_size - 1
            if total_samples_to_consider < 2:
                clm_loss = torch.tensor(0.0, device=motion_feat.device)
            else:
                k_pos = max(1, int(total_samples_to_consider * 0.1))
                k_neg = max(1, int(total_samples_to_consider * 0.4))
                
                sorted_diffs, sorted_indices = torch.sort(abs_diff_no_self, dim=-1, descending=False)
                
                pos_indices = sorted_indices[:, :k_pos] 
                neg_indices = sorted_indices[:, :-1][:, -k_neg:] 

                pos_mask = torch.zeros_like(abs_diff, dtype=torch.bool)
                neg_mask = torch.zeros_like(abs_diff, dtype=torch.bool)
                
                pos_mask.scatter_(dim=1, index=pos_indices, value=True)
                neg_mask.scatter_(dim=1, index=neg_indices, value=True)
                
                pos_sim = (sim_exp * pos_mask).sum(dim=-1)
                neg_sim = (sim_exp * neg_mask).sum(dim=-1) 

                clm_loss = -torch.log((pos_sim + eps) / (neg_sim + pos_sim + eps)).mean()
            # ========================================================
            
            
            if ptr + batch_size <= queue_size:
                self.clm_feat_queue[ptr:ptr+batch_size] = motion_feat.detach()
                self.clm_label_queue[ptr:ptr+batch_size] = motion_class.detach()
                ptr += batch_size
            else:
                first_len = queue_size - ptr
                self.clm_feat_queue[ptr:] = motion_feat[:first_len].detach()
                self.clm_label_queue[ptr:] = motion_class[:first_len].detach()

                remain = batch_size - first_len
                if remain > 0:
                    self.clm_feat_queue[:queue_size-remain] = self.clm_feat_queue[remain:].detach().clone()
                    self.clm_label_queue[:queue_size-remain] = self.clm_label_queue[remain:].detach().clone()
                    self.clm_feat_queue[queue_size-remain:] = motion_feat[first_len:].detach()
                    self.clm_label_queue[queue_size-remain:] = motion_class[first_len:].detach()
                ptr = queue_size

            self.clm_queue_ptr[0] = ptr

        # # ==============================================================

        x_encoder = torch.cat([actor_feat, lane_feat], dim=1)
        key_padding_mask = torch.cat(
            [data["x_key_padding_mask"], data["lane_key_padding_mask"]], dim=1
        )

        x_encoder = x_encoder + pos_embed
        for blk in self.blocks:
            x_encoder = blk(x_encoder, key_padding_mask=key_padding_mask)
        x_encoder = self.norm(x_encoder)

        # ==============================================================
        
        # CLA: asymmetric head/tail contrastive objective.
        if self.training == False:
            cla_loss = 0
        else:
            scene_feat = x_encoder[:, 0]
            scene_feat = self.linear_cla(scene_feat)
            scene_feat = F.normalize(scene_feat, dim=-1)
            scene_class = data['head']

            t = 0.05
            eps = 1e-8

            
            queue_size = self.feat_queue.shape[0]
            ptr = int(self.queue_ptr.item())
            batch_size = scene_feat.shape[0]

            
            valid_size = ptr
            # print(valid_size)
            if valid_size > 0:
                all_feats = torch.cat([scene_feat, self.feat_queue[:valid_size]], dim=0)
                all_labels = torch.cat([scene_class, self.label_queue[:valid_size]], dim=0)
            else:
                all_feats = scene_feat
                all_labels = scene_class

            sim_matrix = F.cosine_similarity(scene_feat.unsqueeze(1), all_feats.unsqueeze(0), dim=-1) / t
            sim_exp = torch.exp(sim_matrix)
       
            anchor_labels = scene_class.unsqueeze(1)
            key_labels = all_labels.unsqueeze(0)

            pos_mask = (anchor_labels == 1) & (key_labels == 1)
            self_mask = torch.eye(scene_feat.shape[0], dtype=torch.bool, device=scene_feat.device)
            pos_mask[:, :scene_feat.shape[0]] &= ~self_mask

            neg_mask = (anchor_labels == 1) & (key_labels == 0)

            pos_sim = (sim_exp * pos_mask).sum(dim=-1)
            neg_sim = (sim_exp * neg_mask).sum(dim=-1)

            cla_loss = -torch.log((pos_sim + eps) / (neg_sim + pos_sim + eps)).mean()

            
            if ptr + batch_size <= queue_size:
                self.feat_queue[ptr:ptr+batch_size] = scene_feat.detach()
                self.label_queue[ptr:ptr+batch_size] = scene_class.detach()
                ptr += batch_size
            else:
                first_len = queue_size - ptr
                self.feat_queue[ptr:] = scene_feat[:first_len].detach()
                self.label_queue[ptr:] = scene_class[:first_len].detach()

                remain = batch_size - first_len
                if remain > 0:
                    self.feat_queue[:queue_size-remain] = self.feat_queue[remain:].detach().clone()
                    self.label_queue[:queue_size-remain] = self.label_queue[remain:].detach().clone()
                    self.feat_queue[queue_size-remain:] = scene_feat[first_len:].detach()
                    self.label_queue[queue_size-remain:] = scene_class[first_len:].detach()
                ptr = queue_size

            self.queue_ptr[0] = ptr

        # ==============================================================
        # CLI: agent and lane interaction contrastive objectives.
        if self.training == False:
            cli_loss = 0
        else:
            
            inter_feat = x_encoder[:, :N].mean(dim=1)
            inter_feat = self.linear_clia(inter_feat)
            inter_feat = F.normalize(inter_feat, dim=-1)

            inter_class = data["i_others"] + data["i_target"]

            t = 0.05 
            eps = 1e-8

            
            queue_size = self.clia_feat_queue.shape[0]
            ptr = int(self.clia_queue_ptr.item())
            batch_size = inter_feat.shape[0]

            valid_size = ptr
            if valid_size > 0:
                all_feats = torch.cat([inter_feat, self.clia_feat_queue[:valid_size].detach()], dim=0)
                all_labels = torch.cat([inter_class, self.clia_label_queue[:valid_size].detach()], dim=0)
            else:
                all_feats = inter_feat
                all_labels = inter_class
            
            total_size = all_labels.shape[0]  

            sim_matrix = F.cosine_similarity(inter_feat.unsqueeze(1), all_feats.unsqueeze(0), dim=-1) / t
            sim_exp = torch.exp(sim_matrix)

            
            anchor_labels = inter_class.unsqueeze(1) 
            key_labels = all_labels.unsqueeze(0)     
            
            abs_diff = torch.abs(anchor_labels - key_labels)
            
            self_mask = torch.zeros_like(abs_diff, dtype=torch.bool)
            self_mask_diag = torch.eye(batch_size, dtype=torch.bool, device=inter_feat.device)
            self_mask[:, :batch_size] = self_mask_diag
            
            abs_diff_no_self = abs_diff.masked_fill(self_mask, float('inf'))
            
            total_samples_to_consider = total_size - 1

            if total_samples_to_consider < 2:
                clia_loss = torch.tensor(0.0, device=inter_feat.device)
            else:
                k_pos = max(1, int(total_samples_to_consider * 0.1))
                k_neg = max(1, int(total_samples_to_consider * 0.4))
                
                sorted_diffs, sorted_indices = torch.sort(abs_diff_no_self, dim=-1, descending=False)
                
                pos_indices = sorted_indices[:, :k_pos] 
                neg_indices = sorted_indices[:, :-1][:, -k_neg:] 

                pos_mask = torch.zeros_like(abs_diff, dtype=torch.bool)
                neg_mask = torch.zeros_like(abs_diff, dtype=torch.bool)
                
                pos_mask.scatter_(dim=1, index=pos_indices, value=True)
                neg_mask.scatter_(dim=1, index=neg_indices, value=True)

                pos_sim = (sim_exp * pos_mask).sum(dim=-1)
                neg_sim = (sim_exp * neg_mask).sum(dim=-1) 

                clia_loss = -torch.log((pos_sim + eps) / (neg_sim + pos_sim + eps)).mean()
            # ========================================================
            
            
            if ptr + batch_size <= queue_size:
                self.clia_feat_queue[ptr:ptr+batch_size] = inter_feat.detach()
                self.clia_label_queue[ptr:ptr+batch_size] = inter_class.detach()
                ptr += batch_size
            else:
                first_len = queue_size - ptr
                self.clia_feat_queue[ptr:] = inter_feat[:first_len].detach()
                self.clia_label_queue[ptr:] = inter_class[:first_len].detach()

                remain = batch_size - first_len
                if remain > 0:
                    self.clia_feat_queue[:queue_size-remain] = self.clia_feat_queue[remain:].detach().clone()
                    self.clia_label_queue[:queue_size-remain] = self.clia_label_queue[remain:].detach().clone()
                    self.clia_feat_queue[queue_size-remain:] = inter_feat[first_len:].detach()
                    self.clia_label_queue[queue_size-remain:] = inter_class[first_len:].detach()
                ptr = queue_size

            self.clia_queue_ptr[0] = ptr
        # ==============================================================
            
            inter_feat = x_encoder[:, N:].mean(dim=1)
            inter_feat = self.linear_clil(inter_feat)
            inter_feat = F.normalize(inter_feat, dim=-1)

            inter_class = data["i_lane"] + data["i_target"]

            t = 0.05 
            eps = 1e-8

            
            queue_size = self.clil_feat_queue.shape[0]
            ptr = int(self.clil_queue_ptr.item())
            batch_size = inter_feat.shape[0]

            valid_size = ptr
            if valid_size > 0:
                all_feats = torch.cat([inter_feat, self.clil_feat_queue[:valid_size].detach()], dim=0)
                all_labels = torch.cat([inter_class, self.clil_label_queue[:valid_size].detach()], dim=0)
            else:
                all_feats = inter_feat
                all_labels = inter_class
            
            total_size = all_labels.shape[0]  

            sim_matrix = F.cosine_similarity(inter_feat.unsqueeze(1), all_feats.unsqueeze(0), dim=-1) / t
            sim_exp = torch.exp(sim_matrix)

            
            anchor_labels = inter_class.unsqueeze(1) 
            key_labels = all_labels.unsqueeze(0)     
            
            abs_diff = torch.abs(anchor_labels - key_labels)
            
            self_mask = torch.zeros_like(abs_diff, dtype=torch.bool)
            self_mask_diag = torch.eye(batch_size, dtype=torch.bool, device=inter_feat.device)
            self_mask[:, :batch_size] = self_mask_diag
            
            abs_diff_no_self = abs_diff.masked_fill(self_mask, float('inf'))
            
            total_samples_to_consider = total_size - 1

            if total_samples_to_consider < 2:
                clil_loss = torch.tensor(0.0, device=inter_feat.device)
            else:
                k_pos = max(1, int(total_samples_to_consider * 0.1))
                k_neg = max(1, int(total_samples_to_consider * 0.4))
                
                sorted_diffs, sorted_indices = torch.sort(abs_diff_no_self, dim=-1, descending=False)
                
                pos_indices = sorted_indices[:, :k_pos] 
                neg_indices = sorted_indices[:, :-1][:, -k_neg:] 

                pos_mask = torch.zeros_like(abs_diff, dtype=torch.bool)
                neg_mask = torch.zeros_like(abs_diff, dtype=torch.bool)
                
                pos_mask.scatter_(dim=1, index=pos_indices, value=True)
                neg_mask.scatter_(dim=1, index=neg_indices, value=True)

                pos_sim = (sim_exp * pos_mask).sum(dim=-1)
                neg_sim = (sim_exp * neg_mask).sum(dim=-1) 

                clil_loss = -torch.log((pos_sim + eps) / (neg_sim + pos_sim + eps)).mean()
            # ========================================================
            
            
            if ptr + batch_size <= queue_size:
                self.clil_feat_queue[ptr:ptr+batch_size] = inter_feat.detach()
                self.clil_label_queue[ptr:ptr+batch_size] = inter_class.detach()
                ptr += batch_size
            else:
                first_len = queue_size - ptr
                self.clil_feat_queue[ptr:] = inter_feat[:first_len].detach()
                self.clil_label_queue[ptr:] = inter_class[:first_len].detach()

                remain = batch_size - first_len
                if remain > 0:
                    self.clil_feat_queue[:queue_size-remain] = self.clil_feat_queue[remain:].detach().clone()
                    self.clil_label_queue[:queue_size-remain] = self.clil_label_queue[remain:].detach().clone()
                    self.clil_feat_queue[queue_size-remain:] = inter_feat[first_len:].detach()
                    self.clil_label_queue[queue_size-remain:] = inter_class[first_len:].detach()
                ptr = queue_size

            self.clil_queue_ptr[0] = ptr

            cli_loss = clia_loss + clil_loss
        # # ========================================================
        
        y_hat_others = self.future_others(x_encoder[:, 1:N]).view(B, N-1, -1, 2)

        
        scene_logits = self.difficulty_classifier(x_encoder.detach())
        CDF_cls_loss = torch.tensor(0.0, device=x_encoder.device)
        prompt_labels = None

        # Use CDF targets only during training; evaluation predicts its own routing class.
        if self.training:
            
            cdf_val = torch.sqrt(data["CDF_dyn"]**2 + data["CDF_lane"]**2 + data["CDF_agent"]**2)
            gt_labels = torch.zeros(B, dtype=torch.long, device=x_encoder.device)

            mask_medium = (cdf_val >= self.cdf_thr_simple) & (cdf_val < self.cdf_thr_complex)
            gt_labels[mask_medium] = 1
            gt_labels[cdf_val >= self.cdf_thr_complex] = 2
            prompt_labels = gt_labels

            
            class_weights = torch.tensor([1.0, 1.7, 2.5], device=x_encoder.device)
            CDF_cls_loss = F.cross_entropy(scene_logits, gt_labels, weight=class_weights)

        else:
            
            prompt_labels = torch.argmax(scene_logits, dim=-1)

        
        target_agent_feat = x_encoder[:, 0]
        (Ek, Ev), prompt_ortho_loss = self.lt_prompt(
            target_agent_feat, 
            class_labels=prompt_labels, 
            train=self.training
        )

        # 
        multi_modal_query = self.multi_modal_query_embedding(self.modal)
        mode_query = x_encoder[:, 0]
        mode = mode_query[:, None] + multi_modal_query

        
        for blk in self.cross_block_mode:
            mode = blk(mode, x_encoder, key_padding_mask=key_padding_mask, prompt=(Ek, Ev))

        mode = self.self_block_mode(mode)
        
        y_hat = self.target_future(mode).view(B, 6, self.future_steps, 2)
        pi = self.pi(mode).squeeze(-1)

        return {
            "y_hat": y_hat,  # trajectory output from mode query
            "pi": pi,  # probability output from mode query
            "y_hat_others": y_hat_others,  # trajectory of other agents
            "cla_loss": cla_loss,
            "clm_loss": clm_loss,
            "cli_loss": cli_loss, 
            "CDF_cls_loss": CDF_cls_loss,
            "prompt_ortho_loss": prompt_ortho_loss
        }