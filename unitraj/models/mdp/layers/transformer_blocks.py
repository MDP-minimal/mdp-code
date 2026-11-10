from typing import Optional

import torch
import torch.nn as nn
from timm.models.layers import DropPath
from torch import Tensor
import copy
import torch.nn.functional as F

class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        post_norm=False,
    ):
        super().__init__()
        self.post_norm = post_norm

        self.norm1 = norm_layer(dim)
        self.attn = torch.nn.MultiheadAttention(
            dim,
            num_heads=num_heads,
            add_bias_kv=qkv_bias,
            dropout=attn_drop,
            batch_first=True,
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward_pre(
        self,
        src,
        mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
    ):
        src2 = self.norm1(src)
        src2 = self.attn(
            query=src2,
            key=src2,
            value=src2,
            attn_mask=mask,
            key_padding_mask=key_padding_mask,
        )[0]
        src = src + self.drop_path1(src2)
        src = src + self.drop_path2(self.mlp(self.norm2(src)))
        return src

    def forward_post(
        self,
        src,
        mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
    ):
        src2 = self.attn(
            query=src,
            key=src,
            value=src,
            attn_mask=mask,
            key_padding_mask=key_padding_mask,
        )[0]
        src = src + self.drop_path1(self.norm1(src2))
        src = src + self.drop_path2(self.norm2(self.mlp(src)))
        return src

    def forward(
        self,
        src,
        mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
    ):
        if self.post_norm:
            return self.forward_post(
                src=src, mask=mask, key_padding_mask=key_padding_mask
            )

        return self.forward_pre(src=src, mask=mask, key_padding_mask=key_padding_mask)


class Cross_Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.attn = torch.nn.MultiheadAttention(
            dim,
            num_heads=num_heads,
            add_bias_kv=qkv_bias,
            dropout=attn_drop,
            batch_first=True,
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward_pre(
        self,
        src,
        src_kv,
        mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
    ):
        src2 = self.norm1(src)
        src2_kv = self.norm1(src_kv)
        src2 = self.attn(
            query=src2,
            key=src2_kv,
            value=src2_kv,
            attn_mask=mask,
            key_padding_mask=key_padding_mask,
        )[0]
        src = src + self.drop_path1(src2)
        src = src + self.drop_path2(self.mlp(self.norm2(src)))
        return src

    def forward(
        self,
        src,
        src_kv,
        mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
    ):

        return self.forward_pre(src=src, src_kv=src_kv, mask=mask, key_padding_mask=key_padding_mask)




class PromptMultiheadAttention(nn.MultiheadAttention):
    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None, batch_first=False, device=None, dtype=None):
        super().__init__(embed_dim, num_heads, dropout, bias, add_bias_kv, add_zero_attn, kdim, vdim, batch_first, device, dtype)

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None, prompt=None, is_causal=False):
        if prompt is not None:
            pk, pv = prompt # 4,480,128 | 4,480,128
            key = torch.cat((pk,key), dim=1)
            value = torch.cat((pv,value), dim=1)
            
            if key_padding_mask is not None:
                prompt_padding = torch.zeros((pv.shape[0], pv.shape[1]), dtype=key_padding_mask.dtype, device=key_padding_mask.device)
                key_padding_mask = torch.cat((prompt_padding, key_padding_mask), dim=1)
        
        return super().forward(query, key, value, key_padding_mask, need_weights, attn_mask)


class PromptBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        post_norm=False,
    ):
        super().__init__()
        self.post_norm = post_norm

        self.norm1 = norm_layer(dim)
        self.attn = PromptMultiheadAttention(
            dim,
            num_heads=num_heads,
            add_bias_kv=qkv_bias,
            dropout=attn_drop,
            batch_first=True,
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward_pre( # Active forward implementation.
        self,
        src_q,
        src_k,
        mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        prompt: Optional[Tensor] = None,
    ):
        q = self.norm1(src_q)
        k = self.norm1(src_k)
        v = self.norm1(src_k)

        src2 = self.attn(
            query=q,
            key=k,
            value=v,
            attn_mask=mask,
            key_padding_mask=key_padding_mask,
            prompt=prompt,
        )[0]
        q = q + self.drop_path1(src2)
        q = q + self.drop_path2(self.mlp(self.norm2(q)))
        return q


    def forward(
        self,
        src_q,
        src_k,
        mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        prompt: Optional[Tensor] = None,
    ):

        return self.forward_pre(src_q=src_q, src_k=src_k, mask=mask, key_padding_mask=key_padding_mask, prompt=prompt)
    

 # Our method!
def tensor_prompt(a, b, c=None, ortho=False):
    if c is None:
        p = torch.nn.Parameter(torch.FloatTensor(a,b), requires_grad=True)
    else:
        p = torch.nn.Parameter(torch.FloatTensor(a,b,c), requires_grad=True)
    if ortho:
        nn.init.orthogonal_(p)
    else:
        nn.init.uniform_(p)
    return p  

def ortho_penalty(t):
    """Compute the orthogonality penalty."""
    return ((t @ t.T - torch.eye(t.shape[0]).cuda())**2).mean()

class LongTailPrompt(nn.Module):
    def __init__(self, 
                 emb_dim, 
                 n_classes=3,             # Head, middle, and tail classes.
                 pool_size_per_class=10,  # Number of prompt components per class.
                 prompt_length=10,         # Length of the generated prompt.
                 key_dim=768, 
                 ortho_mu=0.1):           # Orthogonality loss weight.
        
        super().__init__()
                
        self.emb_dim = emb_dim
        self.n_classes = n_classes
        self.pool_size_per_class = pool_size_per_class
        self.total_pool_size = n_classes * pool_size_per_class
        self.prompt_length = prompt_length
        self.key_dim = key_dim
        self.ortho_mu = ortho_mu    

        # Initialize prompt parameters (P, K, A).
        # Shape: [3 * pool_size, length, dim]
        self.e_p = tensor_prompt(self.total_pool_size, self.prompt_length, emb_dim)
        # Shape: [3 * pool_size, key_dim]
        self.e_k = tensor_prompt(self.total_pool_size, self.key_dim)
        # Shape: [3 * pool_size, key_dim]
        self.e_a = tensor_prompt(self.total_pool_size, self.key_dim)

        self._init_orthogonal()

    def _init_orthogonal(self):
        with torch.no_grad():
            self.e_p.data = self.gram_schmidt(self.e_p.data)
            self.e_k.data = self.gram_schmidt(self.e_k.data)
            self.e_a.data = self.gram_schmidt(self.e_a.data)

    def gram_schmidt(self, vv):
        """Apply simplified Gram-Schmidt orthogonalization during initialization."""
        is_3d = len(vv.shape) == 3
        if is_3d:
            original_shape = vv.shape
            vv = vv.view(vv.shape[0], -1)
        
        # Transpose for orthogonalization, preserving the original feature/sample convention.
        vv = vv.T 
        uu = torch.zeros_like(vv)
        
        for k in range(vv.size(1)):
            uk = vv[:, k].clone()
            for j in range(k):
                uj = uu[:, j]
                proj = (uk * uj).sum() / (uj * uj).sum().clamp(min=1e-8) * uj
                uk = uk - proj
            uu[:, k] = uk / uk.norm().clamp(min=1e-8)
            
        uu = uu.T
        if is_3d:
            uu = uu.view(original_shape)
        return uu
    

    def forward(self, x_query, class_labels=None, train=False):
        """
        Args:
            x_query: Decoder hidden states with shape [Batch, Dim] or [Batch, Seq, Dim].
            class_labels: Integer class IDs (0, 1, 2), with shape [Batch].
                If None, use attention over all prompt components.
        """
        # 1. Normalize input dimensions.
        # Handle [Batch, Dim] queries by adding a sequence dimension.
        if x_query.dim() == 2:
            x_query = x_query.unsqueeze(1) # [B, 1, D]
        
        B, _, C = x_query.shape

        # Retrieve prompt parameters.
        K = self.e_k
        A = self.e_a
        p = self.e_p
    
        # 2. Compute attention weights.
        # (B, 1, D) x (Pool, D).T -> (B, Pool); A has shape (Pool, key_dim).
        # Project queries to key_dim, or assume they already have that dimension.
        # Reference form: a_query = x @ A.T; element-wise weighting differs.
        # Reference form: torch.einsum('bd,kd->bkd', x_query, A) applies element-wise weights.
        
        # A generic decoder may require query projection.
        # Preserve the original computation here.
        # (B, 1, key_dim) * (1, Pool, key_dim) -> (B, Pool, key_dim)
        # Assume the query dimension equals key_dim; otherwise a linear projection is needed.
        
        # [B, 1, D] * [Pool, D] broadcasts to [B, Pool, D].
        a_query = torch.einsum('bld,pd->bpd', x_query, A) 
        
        # Cosine Similarity
        n_K = F.normalize(K, dim=1) # [Pool, D]
        q = F.normalize(a_query, dim=2) # [B, Pool, D]
        
        # [B, Pool, D] * [Pool, D] -> [B, Pool]: per-sample prompt-component weights.
        aq_k = torch.einsum('bpd,pd->bp', q, n_K)    

        # 3. Apply class-specific prompt routing.
        # When class_labels are provided, initialize disallowed prompt entries to negative infinity.
        if class_labels is not None:
            # Create a mask with shape [B, total_pool_size].
            mask = torch.zeros((B, self.total_pool_size), device=x_query.device)
            mask.fill_(-float('inf'))
            
            for b_idx in range(B):
                c_idx = class_labels[b_idx].item()
                start = c_idx * self.pool_size_per_class
                end = (c_idx + 1) * self.pool_size_per_class
                mask[b_idx, start:end] = 0.0
            
            # Apply the routing mask to attention scores.
            # The original implementation uses direct weighting, without softmax.
            # For direct weighting, set weights outside the selected class to zero.
            mask_binary = (mask == 0.0).float()
            aq_k = aq_k * mask_binary
        else:
            # Without labels, optional softmax normalization could select components during inference.
            # Keep the original direct-weighting behavior here.
            pass

        # 4. Generate the prompt.
        # [B, Pool] * [Pool, Len, D] -> [B, Len, D]
        # aq_k contains weights; p contains prompt components.
        P_ = torch.einsum('bp,pld->bld', aq_k, p)

        # 5. Split keys and values for prefix tuning.
        # Assume an even prompt_length and split it into keys and values.
        split_idx = int(self.prompt_length / 2)
        Ek = P_[:, :split_idx, :]
        Ev = P_[:, split_idx:, :]

        # 6. Compute the orthogonality loss during training.
        loss = torch.tensor(0.0, device=x_query.device)
        if train and self.ortho_mu > 0:
            loss += ortho_penalty(K) * self.ortho_mu
            loss += ortho_penalty(A) * self.ortho_mu
            loss += ortho_penalty(p.view(p.shape[0], -1)) * self.ortho_mu

        return [Ek, Ev], loss  




class UnifiedPrompt(nn.Module):
    def __init__(self, 
                 emb_dim, 
                 pool_size=30,      
                 prompt_length=10,
                 key_dim=768, 
                 ortho_mu=0.1): 
        
        super().__init__()
                
        self.emb_dim = emb_dim
        self.pool_size = pool_size 
        self.prompt_length = prompt_length
        self.key_dim = key_dim
        self.ortho_mu = ortho_mu    

        # Initialize prompt parameters (P, K, A).
        # Shape: [pool_size, length, dim]
        self.e_p = tensor_prompt(self.pool_size, self.prompt_length, emb_dim)
        # Shape: [pool_size, key_dim]
        self.e_k = tensor_prompt(self.pool_size, self.key_dim)
        # Shape: [pool_size, key_dim]
        self.e_a = tensor_prompt(self.pool_size, self.key_dim)

        self._init_orthogonal()

    def _init_orthogonal(self):
        with torch.no_grad():
            self.e_p.data = self.gram_schmidt(self.e_p.data)
            self.e_k.data = self.gram_schmidt(self.e_k.data)
            self.e_a.data = self.gram_schmidt(self.e_a.data)

    def gram_schmidt(self, vv):
        """Apply simplified Gram-Schmidt orthogonalization during initialization."""
        is_3d = len(vv.shape) == 3
        if is_3d:
            original_shape = vv.shape
            vv = vv.view(vv.shape[0], -1)
        
        # Transpose for orthogonalization.
        vv = vv.T 
        uu = torch.zeros_like(vv)
        
        for k in range(vv.size(1)):
            uk = vv[:, k].clone()
            for j in range(k):
                uj = uu[:, j]
                proj = (uk * uj).sum() / (uj * uj).sum().clamp(min=1e-8) * uj
                uk = uk - proj
            uu[:, k] = uk / uk.norm().clamp(min=1e-8)
            
        uu = uu.T
        if is_3d:
            uu = uu.view(original_shape)
        return uu
    

    def forward(self, x_query, train=False):

        # 1. Normalize input dimensions.
        if x_query.dim() == 2:
            x_query = x_query.unsqueeze(1) # [B, 1, D]
        
        B, _, C = x_query.shape

        # Retrieve prompt parameters.
        K = self.e_k
        A = self.e_a
        p = self.e_p
    
        # 2. Compute attention weights.
        # (B, 1, D) x (Pool, D).T -> (B, Pool)
        
        # [B, 1, D] * [Pool, D] -> [B, Pool, D]
        a_query = torch.einsum('bld,pd->bpd', x_query, A) 
        
        # Cosine Similarity
        n_K = F.normalize(K, dim=1) # [Pool, D]
        q = F.normalize(a_query, dim=2) # [B, Pool, D]
        
        # [B, Pool, D] * [Pool, D] -> [B, Pool]
        # aq_k contains similarity scores between the input and each pool component.
        aq_k = torch.einsum('bpd,pd->bp', q, n_K)     

        # No routing mask is applied.
        # Select pool components softly using similarity scores (aq_k).
        # No class_labels are required to restrict the selection.

        # 3. Generate the prompt.
        # [B, Pool] * [Pool, Len, D] -> [B, Len, D]
        P_ = torch.einsum('bp,pld->bld', aq_k, p)

        # 4. Split keys and values.
        split_idx = int(self.prompt_length / 2)
        Ek = P_[:, :split_idx, :]
        Ev = P_[:, split_idx:, :]

        # 5. Compute the orthogonality loss.
        loss = torch.tensor(0.0, device=x_query.device)
        if train and self.ortho_mu > 0:
            loss += ortho_penalty(K) * self.ortho_mu
            loss += ortho_penalty(A) * self.ortho_mu
            loss += ortho_penalty(p.view(p.shape[0], -1)) * self.ortho_mu

        return [Ek, Ev], loss