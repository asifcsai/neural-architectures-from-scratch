import torch
from torch import nn

from attention import MultiHeadAttention



class TransformerLayer(nn.Module):
    def __init__(self, d_model, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.attn_norm = nn.LayerNorm(d_model, eps=1e-6, elementwise_affine=True)
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.mlp_norm = nn.LayerNorm(d_model, eps=1e-6, elementwise_affine=True)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, int(d_model * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(d_model * mlp_ratio), d_model),
            nn.Dropout(dropout)
        )

        self.adaptive_layer = nn.Sequential(
            nn.GELU(),
            nn.Linear(d_model, d_model * 6)
        )   


        # DiT Initialization

        torch.nn.init.xavier_uniform_(self.mlp[0].weight)
        torch.nn.init.constant_(self.mlp[0].bias, 0)
        torch.nn.init.xavier_uniform_(self.mlp[-2].weight)
        torch.nn.init.constant_(self.mlp[-2].bias, 0)

        torch.nn.init.constant_(self.adaptive_layer[-1].weight, 0)
        torch.nn.init.constant_(self.adaptive_layer[-1].bias, 0)




    def forward(self, x,time_emb, causal_mask=None):
        # Shape of x : (B, N, d_model)
        # Shape of time_emb : (B, 1, d_model)
        residue = x

        # Shape of adaptive_params : (B, 1, 6*d_model)
        adaptive_params = self.adaptive_layer(time_emb)
        pre_attn_shift, pre_attn_scale, post_attn_scale, pre_mlp_shift, pre_mlp_scale, post_mlp_scale = adaptive_params.chunk(6, dim=-1)

        x = self.attn_norm(x)*(pre_attn_scale + 1) + pre_attn_shift
        # Shape of x : (B, N, d_model)
        x = self.attn(x, causal_mask=causal_mask)
        x = x*post_attn_scale + residue

        residue = x
        # Shape of x : (B, N, d_model)
        x = self.mlp_norm(x)*(pre_mlp_scale + 1) + pre_mlp_shift
        x = self.mlp(x)
        x = x*post_mlp_scale + residue
        return x

