import torch
from torch import nn

from attention import MultiHeadAttention



class TransformerLayer(nn.Module):
    def __init__(self, d_model, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.mlp_norm = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, int(d_model * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(d_model * mlp_ratio), d_model),
            nn.Dropout(dropout)
        )


    def forward(self, x, causal_mask=None):
        # Shape of x : (B, N, d_model)
        residue = x
        x = self.attn_norm(x)
        # Shape of x : (B, N, d_model)
        x = self.attn(x, causal_mask=causal_mask)
        x = x + residue

        residue = x
        # Shape of x : (B, N, d_model)
        x = self.mlp_norm(x)
        x = self.mlp(x)
        x = x + residue
        return x

