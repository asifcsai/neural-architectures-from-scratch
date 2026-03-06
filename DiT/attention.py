import torch
from torch import nn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        assert d_model % num_heads == 0, 'd_model must be divisible by num_heads'
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.qkv_layer = nn.Linear(d_model, d_model * 3, bias=in_proj_bias)
        self.output_layer = nn.Linear(d_model, d_model, bias=out_proj_bias)
    

    def forward(self, x, causal_mask=None):
        B, N, C = x.shape
        qkv = self.qkv_layer(x)  # (B, N, 3 * d_model)
        qkv = qkv.view(B, N, 3, self.num_heads, self.d_k) # (B, N, 3, num_heads, d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4) # (3, B, num_heads, N, d_k)
        q, k, v = qkv[0], qkv[1], qkv[2] # (B, num_heads, N, d_k) 
        # q, k, v = qkv.unbind(0) # (B, num_heads, N, d_k)

        # Shape :  (B, num_heads, N, d_k) x (B, num_heads, d_k, N) -> (B, num_heads, N, N)
        # N*N matrix first row for the first query token, second row for the second query token and so on. 
        # Each column represents the attention score of the corresponding token with all other tokens.
        attn_scores = torch.matmul(q, k.transpose(-2, -1))

        # Shape : (B, num_heads, N, N)
        attn_scores = attn_scores / (self.d_k ** 0.5) 
        if causal_mask is not None:
            mask = torch.ones_like(attn_scores, device=attn_scores.device, dtype=torch.bool).triu(diagonal=1)
            attn_scores = attn_scores.masked_fill(mask, -torch.inf)
    
        attn_weights = torch.softmax(attn_scores, dim=-1)

        # Shape : (B, num_heads, N, N) x (B, num_heads, N, d_k) -> (B, num_heads, N, d_k)
        attn_output = torch.matmul(attn_weights, v)
        # attn_output = attn_output.transpose(1, 2).contiguous().view(B, N, C) # (B, N, d_model)
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(B, N, C) # (B, N, d_model)

        attn_output = self.output_layer(attn_output)
        return attn_output




