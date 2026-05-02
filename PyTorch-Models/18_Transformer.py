import torch
import torch.nn as nn
import math
import numpy as np

# Self-Attention 与 Multi-Head Attention (MHA)

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, mask=None):
        """
        q: (batch, heads, q_len, d_k)
        k: (batch, heads, k_len, d_k)
        v: (batch, heads, k_len, d_v)
        mask: broadcastable to (batch, heads, q_len, k_len)
              True/1 means masked position
        """
        d_k = q.size(-1)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k) 

        if mask is not None:
            mask = mask.bool()
            scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)

        attn_weights = torch.softmax(scores, dim=-1)

        output = torch.matmul(attn_weights, v)
        return output, attn_weights

 
class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.q_linear = nn.Linear(hidden_dim, hidden_dim)
        self.k_linear = nn.Linear(hidden_dim, hidden_dim)
        self.v_linear = nn.Linear(hidden_dim, hidden_dim)

        self.out_linear = nn.Linear(hidden_dim, hidden_dim)
 
        self.attention = ScaledDotProductAttention()
    
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape

        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
 
        attn_out, _ = self.attention(q, k, v, mask=mask)
        attn_out = attn_out.transpose(1, 2)
        attn_out = attn_out.contiguous().view(batch_size, seq_len, self.hidden_dim)
        output = self.out_linear(attn_out)
        
        return output

    
x = torch.randn(2, 5, 64)
print(f"Input Shape：{x.shape}")
mask = torch.triu(torch.ones(5, 5), diagonal=1).unsqueeze(0).unsqueeze(0)
mha = MultiHeadAttention(hidden_dim=64, num_heads=8)
out = mha(x, mask=mask)
print(f"Output Shape：{out.shape}")

