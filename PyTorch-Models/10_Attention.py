import torch
import torch.nn as nn
import math


# # Multi-Head Attention - only for cross-attention, where K and V come from the same source.
# class MultiHeadCrossAttention(nn.Module):
#     def __init__(self, d_model, num_heads):
#         super().__init__()

#         assert d_model % num_heads == 0

#         self.num_heads = num_heads
#         self.d_k = d_model // num_heads

#         self.W_q = nn.Linear(d_model, d_model)
#         self.W_k = nn.Linear(d_model, d_model)
#         self.W_v = nn.Linear(d_model, d_model)
#         self.W_o = nn.Linear(d_model, d_model)

#     def forward(self, x_q, x_kv):
#         B, S_q, _ = x_q.shape
#         S_kv = x_kv.shape[1]

#         q = self.W_q(x_q).view(B, S_q, self.num_heads, self.d_k).transpose(1, 2)
#         k = self.W_k(x_kv).view(B, S_kv, self.num_heads, self.d_k).transpose(1, 2)
#         v = self.W_v(x_kv).view(B, S_kv, self.num_heads, self.d_k).transpose(1, 2)

#         scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
#         weights = torch.softmax(scores, dim=-1)

#         attn = torch.matmul(weights, v)

#         out = attn.transpose(1, 2).contiguous().view(B, S_q, -1)

#         return self.W_o(out)

# cross_attn = MultiHeadCrossAttention(64, 4)
# x_q = torch.randn(2, 6, 64)
# x_kv = torch.randn(2, 10, 64)
# out = cross_attn(x_q, x_kv)
# print(out.shape)


# # Softmax Attention
# class ScaledDotProductAttention(nn.Module):
#     def __init__(self):
#         super().__init__()
    
#     def forward(self, x_q, x_k, x_v):
#         # Q: (B, S_q, d_k)
#         # K: (B, S_k, d_k)
#         # V: (B, S_k, d_v)

#         d_k = x_k.size(-1)
#         scores = torch.matmul(x_q, x_k.transpose(-2, -1)) / math.sqrt(d_k)
#         weights = torch.softmax(scores, dim=-1)
#         return torch.matmul(weights, x_v)

# attn = ScaledDotProductAttention()
# x_q = torch.randn(1, 3, 16)
# x_k = torch.randn(1, 5, 16)
# x_v = torch.randn(1, 5, 32)

# out = attn(x_q, x_k, x_v)
# print("Output shape:", out.shape)
# # print("Output:", out)


# # # Multi-Head Attention - Use this for both self-attention and cross-attention.
# class MultiHeadAttention(nn.Module):
#     def __init__(self, d_model, num_heads):
#         super().__init__()

#         assert d_model % num_heads == 0

#         self.num_heads = num_heads
#         self.d_k = d_model // num_heads

#         self.W_q = nn.Linear(d_model, d_model)
#         self.W_k = nn.Linear(d_model, d_model)
#         self.W_v = nn.Linear(d_model, d_model)
#         self.W_o = nn.Linear(d_model, d_model)

#     def forward(self, x_q, x_k, x_v):
#         B, S_q, _ = x_q.shape
#         S_k = x_k.shape[1]

#         q = self.W_q(x_q).view(B, S_q, self.num_heads, self.d_k).transpose(1, 2)
#         k = self.W_k(x_k).view(B, S_k, self.num_heads, self.d_k).transpose(1, 2)
#         v = self.W_v(x_v).view(B, S_k, self.num_heads, self.d_k).transpose(1, 2)

#         scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
#         weights = torch.softmax(scores, dim=-1)
#         attn = torch.matmul(weights, v)
#         out = attn.transpose(1, 2).contiguous().view(B, S_q, -1)
#         return self.W_o(out)

# attn = MultiHeadAttention(64, 4)

# # self-attention
# x = torch.randn(2, 8, 64)
# out1 = attn(x, x, x)
# print(out1.shape)
# # cross-attention
# x_q = torch.randn(2, 6, 64)
# x_kv = torch.randn(2, 10, 64)
# out2 = attn(x_q, x_kv, x_kv)
# print(out2.shape)


# Causal Self-Attention
# class CausalAttention(nn.Module):
#     def __init__(self):
#         super().__init__()
    
#     def forward(self, Q, K, V):
#         batch_size, seq_len, hidden_dim = Q.shape

#         scores = Q @ K.transpose(-2, -1) / math.sqrt(hidden_dim)
#         mask = torch.triu(torch.ones(seq_len, seq_len, device=scores.device, dtype=torch.bool), diagonal=1)
#         scores = scores.masked_fill(mask.unsqueeze(0), float("-inf"))
#         weights = torch.softmax(scores, dim=-1)
#         output = weights @ V
#         return output

# torch.manual_seed(0)
# Q = torch.randn(1, 4, 8)
# K = torch.randn(1, 4, 8)
# V = torch.randn(1, 4, 8)
# causal_attention = CausalAttention()
# out = causal_attention(Q, K, V)
# print(out.shape)
# print("Pos 0 == V[0]?", torch.allclose(out[:, 0], V[:, 0], atol=1e-5))




# # Multi-Head Attention (MHA) with Mask
# class ScaledDotProductAttention(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, q, k, v, mask=None):
#         """
#         q: (batch, heads, q_len, d_k)
#         k: (batch, heads, k_len, d_k)
#         v: (batch, heads, k_len, d_v)
#         mask: broadcastable to (batch, heads, q_len, k_len)
#               True/1 means masked position
#         """
#         d_k = q.size(-1)

#         scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k) 

#         if mask is not None:
#             mask = mask.bool()
#             scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)

#         attn_weights = torch.softmax(scores, dim=-1)

#         output = torch.matmul(attn_weights, v)
#         return output, attn_weights

 
# class MultiHeadAttention(nn.Module):
#     def __init__(self, hidden_dim, num_heads):
#         super().__init__()

#         self.hidden_dim = hidden_dim
#         self.num_heads = num_heads
#         self.head_dim = hidden_dim // num_heads

#         self.q_linear = nn.Linear(hidden_dim, hidden_dim)
#         self.k_linear = nn.Linear(hidden_dim, hidden_dim)
#         self.v_linear = nn.Linear(hidden_dim, hidden_dim)

#         self.out_linear = nn.Linear(hidden_dim, hidden_dim)
 
#         self.attention = ScaledDotProductAttention()
    
#     def forward(self, x, mask=None):
#         batch_size, seq_len, _ = x.shape

#         q = self.q_linear(x)
#         k = self.k_linear(x)
#         v = self.v_linear(x)

#         q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
#         k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
#         v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)

#         q = q.transpose(1, 2)
#         k = k.transpose(1, 2)
#         v = v.transpose(1, 2)
 
#         attn_out, _ = self.attention(q, k, v, mask=mask)
#         attn_out = attn_out.transpose(1, 2)
#         attn_out = attn_out.contiguous().view(batch_size, seq_len, self.hidden_dim)
#         output = self.out_linear(attn_out)
        
#         return output

    
# x = torch.randn(2, 5, 64)
# print(f"Input Shape：{x.shape}")
# mask = torch.triu(torch.ones(5, 5), diagonal=1).unsqueeze(0).unsqueeze(0)
# mha = MultiHeadAttention(hidden_dim=64, num_heads=8)
# out = mha(x, mask=mask)
# print(f"Output Shape：{out.shape}")
