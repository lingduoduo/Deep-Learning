import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# Scaled Dot Product Attention
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x_q, x_k, x_v):
        # Q: (B, S_q, d_k)
        # K: (B, S_k, d_k)
        # V: (B, S_k, d_v)

        d_k = x_k.size(-1)
        scores = torch.matmul(x_q, x_k.transpose(-2, -1)) / math.sqrt(d_k)
        weights = torch.softmax(scores, dim=-1)
        return torch.matmul(weights, x_v)

attn = ScaledDotProductAttention()
x_q = torch.randn(1, 3, 16)
x_k = torch.randn(1, 5, 16)
x_v = torch.randn(1, 5, 32)
out = attn(x_q, x_k, x_v)
print("Output shape:", out.shape)
# print("Output:", out)


# Multi-Head Attention - only for cross-attention, where K and V come from the same source.
class MultiHeadAttention(nn.Module):
    def __init__(self, num_inputs, num_heads):
        super().__init__()

        assert num_inputs % num_heads == 0

        self.num_heads = num_heads
        self.d_k = num_inputs // num_heads

        self.W_q = nn.Linear(num_inputs, num_inputs)
        self.W_k = nn.Linear(num_inputs, num_inputs)
        self.W_v = nn.Linear(num_inputs, num_inputs)
        self.W_o = nn.Linear(num_inputs, num_inputs)

    def forward(self, x_q, x_k, x_v):
        B, S_q, _ = x_q.shape
        S_k = x_k.shape[1]
        S_v = x_v.shape[1]
        assert S_k == S_v

        q = self.W_q(x_q).view(
            B, S_q, self.num_heads, self.d_k
        ).transpose(1, 2)
        k = self.W_k(x_k).view(
            B, S_k, self.num_heads, self.d_k
        ).transpose(1, 2)
        v = self.W_v(x_v).view(
            B, S_v, self.num_heads, self.d_k
        ).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(
            q, k.transpose(-2, -1)
        ) / math.sqrt(self.d_k)

        weights = torch.softmax(scores, dim=-1)
        attn = torch.matmul(weights, v)
        # Merge heads
        out = (
            attn
            .transpose(1, 2)
            .contiguous()
            .view(B, S_q, -1)
        )
        return self.W_o(out)
    
attn = MultiHeadAttention(64, 4) # cross-attention 
x_q = torch.randn(2, 6, 64)
x_kv = torch.randn(2, 10, 64)
out = attn(x_q, x_kv, x_kv)
print(out.shape)


# Self-Attention
class SelfAttention(nn.Module):
    def __init__(self, num_inputs, num_heads):
        super().__init__()

        assert num_inputs % num_heads == 0

        self.num_heads = num_heads
        self.d_k = num_inputs // num_heads

        self.W_q = nn.Linear(num_inputs, num_inputs)
        self.W_k = nn.Linear(num_inputs, num_inputs)
        self.W_v = nn.Linear(num_inputs, num_inputs)
        self.W_o = nn.Linear(num_inputs, num_inputs)

    def forward(self, x):
        B, S, _ = x.shape

        # Q, K, V all come from the same x
        q = self.W_q(x).view(
            B, S, self.num_heads, self.d_k
        ).transpose(1, 2)
        k = self.W_k(x).view(
            B, S, self.num_heads, self.d_k
        ).transpose(1, 2)
        v = self.W_v(x).view(
            B, S, self.num_heads, self.d_k
        ).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(
            q, k.transpose(-2, -1)
        ) / math.sqrt(self.d_k)

        weights = torch.softmax(scores, dim=-1)
        attn = torch.matmul(weights, v)
        # Merge heads
        out = (
            attn
            .transpose(1, 2)
            .contiguous()
            .view(B, S, -1)
        )
        return self.W_o(out)

x = torch.randn(1, 4, 8)
self_attention = SelfAttention(
    num_inputs=8,
    num_heads=2
)
out = self_attention(x)
print(out.shape)
# torch.Size([1, 4, 8])


# Multi-Head Causal Self-Attention with an upper-triangular mask.
class CausalSelfAttention(nn.Module):
    def __init__(self, num_inputs, num_heads):
        super().__init__()

        assert num_inputs % num_heads == 0

        self.num_heads = num_heads
        self.d_k = num_inputs // num_heads

        self.W_q = nn.Linear(num_inputs, num_inputs)
        self.W_k = nn.Linear(num_inputs, num_inputs)
        self.W_v = nn.Linear(num_inputs, num_inputs)
        self.W_o = nn.Linear(num_inputs, num_inputs)

    def forward(self, x):
        B, S, _ = x.shape

        # Self-attention: Q, K, V all come from x
        q = self.W_q(x).view(
            B, S, self.num_heads, self.d_k
        ).transpose(1, 2)
        k = self.W_k(x).view(
            B, S, self.num_heads, self.d_k
        ).transpose(1, 2)
        v = self.W_v(x).view(
            B, S, self.num_heads, self.d_k
        ).transpose(1, 2)
        # q, k, v: (B, num_heads, S, d_k)

        # Scaled dot-product attention
        scores = torch.matmul(
            q, k.transpose(-2, -1)
        ) / math.sqrt(self.d_k)
        # scores: (B, num_heads, S, S)

        # Causal mask: prevent token i from attending to future tokens j > i
        mask = torch.triu(
            torch.ones(
                S, S,
                device=x.device,
                dtype=torch.bool
            ),
            diagonal=1
        )
        scores = scores.masked_fill(
            mask.unsqueeze(0).unsqueeze(0),
            float("-inf")
        )
        # Attention weights
        weights = torch.softmax(scores, dim=-1)
        # Weighted sum of values
        attn = torch.matmul(weights, v)
        # attn:(B, num_heads, S, d_k)
        # Merge heads
        out = (
            attn
            .transpose(1, 2)
            .contiguous()
            .view(B, S, -1)
        )
        return self.W_o(out)


x = torch.randn(1, 4, 8)
causal_attention = CausalSelfAttention(
    num_inputs=8,
    num_heads=2
)
out = causal_attention(x)
print("Output shape:", out.shape)
# torch.Size([1, 4, 8])

# GroupQueryAttention
class GroupQueryAttention(nn.Module):
    def __init__(self, num_inputs: int, num_heads: int, num_kv_heads: int):
        super().__init__()

        assert num_inputs % num_heads == 0
        assert num_heads % num_kv_heads == 0

        self.num_inputs = num_inputs
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.d_k = num_inputs // num_heads

        self.W_q = nn.Linear(num_inputs, num_heads * self.d_k)
        self.W_k = nn.Linear(num_inputs, num_kv_heads * self.d_k)
        self.W_v = nn.Linear(num_inputs, num_kv_heads * self.d_k)
        self.W_o = nn.Linear(num_inputs, num_inputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = x.shape

        q = self.W_q(x).view(
            batch_size, seq_len, self.num_heads, self.d_k
        ).transpose(1, 2)
        k = self.W_k(x).view(
            batch_size, seq_len, self.num_kv_heads, self.d_k
        ).transpose(1, 2)
        v = self.W_v(x).view(
            batch_size, seq_len, self.num_kv_heads, self.d_k
        ).transpose(1, 2)

        repeats = self.num_heads // self.num_kv_heads
        k = k.repeat_interleave(repeats, dim=1)
        v = v.repeat_interleave(repeats, dim=1)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
            diagonal=1
        )
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        attn = torch.matmul(weights, v)
        out = attn.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.num_inputs
        )
        return self.W_o(out)
    
torch.manual_seed(0)
gqa = GroupQueryAttention(num_inputs=32, num_heads=8, num_kv_heads=2)
print("W_q shape:", gqa.W_q.weight.shape)  # (32, 32)
print("W_k shape:", gqa.W_k.weight.shape)  # (8, 32)  — only 2 KV heads * d_k=4

x = torch.randn(2, 6, 32)
out = gqa.forward(x)
print("Output shape:", out.shape)           # (2, 6, 32)


# SlidingWindowAttention
class SlidingWindowAttention(nn.Module):

    def __init__(self, num_inputs: int, num_heads: int, window_size: int):
        super().__init__()

        assert num_inputs % num_heads == 0

        self.num_inputs = num_inputs
        self.num_heads = num_heads
        self.window_size = window_size
        self.d_k = num_inputs // num_heads

        self.W_q = nn.Linear(num_inputs, num_inputs)
        self.W_k = nn.Linear(num_inputs, num_inputs)
        self.W_v = nn.Linear(num_inputs, num_inputs)
        self.W_o = nn.Linear(num_inputs, num_inputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        q = self.W_q(x).view(
            batch_size, seq_len, self.num_heads, self.d_k
        ).transpose(1, 2)

        k = self.W_k(x).view(
            batch_size, seq_len, self.num_heads, self.d_k
        ).transpose(1, 2)

        v = self.W_v(x).view(
            batch_size, seq_len, self.num_heads, self.d_k
        ).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        # scores: (B, H, S, S)

        idx = torch.arange(seq_len, device=x.device)
        # allow position i to attend to [i-window_size, i+window_size]
        mask = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs() > self.window_size
        # mask: (S, S)
        scores = scores.masked_fill(
            mask.unsqueeze(0).unsqueeze(0),
            float("-inf")
        )
        weights = torch.softmax(scores, dim=-1)
        attn = torch.matmul(weights, v)
        out = attn.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.num_inputs
        )
        return self.W_o(out)
    
x = torch.randn(2, 6, 32)
attn = SlidingWindowAttention(
    num_inputs=32,
    num_heads=8,
    window_size=1
)
out = attn(x)
print("Output shape:", out.shape)  # [2, 6, 32]


## KV-Cached Attention
# KV cache = reuse across tokens/decoding steps.
class KVCacheAttention(nn.Module):
    def __init__(self, num_inputs, num_heads):
        super().__init__()

        assert num_inputs % num_heads == 0

        self.num_inputs = num_inputs
        self.num_heads = num_heads
        self.d_k = num_inputs// num_heads

        self.W_q = nn.Linear(num_inputs, num_inputs)
        self.W_k = nn.Linear(num_inputs, num_inputs)
        self.W_v = nn.Linear(num_inputs, num_inputs)
        self.W_o = nn.Linear(num_inputs, num_inputs)

    def forward(self, x, cache=None):
        """
        x:     (B, S_new, D)
        cache: optional tuple(k_cache, v_cache)

        k_cache/v_cache:
            (B, H, S_past, d_k)

        return:
            out:       (B, S_new, D)
            new_cache: (K_all, V_all)
        """

        batch_size, seq_len, _ = x.shape

        q = self.W_q(x).view(
            batch_size, seq_len, self.num_heads, self.d_k
        ).transpose(1, 2)

        k = self.W_k(x).view(
            batch_size, seq_len, self.num_heads, self.d_k
        ).transpose(1, 2)

        v = self.W_v(x).view(
            batch_size, seq_len, self.num_heads, self.d_k
        ).transpose(1, 2)

        if cache is not None:
            k_cache, v_cache = cache
            k_all = torch.cat([k_cache, k], dim=2)
            v_all = torch.cat([v_cache, v], dim=2)
        else:
            k_all = k
            v_all = v

        new_cache = (k_all, v_all)

        s_total = k_all.shape[2]
        s_past = s_total - seq_len

        scores = torch.matmul(q, k_all.transpose(-2, -1)) / math.sqrt(self.d_k)
        # scores: (B, H, S_new, S_total)

        # causal mask
        query_pos = torch.arange(
            s_past,
            s_total,
            device=x.device
        ).unsqueeze(1)

        key_pos = torch.arange(
            s_total,
            device=x.device
        ).unsqueeze(0)

        mask = key_pos > query_pos
        # mask: (S_new, S_total)

        scores = scores.masked_fill(
            mask.unsqueeze(0).unsqueeze(0),
            float("-inf")
        )

        weights = torch.softmax(scores, dim=-1)
        attn = torch.matmul(weights, v_all)
        # attn: (B, H, S_new, d_k)
        out = attn.transpose(1, 2).contiguous().view(
            batch_size,
            seq_len,
            self.num_inputs
        )
        out = self.W_o(out)
        return out, new_cache

# Demo: full forward vs incremental decode
torch.manual_seed(0)
attn = KVCacheAttention(num_inputs=64, num_heads=4)
x = torch.randn(1, 6, 64)

full_out, _ = attn(x)
out1, cache = attn(x[:, :4])
out2, cache = attn(x[:, 4:5], cache=cache)
out3, cache = attn(x[:, 5:6], cache=cache)
inc_out = torch.cat([out1, out2, out3], dim=1)

print('Full shape:', full_out.shape)
print('Match:', torch.allclose(full_out, inc_out, atol=1e-5))
print('Final cache K shape:', cache[0].shape)


# Flash Attention
class FlashAttention(nn.Module):
    def __init__(self, num_inputs, num_heads):
        super().__init__()

        assert num_inputs % num_heads == 0

        self.num_inputs = num_inputs
        self.num_heads = num_heads
        self.d_k = num_inputs // num_heads

    def forward(self, Q, K, V, block_size):
        batch_size, seq_len, hidden_dim = Q.shape

        output = torch.zeros_like(Q)

        for i in range(0, seq_len, block_size):
            qi = Q[:, i:i + block_size]
            bs_q = qi.shape[1]

            row_max = torch.full(
                (batch_size, bs_q, 1),
                float("-inf"),
                device=Q.device
            )
            row_sum = torch.zeros(
                batch_size, bs_q, 1,
                device=Q.device
            )
            acc = torch.zeros(
                batch_size, bs_q, hidden_dim,
                device=Q.device
            )
            for j in range(0, seq_len, block_size):
                kj = K[:, j:j + block_size]
                vj = V[:, j:j + block_size]
                scores = torch.bmm(qi, kj.transpose(1, 2)) / math.sqrt(hidden_dim)
                block_max = scores.max(dim=-1, keepdim=True).values
                new_max = torch.maximum(row_max, block_max)
                correction = torch.exp(row_max - new_max)
                exp_scores = torch.exp(scores - new_max)
                acc = acc * correction + torch.bmm(exp_scores, vj)
                row_sum = row_sum * correction + exp_scores.sum(dim=-1, keepdim=True)
                row_max = new_max
            output[:, i:i + block_size] = acc / row_sum
        return output

torch.manual_seed(0)
Q = torch.randn(1, 4, 8)
K = torch.randn(1, 4, 8)
V = torch.randn(1, 4, 8)
flash_attention = FlashAttention(num_inputs=8, num_heads=1)
out = flash_attention(Q, K, V, block_size=2)
print(out.shape)
scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(8)
ref = torch.bmm(torch.softmax(scores, dim=-1), V)
print("Match:", torch.allclose(out, ref, atol=1e-4))

