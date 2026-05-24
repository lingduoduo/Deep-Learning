
import numpy as np
import torch
import torch.nn as nn
import math

# Sinusoidal Positional Encoding
def sinusoidal_position_embedding_numpy(seq_len, d_model, theta=10000.0):
    pe = np.zeros((seq_len, d_model))

    position = np.arange(seq_len)[:, None]          # (seq_len, 1)
    div_term = np.exp(
        np.arange(0, d_model, 2) * (-np.log(theta) / d_model)
    )                                               # (d_model // 2,)

    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term[:pe[:, 1::2].shape[1]])
    return pe

x = np.random.randn(2, 5, 8)  # batch, seq_len, d_model
pe = sinusoidal_position_embedding_numpy(seq_len=5, d_model=8)
x_with_pos = x + pe[None, :, :]
print(x_with_pos.shape)

def precompute_freqs_cis_numpy(dim, max_seq_len, theta=10000.0):
    """
    dim: head_dim, must be even
    returns:
        cos: (max_seq_len, dim // 2)
        sin: (max_seq_len, dim // 2)
    """
    i = np.arange(0, dim, 2)
    freqs = 1.0 / (theta ** (i / dim))

    t = np.arange(max_seq_len)
    angles = np.outer(t, freqs)

    cos = np.cos(angles)
    sin = np.sin(angles)
    return cos, sin

def apply_rotary_emb_numpy(x, cos, sin):
    """
    x: (..., seq_len, dim)
       examples:
       (batch, seq_len, dim)
       (batch, heads, seq_len, head_dim)

    cos/sin: (seq_len, dim // 2)
    """
    seq_len = x.shape[-2]
    dim = x.shape[-1]

    cos = cos[:seq_len]
    sin = sin[:seq_len]

    # make cos/sin broadcastable to x
    while cos.ndim < x.ndim - 1:
        cos = cos[None, ...]
        sin = sin[None, ...]

    x_pair = x.reshape(*x.shape[:-1], dim // 2, 2)

    x_even = x_pair[..., 0]
    x_odd = x_pair[..., 1]

    rotated_even = x_even * cos - x_odd * sin
    rotated_odd = x_even * sin + x_odd * cos
    rotated = np.stack([rotated_even, rotated_odd], axis=-1)
    return rotated.reshape(*x.shape)

x = np.random.randn(2, 4, 5, 8)  # batch, heads, seq_len, head_dim
cos, sin = precompute_freqs_cis_numpy(dim=8, max_seq_len=100)
x_rope = apply_rotary_emb_numpy(x, cos, sin)
print(x_rope.shape)


# RoPE (Rotary Positional Embedding)
def apply_rope(q, k):
    batch, seq_len, hidden = q.shape
    assert hidden % 2 == 0

    pos = torch.arange(seq_len, device=q.device).float().unsqueeze(1)
    dim = torch.arange(0, hidden, 2, device=q.device).float()

    freqs = 1.0 / (10000.0 ** (dim / hidden))
    angles = pos * freqs

    cos_a = torch.cos(angles).unsqueeze(0)
    sin_a = torch.sin(angles).unsqueeze(0)

    def rotate(x):
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]

        rotated = torch.stack(
            [
                x1 * cos_a - x2 * sin_a,
                x1 * sin_a + x2 * cos_a,
            ],
            dim=-1,
        )

        return rotated.flatten(-2)

    return rotate(q), rotate(k)

q = torch.randn(1, 8, 16)
k = torch.randn(1, 8, 16)
qr, kr = apply_rope(q, k)
print('Shape preserved:', qr.shape == q.shape)
print('Norm preserved:', torch.allclose(q.norm(dim=-1), qr.norm(dim=-1), atol=1e-4))


class PositionalEncodingTorch(nn.Module):
    def __init__(self, d_model, max_len=5000, theta=10000.0):
        super().__init__()

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(theta) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)

        # handle odd d_model safely
        pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].shape[1]])

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        self.register_buffer("pe", pe)

    def forward(self, x):
        seq_len = x.size(1)  # x: (batch, seq_len, d_model)
        return x + self.pe[:, :seq_len, :]

x = torch.randn(2, 5, 8)
pos_enc = PositionalEncodingTorch(d_model=8, max_len=100)
out = pos_enc(x)
print(out.shape)

