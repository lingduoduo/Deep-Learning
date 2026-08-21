import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class TransformerBlock(nn.Module):

    def __init__(self, d_model):
        super().__init__()
        attention = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
        self.attention = attention(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.norm1(x + self.attention(x))
        x = self.norm2(x + self.ffn(x))
        return x

class GPT2BlockTorch(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )

    def forward(self, x):
        B, S, D = x.shape
        # causal mask:
        # True means "do not attend"
        causal_mask = torch.triu(
            torch.ones(S, S, device=x.device, dtype=torch.bool),
            diagonal=1
        )
        # Pre-LN
        h = self.ln1(x)
        attn_out, _ = self.attention(
            h, h, h,
            attn_mask=causal_mask,
            need_weights=False
        )
        # residual
        x = x + attn_out
        # Pre-LN + MLP + residual
        x = x + self.mlp(self.ln2(x))
        return x
    
block = GPT2BlockTorch(
    d_model=64,
    num_heads=4
)

x = torch.randn(2, 8, 64)
out = block(x)
print(out.shape)
# torch.Size([2, 8, 64])