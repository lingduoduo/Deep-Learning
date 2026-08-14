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
    