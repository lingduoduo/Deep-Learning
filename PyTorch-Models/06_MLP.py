import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLUMLP(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff)
        self.up_proj = nn.Linear(d_model, d_ff)
        self.down_proj = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


mlp = SwiGLUMLP(d_model=64, d_ff=128)
x = torch.randn(2, 8, 64)
print('Output:', mlp(x).shape)
print('Params:', sum(p.numel() for p in mlp.parameters()))
