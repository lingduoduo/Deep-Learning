import torch
import torch.nn as nn
 

# 层归一化 (Layer Normalization)
# 公式: y = gamma * (x - mean) / sqrt(var + eps) + beta

# RMS 归一化 (Root Mean Square Layer Normalization)
# 公式: y = gamma * x / RMS(x), 其中 RMS(x) = sqrt(mean(x^2) + eps)

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps
    
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var +self.eps)
        return self.gamma * x_norm + self.beta

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x, dim=-1, keepdim=True) + self.eps)
        x_norm = x / rms
        return self.gamma * x_norm


