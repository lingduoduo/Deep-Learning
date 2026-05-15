import torch
import torch.nn as nn
import math

# 层归一化 (Layer Normalization)
# 公式: y = gamma * (x - mean) / sqrt(var + eps) + beta

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

x = torch.randn(2, 8)
layer = LayerNorm(8)
out = layer(x)
ref = torch.nn.functional.layer_norm(x, [8], layer.gamma, layer.beta)
print("Match ref?", torch.allclose(out, ref, atol=1e-4))


# Batch Norm
class BatchNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5, momentum=0.1):
        super().__init__()

        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))

        self.register_buffer("running_mean", torch.zeros(hidden_size))
        self.register_buffer("running_var", torch.ones(hidden_size))

        self.eps = eps
        self.momentum = momentum
    
    def forward(self, x):
        if self.training:
            batch_mean = x.mean(dim=0)
            batch_var = x.var(dim=0, unbiased=False)

            self.running_mean.mul_(1 - self.momentum).add_(self.momentum * batch_mean.detach())
            self.running_var.mul_(1 - self.momentum).add_(self.momentum * batch_var.detach())

            mean = batch_mean
            var = batch_var
        else:
            mean = self.running_mean
            var = self.running_var

        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta

x = torch.randn(8, 4)
layer = BatchNorm(4)
out_train = layer(x)
print("[Train] Output shape:", out_train.shape)
print("[Train] Column means:", out_train.mean(dim=0))
print("[Train] Column stds: ", out_train.std(dim=0, unbiased=False))
print("Updated running_mean:", layer.running_mean)
print("Updated running_var:", layer.running_var)

layer.eval()
out_eval = layer(x)
print("[Eval] Output shape:", out_eval.shape)


# RMS 归一化 (Root Mean Square Layer Normalization)
# 公式: y = gamma * x / RMS(x), 其中 RMS(x) = sqrt(mean(x^2) + eps)
class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x.pow(2), dim=-1, keepdim=True) + self.eps)
        x_norm = x / rms
        return self.gamma * x_norm

x = torch.randn(8, 4)
layer = RMSNorm(4)
out = layer(x)
print('RMS of output:', out.pow(2).mean(dim=-1).sqrt())