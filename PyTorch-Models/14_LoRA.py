import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor


class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank, alpha=1.0):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

        # Freeze original linear layer
        self.linear.weight.requires_grad_(False)
        if self.linear.bias is not None:
            self.linear.bias.requires_grad_(False)

        # LoRA low-rank matrices
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        self.scaling = alpha / rank

    def forward(self, x: Tensor) -> Tensor:
        lora_out = x @ self.lora_A.T @ self.lora_B.T
        return self.linear(x) + lora_out * self.scaling


layer = LoRALinear(16, 8, rank=4)
x = torch.randn(2, 16)
print("Output:", layer(x).shape)
print("Trainable:", sum(p.numel() for p in layer.parameters() if p.requires_grad))
print("Total:    ", sum(p.numel() for p in layer.parameters()))
