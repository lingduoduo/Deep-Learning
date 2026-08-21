import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        kH, kW = kernel_size
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kH, kW))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        # x: (B, C_in, H, W)
        if self.padding > 0:
            x = F.pad(x, [self.padding] * 4) #  padding for left, right, top, bottom

        B, C_in, H, W = x.shape
        C_out, _, kH, kW = self.weight.shape

        H_out = (H - kH) // self.stride + 1
        W_out = (W - kW) // self.stride + 1

        patches = x.unfold(2, kH, self.stride).unfold(3, kW, self.stride)
        # patches: (B, C_in, H_out, W_out, kH, kW)
        out = torch.einsum("bihwjk,oijk->bohw", patches, self.weight)
        out = out + self.bias.view(1, -1, 1, 1)
        return out

x = torch.randn(1, 3, 8, 8)
conv = Conv2d(
    in_channels=3,
    out_channels=16,
    kernel_size=3,
    stride=1,
    padding=0
)
y = conv(x)
print('Output:', y.shape)
print(
    "Match:",
    torch.allclose(
        conv(x),
        F.conv2d(
            x,
            conv.weight,
            conv.bias,
            stride=conv.stride,
            padding=conv.padding
        ),
        atol=1e-4
    )
)
