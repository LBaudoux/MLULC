import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2 if num group=None Instance norm, if num group=1 Layer norm"""

    def __init__(self, in_channels, out_channels,stride=1,num_groups=None,bias=False):
        super().__init__()
        if num_groups is None:
            num_groups=out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1,stride=stride,bias=bias),
            nn.GroupNorm(num_groups,out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1,bias=bias),
            nn.GroupNorm(num_groups, out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)