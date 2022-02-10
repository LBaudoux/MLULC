import torch
import torch.nn as nn
from graphs.models.custom_layers.double_conv import DoubleConv


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels,mode='maxpool',factor=2,num_groups=None,size=None,bias=False):
        super().__init__()

        if mode=='maxpool':
            self.downsize = nn.Sequential(
                nn.MaxPool2d(factor),
                DoubleConv(in_channels, out_channels,num_groups=num_groups,bias=bias)
            )
        elif mode =='avgpool':
            self.downsize = nn.Sequential(
                nn.AvgPool2d(factor),
                DoubleConv(in_channels, out_channels,num_groups=num_groups,bias=bias)
            )
        elif mode == 'strideconv':
            self.downsize = nn.Sequential(
                DoubleConv(in_channels, out_channels,stride=factor,num_groups=num_groups,bias=bias)
            )
        elif mode == 'adaptativemaxpool':
            self.downsize = nn.Sequential(
                nn.AdaptiveMaxPool2d(size),
                DoubleConv(in_channels, out_channels, num_groups=num_groups,bias=bias)
            )
        else :
            raise ValueError("Unknown downsizing mode : "+str(mode))

    def forward(self, x):
        return self.downsize(x)