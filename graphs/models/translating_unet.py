import torch.nn as nn
import torch
from graphs.models.custom_layers.down_block import Down
from graphs.models.custom_layers.up_block import Up
from graphs.models.custom_layers.double_conv import DoubleConv

class PosEnc(nn.Module):
    def __init__(self,inc,outc):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(inc, 300), nn.ReLU(inplace=True), nn.Linear(300, outc),
                                nn.ReLU(inplace=True))
    def forward(self,x):
        return self.fc(x)

class TranslatingUnetOSOten(nn.Module):
    def __init__(self, n_channels,n_classes,image_size,bias=False):
        number_of_feature_map = 32
        down_mode = "maxpool"
        up_mode = "bilinear"
        self.max_feature_map=8 * number_of_feature_map
        self.min_size=image_size//2//5//2//2
        self.number_of_feature_map=number_of_feature_map
        super().__init__()

        self.inc = DoubleConv(n_channels, number_of_feature_map,bias=bias)
        self.down1 = Down(number_of_feature_map, 2 * number_of_feature_map,mode=down_mode,bias=bias)
        self.down2 = Down(2 * number_of_feature_map, 4 * number_of_feature_map, factor=5,mode=down_mode,bias=bias)
        self.down3 = Down(4 * number_of_feature_map, self.max_feature_map,mode=down_mode,bias=bias)
        self.down4 = Down( self.max_feature_map,self.max_feature_map,mode=down_mode,bias=bias)
        self.up1 = Up(16 * number_of_feature_map, 4 * number_of_feature_map, up_mode,bias=bias)
        self.up2 = Up(self.max_feature_map, 2 * number_of_feature_map, up_mode,bias=bias)
        self.outc = nn.Conv2d(2*number_of_feature_map, n_classes, kernel_size=1)
        self.sea_conv=DoubleConv(1, n_classes,bias=bias)


        self.pos_enc=PosEnc(128, self.max_feature_map)

    def forward(self, x,coord=None):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        if coord is not None:
            coord = self.pos_enc(coord).unsqueeze(2).unsqueeze(3)
            x5= x5 + coord

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        logits = self.outc(x)
        return logits

class TranslatingUnetfive(nn.Module):
    def __init__(self, n_channels,n_classes,image_size,bias=False):
        number_of_feature_map = 32
        down_mode = "maxpool"
        up_mode="bilinear"
        self.max_feature_map=8 * number_of_feature_map
        self.min_size=int(image_size//2//2.5//2//2)
        self.number_of_feature_map=number_of_feature_map
        super().__init__()

        self.inc = DoubleConv(n_channels, number_of_feature_map,bias=bias)
        self.down1 = Down(number_of_feature_map, 2 * number_of_feature_map,mode=down_mode,bias=bias)
        self.down2 = Down(2 * number_of_feature_map, 4 * number_of_feature_map, mode='adaptativemaxpool',size=(image_size//5,image_size//5))
        self.down3 = Down(4 * number_of_feature_map, self.max_feature_map,mode=down_mode,bias=bias)
        self.down4 = Down( self.max_feature_map,self.max_feature_map,mode=down_mode,bias=bias)
        self.up1 = Up(16 * number_of_feature_map, 4 * number_of_feature_map, up_mode,bias=bias,size=(image_size//2,image_size//2))
        self.up2 = Up(self.max_feature_map, 2 * number_of_feature_map, up_mode,bias=bias)
        self.outc = nn.Conv2d(2*number_of_feature_map, n_classes, kernel_size=1)


        self.pos_enc=PosEnc(128, self.max_feature_map)

    def forward(self, x,coord=None):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        if coord is not None:
            coord = self.pos_enc(coord).unsqueeze(2).unsqueeze(3)
            x5= x5 + coord

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        logits = self.outc(x)
        return logits

class TranslatingUnettwo(nn.Module):
    def __init__(self, n_channels,n_classes,image_size,bias=False):
        number_of_feature_map = 32
        down_mode = "maxpool"
        up_mode="bilinear"
        self.max_feature_map=8 * number_of_feature_map
        self.min_size=image_size//2//2//2//2
        self.number_of_feature_map=number_of_feature_map
        super().__init__()

        self.inc = DoubleConv(n_channels, number_of_feature_map,bias=bias)
        self.down1 = Down(number_of_feature_map, 2 * number_of_feature_map,mode=down_mode,bias=bias)
        self.down2 = Down(2 * number_of_feature_map, 4 * number_of_feature_map,mode=down_mode,bias=bias)
        self.down3 = Down(4 * number_of_feature_map, self.max_feature_map,mode=down_mode,bias=bias)
        self.down4 = Down( self.max_feature_map,self.max_feature_map,mode=down_mode,bias=bias)
        self.up1 = Up(16 * number_of_feature_map, 4 * number_of_feature_map, up_mode,bias=bias)
        self.up2 = Up(self.max_feature_map, 2 * number_of_feature_map, up_mode,bias=bias)
        self.up3 = Up(4*number_of_feature_map, 2 * number_of_feature_map, up_mode,bias=bias)
        self.outc = nn.Conv2d(2*number_of_feature_map, n_classes, kernel_size=1)
        self.sea_conv=DoubleConv(1, n_classes,bias=bias)


        self.pos_enc = PosEnc(128, self.max_feature_map)


    def forward(self, x,coord=None):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        if coord is not None:
            coord = self.pos_enc(coord).unsqueeze(2).unsqueeze(3)
            x5 = x5 + coord

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        logits = self.outc(x)
        return logits

class TranslatingUnetsame(nn.Module):
    def __init__(self, n_channels,n_classes,image_size,bias=False):
        number_of_feature_map = 32
        down_mode = "maxpool"
        up_mode="bilinear"
        self.max_feature_map=8 * number_of_feature_map
        self.min_size=image_size//2//2//2//2
        self.number_of_feature_map=number_of_feature_map
        super().__init__()

        self.inc = DoubleConv(n_channels, number_of_feature_map,bias=bias)
        self.down1 = Down(number_of_feature_map, 2 * number_of_feature_map,mode=down_mode,bias=bias)
        self.down2 = Down(2 * number_of_feature_map, 4 * number_of_feature_map,mode=down_mode,bias=bias)
        self.down3 = Down(4 * number_of_feature_map, self.max_feature_map,mode=down_mode,bias=bias)
        self.down4 = Down( self.max_feature_map,self.max_feature_map,mode=down_mode,bias=bias)
        self.up1 = Up(16 * number_of_feature_map, 4 * number_of_feature_map, up_mode,bias=bias)
        self.up2 = Up(self.max_feature_map, 2 * number_of_feature_map, up_mode,bias=bias)
        self.up3 = Up(4*number_of_feature_map, number_of_feature_map, up_mode,bias=bias)
        self.up4 = Up(2 * number_of_feature_map, 2 * number_of_feature_map, up_mode,bias=bias)
        self.outc = nn.Conv2d(2*number_of_feature_map, n_classes, kernel_size=1)
        self.sea_conv=DoubleConv(1, n_classes,bias=bias)

        self.pos_enc = PosEnc(128, self.max_feature_map)

    def forward(self, x,coord=None):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        if coord is not None:
            coord = self.pos_enc(coord).unsqueeze(2).unsqueeze(3)
            x5 = x5 + coord

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
class Upsample(nn.Module):
    def __init__(self,  scale_factor=2,mode="nearest"):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode=mode
    def forward(self, x):
        return torch.nn.functional.interpolate(x, scale_factor=self.scale_factor,mode=self.mode)

class TranslatingUnetReduce(nn.Module):
    def __init__(self, n_channels,n_classes,image_size,target_size,bias=False):

        number_of_feature_map = 32
        down_mode = "maxpool"
        up_mode="bilinear"
        self.max_feature_map=8 * number_of_feature_map
        self.min_size=target_size//2//2//2//2
        self.number_of_feature_map=number_of_feature_map
        super().__init__()
        self.resize = Upsample(scale_factor=target_size // image_size, mode='nearest')
        self.inc = DoubleConv(n_channels, number_of_feature_map,bias=bias)
        self.down1 = Down(number_of_feature_map, 2 * number_of_feature_map,mode=down_mode,bias=bias)
        self.down2 = Down(2 * number_of_feature_map, 4 * number_of_feature_map,mode=down_mode,bias=bias)
        self.down3 = Down(4 * number_of_feature_map, self.max_feature_map,mode=down_mode,bias=bias)
        self.down4 = Down( self.max_feature_map,self.max_feature_map,mode=down_mode,bias=bias)
        self.up1 = Up(16 * number_of_feature_map, 4 * number_of_feature_map, up_mode,bias=bias)
        self.up2 = Up(self.max_feature_map, 2 * number_of_feature_map, up_mode,bias=bias)
        self.up3 = Up(4*number_of_feature_map, number_of_feature_map, up_mode,bias=bias)
        self.up4 = Up(2 * number_of_feature_map, 2 * number_of_feature_map, up_mode,bias=bias)
        self.outc = nn.Conv2d(2*number_of_feature_map, n_classes, kernel_size=1)
        self.sea_conv=DoubleConv(1, n_classes,bias=bias)

        self.pos_enc = PosEnc(128, self.max_feature_map)

    def forward(self, x,coord=None):
        x=self.resize(x)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        if coord is not None:
            print("snif")
            coord = self.pos_enc(coord).unsqueeze(2).unsqueeze(3)
            x5 = x5 + coord

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

class TranslatingUnetfour(nn.Module):
    def __init__(self, n_channels,n_classes,image_size,bias=False):
        number_of_feature_map = 32
        down_mode = "maxpool"
        up_mode="bilinear"
        self.max_feature_map=8 * number_of_feature_map
        self.min_size=image_size//2//2//2//2
        self.number_of_feature_map=number_of_feature_map
        super().__init__()

        self.inc = DoubleConv(n_channels, number_of_feature_map,bias=bias)
        self.down1 = Down(number_of_feature_map, 2 * number_of_feature_map,mode=down_mode,bias=bias)
        self.down2 = Down(2 * number_of_feature_map, 4 * number_of_feature_map,mode=down_mode,bias=bias)
        self.down3 = Down(4 * number_of_feature_map, self.max_feature_map,mode=down_mode,bias=bias)
        self.down4 = Down( self.max_feature_map,self.max_feature_map,mode=down_mode,bias=bias)
        self.up1 = Up(16 * number_of_feature_map, 4 * number_of_feature_map, up_mode,bias=bias)
        self.up2 = Up(self.max_feature_map, 2 * number_of_feature_map, up_mode,bias=bias)
        self.outc = nn.Conv2d(2*number_of_feature_map, n_classes, kernel_size=1)
        self.sea_conv=DoubleConv(1, n_classes,bias=bias)

        self.pos_enc = PosEnc(128, self.max_feature_map)

    def forward(self, x,coord=None):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        if coord is not None:
            coord = self.pos_enc(coord).unsqueeze(2).unsqueeze(3)
            x5 = x5 + coord

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        logits = self.outc(x)
        return logits

class TranslatingUnettwenty(nn.Module):
    def __init__(self, n_channels,n_classes,image_size,bias=False):
        number_of_feature_map = 32
        down_mode = "maxpool"
        up_mode="bilinear"
        self.max_feature_map=8 * number_of_feature_map
        self.min_size=image_size//2//2//5//2//2
        self.number_of_feature_map=number_of_feature_map
        super().__init__()

        self.inc = DoubleConv(n_channels, number_of_feature_map,bias=bias)
        self.down0 = Down(number_of_feature_map, 2 * number_of_feature_map,mode=down_mode,bias=bias)
        self.down1 = Down(2*number_of_feature_map, 2 * number_of_feature_map, mode=down_mode,bias=bias)
        self.down2 = Down(2 * number_of_feature_map, 4 * number_of_feature_map,mode=down_mode,bias=bias,factor=5)
        self.down3 = Down(4 * number_of_feature_map, self.max_feature_map,mode=down_mode,bias=bias)
        self.down4 = Down( self.max_feature_map,self.max_feature_map,mode=down_mode,bias=bias)
        self.up1 = Up(16 * number_of_feature_map, 4 * number_of_feature_map, up_mode,bias=bias)
        self.up2 = Up(self.max_feature_map, 2 * number_of_feature_map, up_mode,bias=bias)
        self.outc = nn.Conv2d(2*number_of_feature_map, n_classes, kernel_size=1)
        self.sea_conv=DoubleConv(1, n_classes,bias=bias)

        self.pos_enc = PosEnc(128, self.max_feature_map)

    def forward(self, x,coord=None):
        x1 = self.inc(x)
        x1b = self.down0(x1)
        x2 = self.down1(x1b)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        if coord is not None:
            coord = self.pos_enc(coord).unsqueeze(2).unsqueeze(3)
            x5 = x5 + coord

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        logits = self.outc(x)
        return logits