# Original code from https://github.com/milesial/Pytorch-UNet
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, bias=False):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels  # In Down mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=bias),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, bias=False):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels, bias=bias)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, bias=False):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(
                in_channels, out_channels, in_channels // 2, bias=bias
            )
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels, bias=bias)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(
        self,
        in_frames: int = 3,
        n_classes: int = 1,
        bilinear: bool = True,
        dropout_p: float = 0,
        output_activation: Optional[str] = "sigmoid",
        filters: int = 64,
        bias: bool = False,
    ):
        super().__init__()
        self.description = f"UNet_inFrames{in_frames}_outFrames{n_classes}_out_activation{output_activation}"
        self.n_channels = in_frames
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.dropout_p = dropout_p

        if self.dropout_p > 0:
            self.mc_unet = True
        else:
            self.mc_unet = False

        factor = 2 if bilinear else 1

        self.inc = DoubleConv(in_frames, filters)
        self.down1 = Down(filters, 2 * filters, bias=bias)
        self.down2 = Down(2 * filters, 4 * filters, bias=bias)
        self.down3 = Down(4 * filters, 8 * filters, bias=bias)
        self.down4 = Down(8 * filters, 16 * filters // factor, bias=bias)

        self.up1 = Up(16 * filters, 8 * filters // factor, bilinear, bias=bias)
        self.up2 = Up(8 * filters, 4 * filters // factor, bilinear, bias=bias)
        self.up3 = Up(4 * filters, 2 * filters // factor, bilinear, bias=bias)
        self.up4 = Up(2 * filters, filters, bilinear, bias=bias)
        self.outc = OutConv(filters, n_classes)

        if output_activation is None:
            output_activation = "none"
        output_activation = output_activation.lower()
        if output_activation is None or output_activation in ["none", "", "identity"]:
            self.out_activation = nn.Identity()
        elif output_activation in ["sigmoid", "sigmoide", "sig"]:
            self.out_activation = nn.Sigmoid()
        elif output_activation in ["relu"]:
            self.out_activation = nn.Hardtanh(
                min_val=0, max_val=1.0
            )  # works as relu clip between [0,1]
        elif output_activation in ["tanh"]:
            self.out_activation = nn.Tanh()
        elif output_activation in ["softmax"]:
            self.out_activation = nn.Softmax(dim=1)
        elif output_activation in ["logsoftmax"]:
            self.out_activation = nn.LogSoftmax(dim=1)
        elif output_activation in ["softplus"]:
            self.out_activation = nn.Softplus()
        else:
            raise ValueError(f"Activation function {output_activation} not recognized")

    def forward(self, x):
        x1 = self.inc(x)
        # convolution (64 filters 3x3 , padd=1 )=> [BN] => ReLU) and convolution (64 filters 3x3, pad=1 )=> [BN] => ReLU)
        x2 = F.dropout(self.down1(x1), p=self.dropout_p, training=self.mc_unet)
        # maxpool (2x2) => convolution (128 filters 3x3 , padd=1 )=> [BN] => ReLU) and convolution (128 filters 3x3, pad=1 )=> [BN] => ReLU)
        x3 = F.dropout(self.down2(x2), p=self.dropout_p, training=self.mc_unet)
        # maxpool (2x2) => convolution (256 filters 3x3 , padd=1 )=> [BN] => ReLU) and convolution (256 filters 3x3, pad=1 )=> [BN] => ReLU)
        x4 = F.dropout(self.down3(x3), p=self.dropout_p, training=self.mc_unet)
        # maxpool (2x2) => convolution (512 filters 3x3 , padd=1 )=> [BN] => ReLU) and convolution (512 filters 3x3, pad=1 )=> [BN] => ReLU)
        x5 = F.dropout(self.down4(x4), p=self.dropout_p, training=self.mc_unet)
        # maxpool (2x2) => convolution (512 o 1024 filters 3x3 , padd=1 )=> [BN] => ReLU) and convolution (512 o 1024 filters 3x3, pad=1 )=> [BN] => ReLU)
        x = F.dropout(self.up1(x5, x4), p=self.dropout_p, training=self.mc_unet)
        x = F.dropout(self.up2(x, x3), p=self.dropout_p, training=self.mc_unet)
        x = F.dropout(self.up3(x, x2), p=self.dropout_p, training=self.mc_unet)
        x = F.dropout(self.up4(x, x1), p=self.dropout_p, training=self.mc_unet)
        out = self.outc(x)
        out = self.out_activation(out)
        return out
