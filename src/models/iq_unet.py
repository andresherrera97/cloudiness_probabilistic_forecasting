# Original code from https://github.com/milesial/Pytorch-UNet
import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantileEmbedding(nn.Module):
    def __init__(self, num_quantiles, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.embedding = nn.Linear(1, embedding_dim)

    def forward(self, tau):
        x = tau.unsqueeze(-1)  # Add feature dimension
        x = self.embedding(x)
        x = F.relu(x)
        return x


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


class IQN_UNet(nn.Module):
    def __init__(
        self,
        in_frames: int = 3,
        n_classes: int = 1,
        bilinear: bool = True,
        dropout_p: float = 0,
        filters: int = 64,
        bias: bool = False,
        num_quantiles: int = 32,
        embedding_dim: int = 64,
    ):
        super().__init__()
        self.description = f"IQN_UNet_inFrames{in_frames}_outFrames{n_classes}"
        self.n_channels = in_frames
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.dropout_p = dropout_p

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

        self.quantile_embedding = QuantileEmbedding(num_quantiles, embedding_dim)

    def forward(self, x, tau):
        tau_embedding = self.quantile_embedding(tau)

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Integrate tau embedding
        x5 = x5 * tau_embedding.unsqueeze(-1).unsqueeze(-1)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        out = self.outc(x)
        return out

    def quantile_huber_loss(self, predicted, target, tau, kappa=1.0):
        diff = target - predicted
        loss = torch.where(
            torch.abs(diff) <= kappa,
            0.5 * diff.pow(2),
            kappa * (torch.abs(diff) - 0.5 * kappa),
        )
        weight = torch.abs(tau - (diff < 0).float())
        return (weight * loss).mean()
