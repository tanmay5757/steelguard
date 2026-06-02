import torch
import torch.nn as nn
import torchvision.models as models

class DoubleConv(nn.Module):
    """Two conv layers each followed by BatchNorm + ReLU."""
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2)

        self.d1  = DoubleConv(3,   64)
        self.d2  = DoubleConv(64,  128)
        self.d3  = DoubleConv(128, 256)
        self.mid = DoubleConv(256, 512)

        self.u3 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.c3 = DoubleConv(512, 256)

        self.u2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.c2 = DoubleConv(256, 128)

        self.u1 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.c1 = DoubleConv(128, 64)

        self.out = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        d1 = self.d1(x)
        d2 = self.d2(self.pool(d1))
        d3 = self.d3(self.pool(d2))
        m  = self.mid(self.pool(d3))

        u3 = self.c3(torch.cat([self.u3(m),  d3], 1))
        u2 = self.c2(torch.cat([self.u2(u3), d2], 1))
        u1 = self.c1(torch.cat([self.u1(u2), d1], 1))

        return self.out(u1)


class ResNetUNet(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)

        self.layer0 = nn.Sequential(base.conv1, base.bn1, base.relu)
        self.layer1 = nn.Sequential(base.maxpool, base.layer1)
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4

        self.up4   = nn.ConvTranspose2d(512, 256, 2, 2)
        self.conv4 = DoubleConv(512, 256)

        self.up3   = nn.ConvTranspose2d(256, 128, 2, 2)
        self.conv3 = DoubleConv(256, 128)

        self.up2   = nn.ConvTranspose2d(128, 64, 2, 2)
        self.conv2 = DoubleConv(128, 64)

        self.up1   = nn.ConvTranspose2d(64, 64, 2, 2)
        self.conv1 = DoubleConv(128, 64)

        self.out = nn.Conv2d(64, 1, 1)

        self.final_upsample = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True
        )

    def forward(self, x):
        l0 = self.layer0(x)
        l1 = self.layer1(l0)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)

        u4 = self.conv4(torch.cat([self.up4(l4), l3], 1))
        u3 = self.conv3(torch.cat([self.up3(u4), l2], 1))
        u2 = self.conv2(torch.cat([self.up2(u3), l1], 1))
        u1 = self.conv1(torch.cat([self.up1(u2), l0], 1))

        return self.final_upsample(self.out(u1))
__all__ = [
    "DoubleConv",
    "UNet",
    "ResNetUNet"
]