"""Small GroupNorm U-Net shared by both precipitation baselines."""

import torch
from torch import nn


def _block(in_channels: int, out_channels: int) -> nn.Sequential:
    groups = 8 if out_channels >= 8 else 1
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
        nn.GroupNorm(groups, out_channels),
        nn.SiLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
        nn.GroupNorm(groups, out_channels),
        nn.SiLU(inplace=True),
    )


class SmallUNet(nn.Module):
    def __init__(self, in_channels: int = 2, base_channels=(16, 32, 64, 128)):
        super().__init__()
        c1, c2, c3, c4 = map(int, base_channels)
        self.enc1 = _block(in_channels, c1)
        self.enc2 = _block(c1, c2)
        self.enc3 = _block(c2, c3)
        self.enc4 = _block(c3, c4)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = _block(c4, c4 * 2)
        self.up4 = nn.ConvTranspose2d(c4 * 2, c4, 2, stride=2)
        self.dec4 = _block(c4 * 2, c4)
        self.up3 = nn.ConvTranspose2d(c4, c3, 2, stride=2)
        self.dec3 = _block(c3 * 2, c3)
        self.up2 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.dec2 = _block(c2 * 2, c2)
        self.up1 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.dec1 = _block(c1 * 2, c1)
        self.head = nn.Conv2d(c1, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        z = self.bottleneck(self.pool(e4))
        z = self.dec4(torch.cat((self.up4(z), e4), dim=1))
        z = self.dec3(torch.cat((self.up3(z), e3), dim=1))
        z = self.dec2(torch.cat((self.up2(z), e2), dim=1))
        z = self.dec1(torch.cat((self.up1(z), e1), dim=1))
        return self.head(z)
