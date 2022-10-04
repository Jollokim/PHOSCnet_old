import torch

import torch.nn as nn

from modules.pyramidpooling import TemporalPyramidPooling

from timm.models.registry import register_model

__all__ = [
    'PHOSCnet_residual',
]

class ResidualPHOSCnet(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 64
            ResBlock(64, 64, downsample=False),
            ResBlock(64, 64, downsample=False),
            ResBlock(64, 64, downsample=False),
            # 128
            ResBlock(64, 128, downsample=True),
            ResBlock(128, 128, downsample=False),
            ResBlock(128, 128, downsample=False),
            ResBlock(128, 128, downsample=False),
            # 256
            ResBlock(128, 256, downsample=True),
            ResBlock(256, 256, downsample=False),
            ResBlock(256, 256, downsample=False),
            ResBlock(256, 256, downsample=False),
            ResBlock(256, 256, downsample=False),
            ResBlock(256, 256, downsample=False),
            # 512
            ResBlock(256, 512, downsample=True),
            ResBlock(512, 512, downsample=False),
            ResBlock(512, 512, downsample=False),
        )


        self.temporal_pool = TemporalPyramidPooling([1, 2, 5])

        self.phos = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),

            nn.Linear(4096, 165),
            nn.ReLU()
        )

        self.phoc = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),

            nn.Linear(4096, 604),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> dict:
        x = self.conv(x)
        x = self.temporal_pool(x)

        return {'phos': self.phos(x), 'phoc': self.phoc(x)}


# resnet34 residual block implementation from: https://towardsdev.com/implement-resnet-with-pytorch-a9fb40a77448
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample):
        super().__init__()
        if downsample:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.shortcut = nn.Sequential()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, input):
        shortcut = self.shortcut(input)
        input = nn.ReLU()(self.bn1(self.conv1(input)))
        input = nn.ReLU()(self.bn2(self.conv2(input)))
        input = input + shortcut
        return nn.ReLU()(input)

@register_model
def PHOSCnet_residual(**kwargs):
    return ResidualPHOSCnet()