import torch

import torch.nn as nn

from modules.pyramidpooling import TemporalPyramidPooling

from timm.models.registry import register_model

__all__ = [
    'PHOSCnet_character_counter'
]


class CharacterCounterNet(nn.Module):
    def __init__(self, outputs:int=17):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, (3, 3), padding='same'),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3), padding='same'),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=2),
            nn.Conv2d(64, 128, (3, 3), padding='same'),
            nn.ReLU(),
            nn.Conv2d(128, 128, (3, 3), padding='same'),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=2),
            nn.Conv2d(128, 256, (3, 3), padding='same'),
            nn.ReLU(),
            nn.Conv2d(256, 256, (3, 3), padding='same'),
            nn.ReLU(),
            nn.Conv2d(256, 256, (3, 3), padding='same'),
            nn.ReLU(),
            nn.Conv2d(256, 256, (3, 3), padding='same'),
            nn.ReLU(),
            nn.Conv2d(256, 256, (3, 3), padding='same'),
            nn.ReLU(),
            nn.Conv2d(256, 256, (3, 3), padding='same'),
            nn.ReLU(),
            nn.Conv2d(256, 512, (3, 3), padding='same'),
            nn.ReLU(),
            nn.Conv2d(512, 512, (3, 3), padding='same'),
            nn.ReLU(),
            nn.Conv2d(512, 512, (3, 3), padding='same'),
            nn.ReLU(),
        )

        self.temporal_pool = TemporalPyramidPooling([1, 2, 5])

        self.head = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),

            nn.Linear(4096, outputs),
            nn.Softmax(1)
        )

        

    def forward(self, x: torch.Tensor) -> dict:
        x = self.conv(x)
        x = self.temporal_pool(x)

        return self.head(x)


@register_model()
def PHOSCnet_character_counter(outputs=17, **kwargs):
    return CharacterCounterNet(outputs)

if __name__ == '__main__':
    model = CharacterCounterNet(17)

    x = torch.randn(5, 50, 250, 3).view(-1, 3, 50, 250)

    y = model(x)

    print(y)
    print(y.shape)