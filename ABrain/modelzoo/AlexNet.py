import torch
import torch.nn as nn

from .base import ClassifyModel


class AlexEncoder(nn.Module):
    '''`AlexEncoder` is a part of `AlexNet`

    ### Args:
        - `in_channels:int=3` the number of input's channels.
    '''

    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        self.layer_1 = nn.Sequential(
            nn.Conv2d(in_channels, 96, kernel_size=11, stride=4, padding=1),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.layer_2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=1, groups=2),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.layer_3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.layer_4 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1, groups=2),
            nn.ReLU()
        )
        self.layer_5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1, groups=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

    def forward(self, x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_5(x)
        return x


class AlexNet(ClassifyModel):
    '''AlexNet

    `AlexNet` in paper "ImageNet Classification with Deep Convolutional Neural Networks",
    It consists of `AlexEncoder` and `classifier`.

    ### Args:
        - `n_class:int` the number of output's classes.
        - `in_channels` the number of input images' channels. 

    ### Example:
    ```sh
    >>> B,C,W,H = 16,3,224,224
    >>> N = 1000
    >>> input = torch.rand(B,C,W,H)
    >>> model = AlexNet(n_class=N,in_channels=C)
    >>> output = model(input)
    >>> print(output.shape)
    >>> torch.Size([16, 1000])
    ```
    '''

    def __init__(self, n_class: int, in_channels: int = 3, *args, **kwargs) -> None:
        '''
        ### Args:
        - `n_class:int` the number of output's classes.
        - `in_channels` the number of input images' channels. '''
        super().__init__(n_class)
        self.encoder = AlexEncoder(in_channels)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(7*7*256, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, n_class),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.encoder(x)
        y = self.classifier(x)
        return y
