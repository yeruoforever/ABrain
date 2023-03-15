from typing import List, Union, cast, Dict

import torch
import torch.nn as nn


def vgg_backbone(
    spatial_dim: int,
    cfg: List[Union[str, int]],
    batch_norm: bool = False,
    in_channels=3
) -> nn.Sequential:
    layers: List[nn.Module] = []
    is_2d = spatial_dim == 2
    Conv = nn.Conv2d if is_2d else nn.Conv3d
    Pool = nn.MaxPool2d if is_2d else nn.MaxPool3d
    Norm = nn.BatchNorm2d if is_2d else nn.BatchNorm3d
    for v in cfg:
        if v == "M":
            layers += [Pool(kernel_size=2, stride=2)]
        elif v == 'LRN':
            layers += [nn.LocalResponseNorm(size=3)]
        else:
            v = cast(int, v)
            if v < 0:
                conv = Conv(in_channels, -v, kernel_size=1, padding=0)
            else:
                conv = Conv(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv, Norm(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class VGG(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        cfg: List[Union[str, int]],
        n_classes: int,
        in_channels: int,
        dropout: float = 0.5,
        batch_norm: bool = False,
    ) -> None:
        super().__init__()
        self.feature = vgg_backbone(
            spatial_dim=spatial_dims,
            cfg=cfg,
            batch_norm=batch_norm,
            in_channels=in_channels
        )
        is_2d = spatial_dims == 2
        AdpAvgPool = nn.AdaptiveAvgPool2d if is_2d else nn.AdaptiveAvgPool3d
        self.avgpool = AdpAvgPool(7),
        out_features = 512*7*7
        if is_2d:
            out_features *= 7
        self.classifier = nn.Sequential(
            nn.Linear(out_features, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, n_classes),
        )

    def forward(self, data) -> torch.Tensor:
        x = self.feature(data)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x


VGG_Type: Dict[str, List[Union[int, str]]] = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'A-LRN': [64, 'LRN', 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'C': [64, 64, 'M', 128, 128, 'M', 256, 256, -256, 'M', 512, 512, -512, 'M', 512, 512, -512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}
