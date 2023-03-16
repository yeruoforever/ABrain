from typing import Tuple, List, Sequence, Callable, Optional
from functools import partial


import torch
import torch.nn as nn

from monai.networks.blocks import Convolution


def inception_v1(spatial_dims: int):
    return InceptionBackboneV1(spatial_dims)


def inception_v2():
    pass


def inception_v3():
    pass


def inception_v4():
    pass


class Inception(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        ch1x1: int,
        ch3x3red: int,
        ch3x3: int,
        ch5x5red: int,
        ch5x5: int,
        proj: int
    ) -> None:
        super().__init__()
        is_2d = spatial_dims == 2
        Pool = nn.MaxPool2d if is_2d else nn.MaxPool2d
        Conv = partial(Convolution, norm="BATCH")
        self.branch1 = Convolution(spatial_dims, in_channels, ch1x1)
        self.branch2 = nn.Sequential(
            Conv(spatial_dims, in_channels,
                 ch3x3red, kernel_size=1, padding=0),
            Conv(spatial_dims, ch3x3red, ch3x3,
                 kernel_size=3, padding=1)
        )
        self.branch3 = nn.Sequential(
            Conv(spatial_dims, in_channels,
                 ch5x5red, kernel_size=1, padding=0),
            Conv(spatial_dims, ch5x5red, ch5x5,
                 kernel_size=3, padding=1)
        )
        self.branch4 = nn.Sequential(
            Pool(kernel_size=3, stride=1, padding=1, ceil_mode=True),
            Conv(spatial_dims, in_channels,
                 proj, kernel_size=1, padding=0)
        )

    def forward(self, data):
        x1 = self.branch1(data)
        x2 = self.branch2(data)
        x3 = self.branch3(data)
        x4 = self.branch4(data)
        return torch.cat((x1, x2, x3, x4), dim=1)



def factored_conv(spatial_dims:int,in_channels:int,out_channels:int,kernel_size:int,padding:int):
    k,p = kernel_size,padding
    if spatial_dims==2:
        return [
            Convolution(spatial_dims,in_channels,out_channels,kernel_size=(k,1,),padding=(p,0,)),
            Convolution(spatial_dims,out_channels,out_channels,kernel_size=(1,k,),padding=(0,p,)),
        ]
    else:
        return [
            Convolution(spatial_dims,in_channels,out_channels,kernel_size=(k,1,1),padding=(p,0,0)),
            Convolution(spatial_dims,out_channels,out_channels,kernel_size=(1,k,1),padding=(0,p,0)),
            Convolution(spatial_dims,out_channels,out_channels,kernel_size=(1,1,k),padding=(0,0,p)),
        ]

class Branch(nn.Sequential):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        kernal_size: int,
        padding: int,
        units: int = 1,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        k, p=kernal_size,padding
        layers = [Convolution(spatial_dims, in_channels,
                              mid_channels, kernel_size=1, padding=0)]
        for i in range(units-1):
            layers.append(factored_conv(
                spatial_dims,
                in_channels,
                mid_channels,
                k,
                p
            ))
        layers.append(factored_conv(
            spatial_dims,
            mid_channels,
            out_channels,
            k,
            p
        ))
        self.extend(layers)

class InceptionA(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self,x):
        return x

class InceptionAux(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        num_classes: int,
        # conv_block: Optional[Callable[..., nn.Module]] = None,
        dropout: float = 0.7,
    ) -> None:
        super().__init__()
        is_2d = spatial_dims == 2
        Conv = nn.Conv2d if is_2d else nn.Conv3d
        AdpAvgPool = nn.AdaptiveAvgPool2d if is_2d else nn.AdaptiveAvgPool3d
        self.adp_avg_pool = AdpAvgPool(4)
        self.conv = Conv(in_channels, 128, kernel_size=1)
        output_features = 2048
        if is_2d:
            output_features *= 4
        self.fc1 = nn.Linear(output_features, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # aux1: N x 512 x 14 x 14, aux2: N x 528 x 14 x 14
        x = self.adp_avg_pool(x)
        # aux1: N x 512 x 4 x 4, aux2: N x 528 x 4 x 4
        x = self.conv(x)
        # N x 128 x 4 x 4
        x = torch.flatten(x, 1)
        # N x 2048
        x = nn.functional.relu(self.fc1(x), inplace=True)
        # N x 1024
        x = self.dropout(x)
        # N x 1024
        x = self.fc2(x)
        # N x 1000 (num_classes)

        return x


class InceptionAuxV2V3(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        n_classes: int,
        # dropout: float,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        is_2d = spatial_dims == 2
        AvgPool = nn.AvgPool2d if is_2d else nn.AvgPool3d
        AdpAvgPool = nn.AdaptiveAvgPool2d if is_2d else nn.AdaptiveAvgPool3d

        self.avg_pool = AvgPool(kernel_size=5, stride=3)
        self.conv0 = Convolution(in_channels, 128, kernel_size=1)
        self.conv1 = Convolution(128, 768, kernel_size=5)
        self.adp_avg_pool = AdpAvgPool(1)
        # self.conv1.stddev = 0.01  # type: ignore[assignment]
        self.fc = nn.Linear(768, n_classes)
        # self.fc.stddev = 0.001  # type: ignore[assignment]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # N x 768 x 17 x 17
        x = self.avg_pool(x)
        # N x 768 x 5 x 5
        x = self.conv0(x)
        # N x 128 x 5 x 5
        x = self.conv1(x)
        # N x 768 x 1 x 1
        # Adaptive average pooling
        x = self.adp_avg_pool(x, (1, 1))
        # N x 768 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 768
        x = self.fc(x)
        # N x 1000
        return


class InceptionBackboneV1(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        is_2d = spatial_dims == 2
        Pool = nn.MaxPool2d if is_2d else nn.MaxPool3d
        Pool = partial(Pool, stride=2, ceil_mode=True)

        self.conv_1 = Convolution(
            spatial_dims, 3, 64, kernel_size=7, strides=2, padding=3)
        self.maxpool_1 = Pool(3)

        self.conv_2 = Convolution(spatial_dims, 64, 64, kernel_size=1)
        self.conv_3 = Convolution(
            spatial_dims, 64, 192, kernel_size=3, padding=1)
        self.maxpool_2 = Pool(3),

        self.incptn_3a = Inception(spatial_dims, 192, 64, 96, 128, 16, 32, 32)
        self.incptn_3b = Inception(
            spatial_dims, 256, 128, 128, 192, 32, 96, 64)
        self.maxpool_3 = Pool(3)

        self.incptn_4a = Inception(spatial_dims, 480, 192, 96, 208, 16, 48, 64)
        self.incptn_4b = Inception(
            spatial_dims, 512, 160, 112, 224, 24, 64, 64)
        self.incptn_4c = Inception(
            spatial_dims, 512, 128, 128, 256, 24, 64, 64)
        self.incptn_4d = Inception(
            spatial_dims, 512, 112, 144, 288, 32, 64, 64)
        self.incptn_4e = Inception(
            spatial_dims, 528, 256, 160, 320, 32, 128, 128)
        self.maxpool_4 = Pool(2)

        self.incptn_5a = Inception(
            spatial_dims, 832, 256, 160, 320, 32, 128, 128)
        self.incptn_5b = Inception(
            spatial_dims, 832, 384, 192, 384, 48, 128, 128)

    def forward(self, data):

        x = self.conv_1(data)
        x = self.maxpool_1(x)

        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.maxpool_2(x)

        x = self.incptn_3a(x)
        x = self.incptn_3b(x)
        x = self.maxpool_3(x)

        x = self.incptn_4a(x)
        res_1 = x
        x = self.incptn_4b(x)
        x = self.incptn_4c(x)
        x = self.incptn_4d(x)
        res_2 = x
        x = self.incptn_4e(x)
        x = self.maxpool_4(x)

        x = self.incptn_5a(x)
        x = self.incptn_5b(x)

        return x, res_2, res_1


class InceptionV1(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        n_classes: int,
        dropout: float = 0.2,
        dropout_aux: float = 0.7,
        deep_supervised: bool = False,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        is_2d = spatial_dims == 2
        AdpAvgPool = nn.AdaptiveAvgPool2d if is_2d else nn.AdaptiveAvgPool3d
        Dropout = nn.Dropout2d if is_2d else nn.Dropout3d
        self.backbone = InceptionBackboneV1(spatial_dims)
        if deep_supervised:
            self.aux1 = InceptionAux(spatial_dims, 512, n_classes, dropout_aux)
            self.aux2 = InceptionAux(spatial_dims, 528, n_classes, dropout_aux)
        else:
            self.aux1 = None
            self.aux2 = None

        self.classifier = nn.Sequential(
            AdpAvgPool(1),
            nn.Flatten(1),
            Dropout(dropout),
            nn.Linear(1024, n_classes)
        )

    def forward(self, data):
        x, aux2, aux1 = self.backbone(data)
        y = self.classifier(x)
        if self.training and self.aux1 is not None and self.aux2 is not None:
            y2 = self.aux2(aux2)
            y1 = self.aux1(aux1)
            return y, y2, y1
        else:
            return y
