from typing import List, Callable, Sequence, Iterable

import torch
import torch.nn as nn
import logging

from .base import SegmentationModel


class CenterCrop3D(nn.Module):
    def __init__(self, size) -> None:
        super().__init__()
        self.size = size

    def forward(self, img: torch.Tensor):
        shape = torch.tensor(img.shape[-3:])
        diff = shape-self.size
        w = diff.div(2, rounding_mode='trunc').tolist()
        d = self.size.tolist()
        return img[:, :, w[0]:w[0]+d[0], w[1]:w[1]+d[1], w[2]:w[2]+d[2]]


class UInput(nn.Module):
    def __init__(self, in_chs: int, out_chs: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_chs, out_chs, kernel_size=3, stride=1),
            nn.BatchNorm3d(out_chs),
            nn.ReLU(inplace=True)
        )

    def forward(self, data):
        x = self.conv(data)
        return x


class UStage(nn.Module):
    def __init__(self, chs: List[int]) -> None:
        super().__init__()
        self.conv = nn.Sequential()
        ch_pre = chs[0]
        for ch in chs[1:]:
            self.conv.append(nn.Conv3d(ch_pre, ch, kernel_size=3, stride=1))
            self.conv.append(nn.BatchNorm3d(ch))
            self.conv.append(nn.ReLU(inplace=True))
            ch_pre = ch

    def forward(self, data):
        x = self.conv(data)
        return x


class UStageDown(UStage):
    def __init__(self, chs: List[int]) -> None:
        super().__init__(chs)
        self.pool = nn.MaxPool3d(kernel_size=2)

    def forward(self, data):
        x = super().forward(data)
        x_p = self.pool(x)
        return x, x_p


class UStageUp(UStage):
    def __init__(self, skip_ch, chs: List[int]) -> None:
        in_ch = chs[0]
        chs[0] = in_ch + skip_ch
        super().__init__(chs)
        self.up_conv = nn.ConvTranspose3d(
            in_ch, in_ch, kernel_size=2, stride=2)

    def forward(self, skiped, data):
        x_u = self.up_conv(data)
        x = torch.cat((skiped, x_u), dim=1)
        x = super().forward(x)
        return x


class UBottleneck(UStage):
    def __init__(self,in_ch:int, chs: List[int]) -> None:
        super().__init__([in_ch,*chs])
    pass


class UContracting(nn.Module):
    def __init__(self, in_ch, conf: List) -> None:
        super().__init__()
        self.stages = nn.ModuleList()
        chs = [in_ch, ]
        for layer in conf:
            if isinstance(layer, int):
                chs.append(layer)
                in_ch = layer
            elif layer == 'd':
                self.stages.append(UStageDown(chs))
                chs = [in_ch, ]

    def forward(self, x):
        stages = []
        for s in self.stages:
            x_, x = s(x)
            stages.append(x_)
        return stages, x


class UExpansive(nn.Module):
    def __init__(self, in_ch, skips: List, conf: List) -> None:
        super().__init__()
        chs = [in_ch, ]
        first_u = True
        self.stages = nn.ModuleList()
        i = 1
        for l in conf:
            if first_u and l == 'u':
                first_u = False
                continue
            elif isinstance(l, int):
                chs.append(l)
                in_ch = l
            elif l == 'u':
                self.stages.append(UStageUp(skips[-i], chs))
                chs = [in_ch, ]
                i += 1
        self.stages.append(UStageUp(skips[-i],chs))

    def forward(self, skips, x):
        stages = []
        for i in range(len(skips)):
            x = self.stages[i](skips[i], x)
            stages.append(x)
        return stages


class UOutput(nn.Module):
    def __init__(self, in_ch: int, n_class: int) -> None:
        super().__init__()
        self.conv_1x1 = nn.Sequential(
            nn.Conv3d(in_ch, n_class, kernel_size=1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.conv_1x1(x[-1])


class USkipCrop(nn.Module):
    def __init__(self, skip_size: List) -> None:
        super().__init__()
        self.crops = nn.ModuleList()
        for s in skip_size:
            if not isinstance(s,torch.Tensor):
                s = torch.tensor(s,dtype=torch.int)
            self.crops.append(CenterCrop3D(s))

    def forward(self, skips):
        outs = []
        for i, crop in enumerate(self.crops):
            outs.append(crop(skips[-i-1]))
        return outs


class UFrameWork(SegmentationModel):

    def __init__(self, n_class: int,
                 input_layer: nn.Module,
                 contracting: nn.Module,
                 bottleneck: nn.Module,
                 expansive: nn.Module,
                 output_layer: nn.Module,
                 skip_crops: nn.Module = nn.Identity()
                 ) -> None:
        super().__init__(n_class)
        self.input = input_layer
        self.contracting = contracting
        self.bottleneck = bottleneck
        self.expansive = expansive
        self.output = output_layer
        self.skip_crops = skip_crops

    def forward(self, data):
        x = self.input(data)
        stages, x = self.contracting(x)
        x = self.bottleneck(x)
        skips = self.skip_crops(stages)
        stages_out = self.expansive(skips, x)
        o = self.output(stages_out)
        return o


def u_deteil(input_size: List[int], art: List):
    s = torch.tensor(input_size)
    out_stage = []
    skip_size = []
    stage_channels = []
    ch = 0
    cnt = 1
    with torch.no_grad():
        for layer in art:
            if isinstance(layer, int):
                s.sub_(2)
                ch = layer
            elif layer == 'u':
                s.multiply_(2)
                skip_size.append(s.clone())
                # stage_channels[-cnt] += ch
                cnt += 1
            elif layer == 'd':
                out_stage.append(s.clone())
                s.div_(2, rounding_mode="trunc")
                stage_channels.append(ch)
        out_stage.append(s.clone())
    return s, skip_size, stage_channels, out_stage[1:]


def findfirst(f: Callable, l: Iterable):
    for i, e in enumerate(l):
        if f(e):
            return i
    return -1


def findlast(f: Callable, l: Sequence):
    ans = findfirst(f, reversed(l))
    if ans == -1:
        return ans
    else:
        return len(l)-ans-1


class UNet3D(UFrameWork):

    def __init__(self, in_chs, n_class: int, input_size, conf: List,) -> None:
        # 1,9
        # [32,64,'d',64,128,'d',128,256,'d',256,512,'u',256,256,'u',128,128,'u',64,64]
        a = findlast(lambda x: x == 'd', conf)
        b = findfirst(lambda x: x == 'u', conf)
        contracting = conf[:a+1]
        bottleneck = conf[a+1:b]
        expansive = conf[b:]
        out_size, skip_sizes, skip_feats, stage_sizes = u_deteil(
            input_size, conf)
        logging.info(f"[3D U-Net] output size:{out_size}")
        logging.info(f"[3D U-Net] skip size:{skip_sizes}")
        logging.info(f"[3D U-Net] skip features:{skip_feats}")
        logging.info(f"[3D U-Net] stage size:{stage_sizes}")
        super().__init__(
            n_class,
            input_layer=nn.Identity(),
            contracting=UContracting(in_chs, contracting),
            bottleneck=UBottleneck(contracting[-2],bottleneck),
            expansive=UExpansive(bottleneck[-1], skip_feats, expansive),
            output_layer=UOutput(expansive[-1], n_class),
            skip_crops=USkipCrop(skip_sizes)
        )

    def forward(self, *data):
        return super().forward(*data)
