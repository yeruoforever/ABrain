from typing import List
import torch
import torch.nn as nn

from .base import SegmentationModel


class VInput(nn.Module):
    def __init__(self, in_chs: int, out_chs: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_chs, out_chs//2, kernel_size=5, stride=1, padding=2),
            nn.PReLU(out_chs//2)
        )
        self.pooling = nn.Conv3d(out_chs//2, out_chs, kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)+x
        x_sampling = self.pooling(x)
        return x, x_sampling


class VOutput(nn.Module):
    def __init__(self, in_chs: int, out_chs: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose3d(in_chs, in_chs, 5, 1, 2),
            nn.PReLU(in_chs)
        )
        self.segmenatation = nn.Sequential(
            nn.Conv3d(in_chs, out_chs, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=-4)
        x = self.conv(x)+x
        x = self.segmenatation(x)
        return x


class VStageDown(nn.Module):
    def __init__(self, in_chs: int, out_chs: int, n_convs: int) -> None:
        super().__init__()
        self.rediualBlock = nn.Sequential()
        for _ in range(n_convs):
            self.rediualBlock.append(
                nn.Conv3d(in_chs, in_chs, 5, 1, 2))
            self.rediualBlock.append(nn.PReLU(in_chs))
        self.sample = nn.Conv3d(in_chs, out_chs, 2, 2)

    def forward(self, x):
        x = torch.add(x, self.rediualBlock(x))
        x_sampled = self.sample(x)
        return x, x_sampled


class VStageUp(nn.Module):
    def __init__(self, in_chs: int, out_chs: int, n_convs: int) -> None:
        super().__init__()
        self.rediualBlock = nn.Sequential()
        for _ in range(n_convs):
            self.rediualBlock.append(
                nn.Conv3d(in_chs, in_chs, 5, 1, 2))
            self.rediualBlock.append(nn.PReLU(in_chs))
        self.sample = nn.ConvTranspose3d(in_chs, out_chs//2, 2, 2)

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=-4)
        x = torch.add(x, self.rediualBlock(x))
        x = self.sample(x)
        return x


class VEncoder(nn.Module):
    def __init__(self, in_chs: int, block_size=List[int], max_chs: int = 256) -> None:
        super().__init__()
        self.stages = nn.Sequential()
        for sz in block_size:
            out = min(max_chs, in_chs*2)
            block = VStageDown(in_chs, out, sz)
            self.stages.append(block)
            in_chs = out
        self.out_chs = out

    def forward(self, x):
        skips = []
        for stage in self.stages:
            sk, x = stage(x)
            skips.append(sk)
        skips.reverse()
        return skips, x


class VDecoder(nn.Module):
    def __init__(self, in_chs: int, block_size=List[int], min_chs: int = 16) -> None:
        super().__init__()
        self.stages = nn.Sequential()
        block_size: list = block_size.copy()
        block_size.reverse()
        for sz in block_size:
            out_chs = max(min_chs, in_chs//2)
            block = VStageUp(in_chs, out_chs, sz)
            self.stages.append(block)
            in_chs = out_chs
        self.out_chs = in_chs

    def forward(self, x_stages, x):
        stages = []
        for stage, xi in zip(self.stages, x_stages):
            x = stage(xi, x)
            stages.append(x)
        return stages, x


class VBottleneck(nn.Module):
    def __init__(self, in_chs: int, out_chs: int, n_convs: int,) -> None:
        super().__init__()
        self.conv = nn.Sequential()
        for _ in range(n_convs):
            self.conv.append(nn.Conv3d(in_chs, in_chs, 5, 1, 2))
            self.conv.append(nn.PReLU(in_chs))
        self.upsample = nn.Sequential(
            nn.ConvTranspose3d(in_chs, out_chs, 2, 2),
            nn.PReLU(out_chs)
        )

    def forward(self, x):
        x = self.conv(x)+x
        x = self.upsample(x)
        return x


class VNetFrameWork(SegmentationModel):
    def __init__(self, n_class: int,
                 input_layer: nn.Module, encoder: nn.Module, decoder: nn.Module, bottleneck: nn.Module, output_layer: nn.Module
                 ) -> None:
        super().__init__(n_class)
        self.input = input_layer
        self.encoder = encoder
        self.bottleneck = bottleneck
        self.decoder = decoder
        self.output = output_layer

    def forward(self, x):
        x_ori, x = self.input(x)
        x_stages, x = self.encoder(x)
        x_bot = self.bottleneck(x)
        x_stages, x = self.decoder(x_stages, x_bot)
        x = self.output(x_ori, x)
        return x_stages, x


class VNet(VNetFrameWork):
    def __init__(self, n_class: int, ori_chs: int, in_chs: int, block_size: List[int], max_chs: int = 256, min_chs=16) -> None:
        input_layer = VInput(ori_chs, in_chs)
        encoder = VEncoder(in_chs, block_size[:-1], max_chs)
        bottleneck = VBottleneck(
            encoder.out_chs, encoder.out_chs//2, block_size[-1])
        decoder = VDecoder(encoder.out_chs, block_size[:-1], min_chs)
        output_layer = VOutput(decoder.out_chs, n_class)
        super().__init__(
            n_class=n_class,
            input_layer=input_layer,
            encoder=encoder,
            bottleneck=bottleneck,
            decoder=decoder,
            output_layer=output_layer
        )
