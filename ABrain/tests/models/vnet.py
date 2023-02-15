import unittest

import torch

from ...modelzoo.VNet import VStageDown, VStageUp, VBottleneck, VEncoder, VDecoder, VInput, VOutput, VNetFrameWork, VNet


class TestVNet(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def test_VStageUp(self):
        B, C_in, W, H, D = 2, 16, 128, 128, 64
        C_out = C_in*2
        S = 3
        stage: VStageDown = VStageDown(C_in, C_out, S)
        n = len(stage.rediualBlock)
        # print(stage.rediualBlock)
        self.assertEqual(n, S*3)  # a conv with a pooling and a normalization with default.
        x = torch.rand(B, C_in, W, H, D)
        y, y_small = stage(x)
        self.assertTrue(True)
        self.assertEqual(y.shape, (B, C_in, W, H, D))
        self.assertEqual(y_small.shape, (B, C_out, W//2, H//2, D//2))

    def test_VStageDown(self):
        B, C_in, W, H, D = 2, 256, 16, 16, 8
        C_out = C_in//4
        S = 3
        stage: VStageUp = VStageUp(C_in, C_in//2, S)
        n = len(stage.rediualBlock)
        # print(stage.rediualBlock)
        self.assertEqual(n, S*3)  # a conv with a pooling and a normalization with default. 
        x1 = torch.rand(B, C_in//2, W, H, D)
        x2 = torch.rand(B, C_in//2, W, H, D)
        y = stage(x1, x2)
        self.assertTrue(True)
        self.assertEqual(y.shape, (B, C_out, W*2, H*2, D*2))

    def test_VIO(self):
        B, C_in, W, H, D = 2, 1, 128, 128, 64
        C_out = 32
        input_layer = VInput(C_in, C_out)
        # print(input_layer)
        x = torch.rand(B, C_in, W, H, D)
        y, y_small = input_layer(x)
        self.assertEqual(y.shape, (B, C_out//2, W, H, D))
        self.assertEqual(y_small.shape, (B, C_out, W//2, H//2, D//2))

        B, C_in, W, H, D = 2, 32, 128, 128, 64
        C_out = 7
        x1 = torch.rand(B, C_in//2, W, H, D)
        x2 = torch.rand(B, C_in//2, W, H, D)
        out_layer = VOutput(C_in, C_out)
        # print(out_layer)
        y = out_layer(x1, x2)
        self.assertEqual(y.shape, (B, C_out, W, H, D))

    def test_bottleneck(self):
        B, C_in, W, H, D = 2, 256, 8, 8, 4
        C_out = C_in//2
        S = 3
        x = torch.rand(B, C_in, W, H, D)
        bottleneck: VBottleneck = VBottleneck(C_in, C_out, S)
        # print(bottleneck)
        y = bottleneck(x)
        n = len(bottleneck.conv)
        self.assertEqual(n, S*2)  # a conv with a activate function
        self.assertEqual(y.shape, (B, C_out, W*2, H*2, D*2))

    def test_VEncoder(self):
        B, C_in, W, H, D = 2, 32, 64, 64, 32
        encoder = VEncoder(C_in, [2, 3, 3])
        x = torch.rand(B, C_in, W, H, D)
        x_s, x = encoder(x)
        self.assertEqual(x.shape, (B, C_in*8, W//8, H//8, D//8))
        self.assertEqual(len(x_s), 3)

    def test_VDecoder(self):
        B, C_in, W, H, D = 2, 256, 16, 16, 8
        decoder = VDecoder(C_in, [2, 3, 3])
        x = torch.rand(B, C_in//2, W, H, D)
        xs = [
            torch.rand(B, C_in//2, W, H, D),
            torch.rand(B, C_in//4, W*2, H*2, D*2),
            torch.rand(B, C_in//8, W*4, H*4, D*4),
        ]
        xs, x = decoder(xs, x)
        self.assertEqual(x.shape, (B, C_in//16, W*8, H*8, D*8))
        self.assertEqual(len(xs), 3)

    def test_VNetFramework(self):
        B, C, W, H, D = 2, 1, 128, 128, 64
        N = 10
        img = torch.rand(B, C, W, H, D)
        vnet = VNetFrameWork(
            N,
            input_layer=VInput(1, 32),
            encoder=VEncoder(32, [2, 3, 3]),
            bottleneck=VBottleneck(256, 128, 3),
            decoder=VDecoder(256, [2, 3, 3]),
            output_layer=VOutput(32, 10)
        )
        stages, y = vnet(img)
        self.assertEqual(y.shape, (B, N, W, H, D))
        self.assertEqual(len(stages), 3)

    def test_VNet(self):
        B, C, W, H, D = 2, 1, 128, 128, 64
        N = 10
        img = torch.rand(B, C, W, H, D)
        vnet = VNet(N, C, 32, [2, 3, 3, 3])
        xs, y = vnet(img)
        self.assertEqual(y.shape, (B, N, W, H, D))
        self.assertEqual(len(xs), 3)
        x=0
        for k,v in vnet.state_dict().items():
            x=x+v.numel()
        print(x)
            
