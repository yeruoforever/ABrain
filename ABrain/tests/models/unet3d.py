import unittest

import torch
from ...modelzoo.UNet3D import UInput, UStage, UStageDown, UStageUp, USkipCrop, UNet3D


class TestUNet3d(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def test_UInput(self):
        B, C_in, W, H, D = 2, 3, 132, 132, 116
        C_out = 32
        data = torch.rand(B, C_in, W, H, D)
        model = UInput(C_in, C_out)
        out = model(data)
        self.assertEqual(out.shape, (B, C_out, W-2, H-2, D-2))

    def test_UStage(self):
        B, C_in, W, H, D = 2, 64, 64, 64, 56
        C_out = 128
        data = torch.rand(B, C_in, W, H, D)
        conf = [64, 64, 128]
        model = UStage(conf)
        out = model(data)
        d = (len(conf)-1)*2
        self.assertEqual(out.shape, (B, C_out, W-d, H-d, D-d))

    def test_UStateDown(self):
        B, C_in, W, H, D = 2, 64, 64, 64, 56
        C_out = 128
        data = torch.rand(B, C_in, W, H, D)
        conf = [64, 64, C_out]
        model = UStageDown(conf)
        origin, pooled = model(data)
        d = (len(conf)-1)*2
        self.assertEqual(origin.shape, (B, C_out, (W-d), (H-d), (D-d)))
        self.assertEqual(
            pooled.shape, (B, C_out, (W-d)//2, (H-d)//2, (D-d)//2))

    def test_UStageUp(self):
        C_skip = 128
        B, C_in, W, H, D = 2, 256, 64, 64, 56
        C_out = 128
        skiped = torch.rand(B, C_skip, W*2, H*2, D*2)
        data = torch.rand(B, C_in, W, H, D)
        conf = [C_in, 128, C_out]
        model = UStageUp(C_skip, conf)
        out = model(skiped, data)
        d = (len(conf)-1)*2
        self.assertEqual(out.shape, (B, C_out, W*2-d, H*2-d, D*2-d))

    def test_USkips(self):
        stages = []
        target_size = [(18, 18, 14), (28, 28, 20), (48, 48, 32)]
        for size in target_size:
            stages.append(torch.rand(16, 32, 64, 64, 64))
        model = USkipCrop(target_size)
        outs = model(stages)
        for o, s in zip(outs, target_size):
            self.assertEqual(o.shape[-3:], s)

    def test_UNet3D(self):
        conf = [
            32, 64, 'd', 64, 128, 'd', 128, 256, 'd',
            256, 512,
            'u', 256, 256, 'u', 128, 128, 'u', 64, 64
        ]
        B, C_i, W, H, D = 2, 3, 132, 132, 116
        n_class = 3
        img = torch.rand(B, C_i, W, H, D)
        model = UNet3D(C_i, n_class, (W, H, D), conf)
        out = model(img)
        self.assertEqual(out.shape[-3:], (44, 44, 28))
        cnt = 0
        for k,v in model.state_dict().items():
            cnt+= v.numel()
        print(cnt) # our is 19080209 for n_class = 3
                   # origin paper in 19069955.
        

