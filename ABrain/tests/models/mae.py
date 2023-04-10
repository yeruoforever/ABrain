import unittest

import torch
import functools 

from ...modelzoo.MAE import sincos_pos_embed,PatchEmbed,MSA,ViTBlock,MaskedAutoEncoderViT

class TestMAE(unittest.TestCase):
    def setUp(self) -> None:
        self.gpu=torch.device("cuda:0")
    def test_pos_embed(self):
        dim = 64
        grid_size = 6
        cls_token=False
        embedding = sincos_pos_embed(dim,grid_size,cls_token)
        self.assertEqual(embedding.shape,(6,dim))
        grid_size=(3,4,5)
        embedding = sincos_pos_embed(dim*3,grid_size,cls_token)
        self.assertEqual(embedding.shape,(3*4*5,dim*3))
        cls_token = True
        embedding = sincos_pos_embed(dim*3,grid_size,cls_token)
        self.assertEqual(embedding.shape,(3*4*5+1,dim*3))

    def test_PatchEmbed(self):
        img_size = (128,128,128)
        patch_size = (16,16,16)
        B,C,D=16,3,128
        pe = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=C,
            embed_dim=D,
        ).to(self.gpu)
        x = torch.rand(B,C,*img_size,device=self.gpu)
        out = pe(x)
        grid = tuple(i//j for i,j in zip(img_size,patch_size))
        L = functools.reduce(lambda x,y:x*y,grid,1)
        self.assertEqual(out.shape,(B,L,D))
        pe = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=C,
            embed_dim=D,
            conv_patchify=False
        ).to(self.gpu)
        pixels=functools.reduce(lambda x,y:x*y,patch_size,1)
        x = torch.rand(B,L,C*pixels,device=self.gpu)
        out = pe(x)
        self.assertEqual(out.shape,(B,L,D))

    def test_MSA_ViTBlock(self):
        B,L,D,H=64,128,768,16
        msa = MSA(
            dim=D,
            num_heads=H
        ).to(self.gpu)
        x = torch.rand(B,L,D,device=self.gpu)
        out = msa(x)
        self.assertEqual(out.shape,(B,L,D))
        block = ViTBlock(D,H).to(self.gpu)
        out = block(x)
        self.assertEqual(out.shape,(B,L,D))

    def test_MAE(self):
        img_size = (256,320,320)
        patch_size = (16,16,16)
        B,C,D=2,3,768
        model = MaskedAutoEncoderViT(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=C,
            embed_dim=D,
            depth=24,
            num_heads=16,
            decoder_dim=384,
            decoder_depth=12,
            decoder_heads=8,
            # norm_pix_loss=True
        ).to(self.gpu)
        scaler = torch.cuda.amp.GradScaler()
        optimizer= torch.optim.Adam(model.parameters(),lr=0.001)
        for i in range(10):
            optimizer.zero_grad()
            x=torch.rand(B,C,*img_size,device=self.gpu)
            with torch.cuda.amp.autocast():
                loss,img,pred = model(x,0.7)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            print(loss.item())
        print(loss.shape,img.shape,pred.shape)