from typing import *
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Block


class MSA(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            *args,
            **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        assert dim % num_heads == 0, "`dim` must be divisible by `num_headers`"
        self.num_heads = num_heads
        self.dim_head = dim//num_heads
        # self.scale = self.dim_head**-0.5
        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.att_drop = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(
            proj_drop) if proj_drop > 0. else nn.Identity()

    def forward(self, x):
        B, L, C = x.shape
        qkv = self.qkv(x)  # B,L,3C
        qkv = qkv.reshape(B, L, 3, self.num_heads, self.dim_head)  # B,L,3,h,d
        qkv = qkv.permute(2, 0, 3, 1, 4)  # 3,B,h,L,d
        q, k, v = qkv.unbind(dim=0)
        x = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.att_drop)  # B,h,L,d
        x = x.transpose(1, 2).reshape(B, L, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class ViTBlock(Block):
    def __init__(self, dim, num_heads, mlp_ratio=4, qkv_bias=False, drop=0, attn_drop=0, init_values=None, drop_path=0, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__(dim, num_heads, mlp_ratio, qkv_bias, drop,
                         attn_drop, init_values, drop_path, act_layer, norm_layer)
        del self.attn
        self.attn = MSA(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop
        )


class SliceEncoder(nn.Module):
    def __init__(
        self,
        num_slices: int = 128,
        slice_size: Union[int, Tuple[int]] = 224,
        patch_size: Union[int, Tuple[int]] = 16,
        inner_mask_ratio: float = 0.5,
        outer_mask_ratio: float = 0.5,
        in_channels: int = 3,
        embed_dims: int = 768,
        norm_layer: Optional[nn.LayerNorm] = None,
        flatten: bool = True,
        bias: bool = True,
        spatial_dims: int = 2,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        slice_size = tuplelize(slice_size, spatial_dims)
        patch_size = tuplelize(patch_size, spatial_dims)
        self.M = num_patchs(slice_size, patch_size, spatial_dims)
        self.w = patch_size[0]
        self.h = patch_size[1]
        self.N = num_slices
        self.m = int(self.M*(1-inner_mask_ratio))
        self.n = int(self.N*(1-outer_mask_ratio))
        self.inner_pos = nn.Parameter(torch.zeros(1, self.M, embed_dims))

        self.flatten = flatten
        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dims,
            kernel_size=patch_size,
            stride=patch_size,
            bias=bias,
        )
        self.norm = norm_layer(embed_dims) if norm_layer else nn.Identity()

    def forward(self, img: torch.Tensor):
        # B,C,D,W,H
        B, C, D, W, H = img.shape
        E = self.proj.out_channels

        noise = torch.rand(B, self.N, device=img.device)  # B,N

        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :self.n]  # B,n
        ids_drop = ids_shuffle[:, self.n:]  # B,N-n
        slices_ids = ids_keep.reshape(B, 1, self.n, 1, 1).repeat(1, C, 1, W, H)
        slices_masked = torch.gather(img, dim=2, index=slices_ids)  # B,C,n,W,H
        slices_masked = slices_masked.reshape(
            B,
            C,
            self.n,
            self.w,
            int(W/self.w),
            self.h,
            int(H/self.h),
            # B,C,n,w,W/w,h,H/h
            # 0,1,2,3,  4,5,  6
            #         B,n,*,*,C,w,h
        ).permute(0, 2, 4, 6, 1, 3, 5).reshape(B, self.n, self.M, C, self.w, self.h)
        # B,n,M,C,w,h

        noise = torch.rand(B, self.n, self.M, device=img.device)
        ids_shuffle = torch.argsort(noise, dim=2)
        ids_keep_p = ids_shuffle[:, :, :self.m]  # B,n,m
        patch_ids = ids_keep_p.reshape(B, self.n, self.m, 1, 1, 1)
        patch_ids = patch_ids.repeat(1, 1, 1, C, self.w, self.h)
        patch_masked = torch.gather(slices_masked, dim=2, index=patch_ids)

        patch_masked = patch_masked.reshape(-1, C, self.w, self.h)
        x = self.proj(patch_masked)  # (B,n,m),C,1,1
        if self.flatten:
            # (B,n,m),C,1,1 -> (B,n,m),C
            x = x.flatten(-3)
        x = self.norm(x).reshape(B, self.n, -1, E)  # B,n,m,C
        #                    1,M,C   1,1,M,C          B, n, M, C
        pos_embed = self.inner_pos.unsqueeze(0).repeat(B, self.n, 1, 1)
        #        B,n,M   B,n,M,1       B,n,M,C
        ids = ids_keep_p.unsqueeze(-1).repeat(1, 1, 1, E)
        pos_embed = torch.gather(pos_embed, dim=2, index=ids)  # B,n,m,C
        x = x + pos_embed

        return x, ids_keep, ids_drop, ids_restore


class SliceDecoder(nn.Module):
    def __init__(
        self,
        slice_size: Union[int, Tuple[int]] = 224,
        patch_size: Union[int, Tuple[int]] = 16,
        out_channels: int = 3,
        embed_dim: int = 768,
        spatial_dims: int = 2,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        slice_size = tuplelize(slice_size, spatial_dims)
        patch_size = tuplelize(patch_size, spatial_dims)
        self.M = num_patchs(slice_size, patch_size, spatial_dims)
        self.w = patch_size[0]
        self.h = patch_size[1]
        self.W = slice_size[0]
        self.H = slice_size[1]
        self.inner_pos = nn.Parameter(torch.zeros(1, self.M, embed_dim))
        self.restruct = nn.Linear(embed_dim, out_channels*np.prod(patch_size))
        self.C = out_channels

    def forward(self, x: torch.Tensor):
        # x: B,n,C
        x = x.unsqueeze(2).repeat(1, 1, self.M, 1)  # B,n,M,C
        x = x+self.inner_pos.unsqueeze(0)  # B,n,M,C
        x = self.restruct(x)  # B,n,M,3*p*p
        B, n, M, D = x.shape
        W = int(self.W/self.w)
        H = int(self.H/self.h)
        #             0, 1, 2, 3, 4,      5,      6
        x = x.reshape(B, n, W, H, self.w, self.h, self.C)
        x = x.permute(0, 6, 1, 2, 4, 3, 5)
        x = x.reshape(B, self.C, n, self.W, self.H)
        return x


class MAENMAE(nn.Module):
    """

    ## Example:
    ```python
    gpu = torch.device("cuda:1")
    B, C, D, W, H = 6, 3, 256, 320, 320
    encoder = SliceEncoder(
        num_slices=D,
        slice_size=(W, H),
        patch_size=16,
        in_channels=C,
        embed_dims=128,
    )
    decoder = SliceDecoder(
        slice_size=(W, H),
        patch_size=16,
        out_channels=3,
        embed_dim=128
    )
    model = MAENMAE(
        slice_encoder=encoder,
        slice_decoder=decoder,
        encoder_dim=128,
        encoder_depth=16,
        decoder_dim=128,
        decoder_depth=8
    ).to(gpu)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scaler = torch.cuda.amp.GradScaler()
    for i in tqdm.tqdm(range(1000//6)):
        with torch.cuda.amp.autocast():
            image = torch.rand(B, C, D, W, H, device=gpu)
            optimizer.zero_grad()
            loss,out = model(image)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
            scaler.step(optimizer)
            scaler.update()
    print(out.shape)
    ```
    """

    def __init__(
            self,
            slice_encoder: nn.Module,
            slice_decoder: nn.Module,
            encoder_dim: int = 1024,
            encoder_depth: int = 24,
            encoder_heads: int = 16,
            decoder_dim: int = 512,
            decoder_depth: int = 8,
            decoder_heads: int = 16,
            mlp_ratio: float = 4.,
            norm_layer: nn.Module = nn.LayerNorm,
            *args,
            **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.slice_encoder = slice_encoder
        self.n_slices = self.slice_encoder.N
        self.cls_token = nn.Parameter(torch.zeros(1, 1, encoder_dim))
        self.pos_encoder = nn.Parameter(torch.zeros(
            1, 1+self.n_slices, encoder_dim))  # TODO
        self.inner_blocks = nn.ModuleList([
            ViTBlock(encoder_dim, encoder_heads, mlp_ratio,
                     qkv_bias=True, norm_layer=norm_layer)
            for _ in range(encoder_depth)
        ])
        self.outer_blocks = nn.ModuleList([
            ViTBlock(encoder_dim, encoder_heads, mlp_ratio,
                     qkv_bias=True, norm_layer=norm_layer)
            for _ in range(encoder_depth)
        ])
        self.encoder_norm = norm_layer(encoder_dim)
        self.decoder_embed = nn.Linear(encoder_dim, decoder_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.pos_decoder = nn.Parameter(torch.zeros(
            1, 1+self.n_slices, decoder_dim))  # TODO
        self.decoder_blocks = nn.ModuleList([
            ViTBlock(decoder_dim, decoder_heads, mlp_ratio,
                     qkv_bias=True, norm_layer=norm_layer)
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = norm_layer(decoder_dim)
        self.slice_decoder = slice_decoder
        self.fc = nn.Linear(encoder_dim, encoder_dim)

    def forward_encoder(self, img):
        y, ids_keep, ids_drop, ids_retore = self.slice_encoder(img)
        B, n, m, C = y.shape
        #   B,n                B,n,1
        ids = torch.unsqueeze(ids_keep, dim=-1).repeat(1, 1, C)  # B,n,C
        #          1,N,C .  B,N,C
        z_slice = self.pos_encoder[:, 1:, :].repeat(
            B, 1, 1).gather(index=ids, dim=1)
        cls_token = self.cls_token+self.pos_encoder[:, :1, :]  # 1,1,C
        cls_token = cls_token.repeat(B, 1, 1)
        z = torch.cat([cls_token, z_slice], dim=1)  # B,n+1,C
        for inner, outer in zip(self.inner_blocks, self.outer_blocks):
            y = y.reshape(-1, m, C)          # B,n,m,C -> (B,n),m,C
            y = inner(y)                   # (B,n）,m,C
            y = y.reshape(B, n, m, C)         # B,n,m,C
            y_ = self.fc(y).sum(dim=-2)    # B,n,C
            z[:, 1:, :] += y_                  # B,n,C
            z = outer(z)                   # B,1+n,C
        z = self.encoder_norm(z)
        return z, ids_drop, ids_retore

    def forward_decoder(self, latent, ids_restore):
        x = self.decoder_embed(latent)
        B, n, C = x.shape
        N = ids_restore.shape[1]
        mask_tokens = self.mask_token.repeat(B, N+1-n, 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        ids = ids_restore.unsqueeze(-1).repeat(1, 1, C)
        x_ = torch.gather(x_, dim=1, index=ids)
        x = torch.cat([x[:, :1, :], x_], dim=1)
        x = x+self.pos_decoder
        for block in self.decoder_blocks:
            x = block(x)
        x = self.decoder_norm(x)
        return x[:, 1:, :]

    def forward_loss(self, img, pred, ids_drop, whole_img=False):
        # img: B,C,D,W,H
        # pred: B,D,C
        # ids_drop: B,N-n
        B, C, D, W, H = img.shape
        ids = ids_drop.reshape(B, 1, -1, 1, 1)
        img_ = torch.gather(img, dim=2, index=ids)
        if whole_img:
            restruct = self.slice_decoder(pred)  # B,C,D,W,H
            res_ = torch.gather(restruct, dim=2, index=ids)
            loss = (img_-res_)**2
            loss = loss.mean()
            return loss, restruct
        else:
            ids = ids_drop.reshape(B, -1, 1)
            pred = torch.gather(pred, dim=1, index=ids)
            restruct = self.slice_decoder(pred)
            loss = (img_-restruct)**2
            loss = loss.mean()
            return loss, restruct

    def forward(self, img):
        latent, ids_drop, ids_restore = self.forward_encoder(img)
        pred = self.forward_decoder(latent, ids_restore)
        loss, restruction = self.forward_loss(img, pred, ids_drop)
        return loss, restruction


def tuplelize(sz: Union[int, Tuple[int]], n: int):
    if isinstance(sz, int):
        return tuple(sz for i in range(n))
    if isinstance(sz, tuple):
        assert len(sz) == n, f"the number of elements must be {n}!"
        return sz
    if isinstance(sz, list):
        return tuple(sz)
    if isinstance(sz, torch.Tensor):
        return tuplelize(sz.to_list(), n)
    if isinstance(sz, np.ndarray):
        return tuplelize(sz.tolist(), n)


IntOrTuple = Union[int, Tuple[int]]


def num_patchs(slice_size: IntOrTuple, patch_size: IntOrTuple, spatial_dims: int):
    slice_size = tuplelize(slice_size, spatial_dims)
    patch_size = tuplelize(patch_size, spatial_dims)
    n, m = np.divmod(slice_size, patch_size)
    assert np.all(
        m == 0), f"`Slice size`{slice_size} can not divide by `Patch size`{patch_size}!"
    return np.prod(n)
