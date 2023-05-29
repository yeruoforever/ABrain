from typing import *
from functools import reduce
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

import monai.networks.layers as layers
from timm.models.vision_transformer import Block


def times(a, b):
    return a * b


def prod(x: Sequence[int]) -> int:
    return reduce(times, x, 1)


def to_tuple(x: Union[int, Tuple[int]]):
    if isinstance(x, int):
        return (x,)
    if isinstance(x, tuple):
        return x
    return tuple(x)


def _sincos_pos_embed(dims: int, pos: Tensor):
    assert dims % 2 == 0
    pos = pos.flatten()
    omege = torch.arange(0, dims // 2, dtype=torch.float32)
    omege.div_(dims / 2)
    omege = 1.0 / 10000**omege

    x = torch.einsum("p,d->pd", pos, omege)
    emb_sin = x.sin()
    emb_cos = x.cos()

    return torch.cat((emb_cos, emb_sin), dim=1)


def sincos_pos_embed(
    embed_dims: int,
    grid_size: Union[int, Tuple[int]],
    cls_token: bool = False,
) -> Tensor:
    grid_size = to_tuple(grid_size)
    n = len(grid_size)
    assert embed_dims % n == 0
    axis = [torch.arange(0, gs, dtype=int) for gs in grid_size]
    grid = torch.meshgrid(axis, indexing="ij")
    embedding = torch.cat(
        [_sincos_pos_embed(embed_dims // n, gi) for gi in grid], dim=1
    )
    if cls_token:
        cls_embed = torch.zeros(1, embed_dims, dtype=torch.float32)
        embedding = torch.cat((cls_embed, embedding), dim=0)
    return embedding


def grid_size(img_size: Iterable, patch_size: Iterable) -> list:
    a = Tensor(img_size)
    b = Tensor(patch_size)
    reainder = torch.remainder(a, b)
    assert torch.all(reainder == 0)
    return (a // b).long().tolist()


class PatchEmbed(nn.Module):
    def __init__(
        self,
        img_size: Tuple[int],
        patch_size: Tuple[int],
        in_channels: int,
        embed_dim: int,
        norm_layer: Optional[nn.LayerNorm] = None,
        bias: bool = True,
        conv_patchify: bool = True,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        assert isinstance(img_size, tuple)
        assert isinstance(patch_size, tuple)
        spatial_size = len(patch_size)
        assert len(img_size) == spatial_size
        self.spatial_size = spatial_size

        self.grid_size = grid_size(img_size, patch_size)
        self.num_patches = prod(self.grid_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.conv_patchify = conv_patchify

        if self.conv_patchify:
            self.proj = layers.Conv["Conv", spatial_size](
                in_channels=in_channels,
                out_channels=embed_dim,
                kernel_size=patch_size,
                stride=patch_size,
                bias=bias,
            )
        else:
            self.proj = nn.Linear(
                in_features=prod(patch_size) * in_channels,
                out_features=embed_dim,
                bias=bias,
            )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, img: Tensor) -> Tensor:
        x = self.proj(img)  # B,C,X,Y,Z
        if self.conv_patchify:  # B,C,X,Y,Z -> B,C,N -> B,N,C
            x = x.flatten(-self.spatial_size).transpose(-1, -2)
        x = self.norm(x)
        return x


class MSA(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        assert dim % num_heads == 0, "`dim` must be divisible by `num_headers`"
        self.num_heads = num_heads
        self.dim_head = dim // num_heads
        # self.scale = self.dim_head**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.att_drop = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop) if proj_drop > 0.0 else nn.Identity()

    def forward(self, x):
        B, L, C = x.shape
        qkv = self.qkv(x)  # B,L,3C
        qkv = qkv.reshape(B, L, 3, self.num_heads, self.dim_head)  # B,L,3,h,d
        qkv = qkv.permute(2, 0, 3, 1, 4)  # 3,B,h,L,d
        q, k, v = qkv.unbind(dim=0)
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.att_drop)  # B,h,L,d
        x = x.transpose(1, 2).reshape(B, L, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class ViTBlock(Block):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4,
        qkv_bias=False,
        drop=0,
        attn_drop=0,
        init_values=None,
        drop_path=0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__(
            dim,
            num_heads,
            mlp_ratio,
            qkv_bias,
            proj_drop=drop,
            attn_drop=attn_drop,
            init_values=init_values,
            drop_path=drop_path,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )
        del self.attn
        self.attn = MSA(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )


class MaskedAutoEncoderViT(nn.Module):
    def __init__(
        self,
        img_size: Tuple[int],
        patch_size: Tuple[int],
        in_channels: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        decoder_dim: int,
        decoder_depth: int,
        decoder_heads: int,
        mlp_ratio: float = 4.0,
        norm_layer: Optional[nn.LayerNorm] = nn.LayerNorm,
        norm_pix_loss: bool = False,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.img_size = img_size
        self.C = in_channels

        self.patch_embed = PatchEmbed(
            img_size,
            patch_size,
            in_channels,
            embed_dim,
            norm_layer,
            conv_patchify=False,
        )
        grid_size = self.patch_embed.grid_size
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        pos_embed = sincos_pos_embed(embed_dim, grid_size, cls_token=True)
        self.pos_embed = nn.Parameter(pos_embed.unsqueeze(0))
        self.encode_blocks = nn.ModuleList(
            [
                ViTBlock(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer if norm_layer else nn.Identity,
                )
                for _ in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

        self.decode_embed = nn.Linear(embed_dim, decoder_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        pos_embed = sincos_pos_embed(decoder_dim, grid_size, cls_token=True)
        self.decoder_pos = nn.Parameter(pos_embed.unsqueeze(0))
        self.decode_blocks = nn.ModuleList(
            [
                ViTBlock(
                    decoder_dim,
                    decoder_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer if norm_layer else nn.Identity,
                )
                for _ in range(decoder_depth)
            ]
        )
        self.decoder_norm = norm_layer(decoder_dim) if norm_layer else nn.Identity()

        self.norm_pix_loss = norm_pix_loss
        self.patch_restore = nn.Linear(
            decoder_dim, prod(patch_size) * self.C, bias=True
        )

        self.initialize_weight()

    def initialize_weight(self):
        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs: Tensor):
        """
        imgs: (N,C,D,W,H) or (N,C,W,H)

        patch_size = (d,w,h) or (w,h)

        x: (N,L,dwh*C)
        """
        if self.patch_embed.spatial_size == 3:
            B, C, D, W, H = imgs.shape
            d, w, h = self.patch_embed.patch_size
            nD, nW, nH = grid_size((D, W, H), (d, w, h))
            x = imgs.reshape(B, C, nD, d, nW, w, nH, h)
            x = torch.einsum("bcdihjwk->bdhwijkc", x)
            x = x.reshape(B, nD * nW * nH, d * w * h * C)
            return x
        elif self.patch_embed.spatial_size == 2:
            B, C, W, H = imgs.shape
            w, h = self.patch_embed.patch_size
            nW, nH = grid_size((W, H), (w, h))
            x = imgs.reshape(B, C, nW, w, nH, h)
            x = torch.einsum("bchjwk->bhwjkc", x)
            x = x.reshape(B, nW * nH, w * h * C)
            return x

    def unpatchify(self, x: Tensor):
        """
        x: (B,L,dwh*C) or (B,L,wh*C)

        imgs: (B,C,D,W,H) or (B,C,W,H)
        """
        B = x.shape[0]
        if self.patch_embed.spatial_size == 3:
            d, w, h = self.patch_embed.patch_size
            nD, nW, nH = grid_size((self.img_size, (d, w, h)))
            x = x.reshape(B, nD, nW, nH, d, w, h, self.C)
            x = torch.einsum("bijkdwhc->bcidjhkw", x)
            x = x.reshape(B, self.C, nD * d, nW * w, nH * h)
            return x

        elif self.patch_embed.spatial_size == 2:
            w, h = self.patch_embed.patch_size
            nW, nH = grid_size((self.img_size, (w, h)))
            x = x.reshape(B, nW, nH, w, h, self.C)
            x = torch.einsum("bjkwhc->bcjhkw", x)
            x = x.reshape(B, self.C, nW * w, nH * h)
            return x

    def random_masking(self, B: int, L: int, device, mask_ratio: float):
        len_keep = int((1 - mask_ratio) * L)
        noise = torch.rand(B, L, device=device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # indes of keep (B,L)
        ids_keep = ids_shuffle[:, :len_keep]
        ids_drop = ids_shuffle[:, len_keep:]
        return ids_keep, ids_drop, ids_restore

    def forward(self, img: Tensor, mask_ratio: float):
        img_token = self.patchify(img)
        latent, drop, restore = self.forward_encoder(img_token, mask_ratio)
        pred = self.forward_decoder(latent, drop, restore)
        restore_droped = self.patch_restore(pred)
        ids = drop.unsqueeze(-1).repeat(1, 1, img_token.shape[-1])
        img_droped = img_token.gather(dim=1, index=ids)
        loss = self.forward_loss(img_droped, restore_droped)
        return loss, img_droped, restore_droped

    def forward_encoder(self, img_token: Tensor, mask_ratio: float):
        B, L, D = img_token.shape
        device = img_token.device
        keep, drop, restore = self.random_masking(B, L, device, mask_ratio)
        keep_ = keep.unsqueeze(dim=-1).repeat(1, 1, D)
        img_masked = img_token.gather(dim=1, index=keep_)
        x = self.patch_embed(img_masked)
        postion = self.pos_embed[:, 1:, :].repeat(B, 1, 1)
        keep_ = keep.unsqueeze(dim=-1).repeat(1, 1, x.shape[-1])
        postion = postion.gather(dim=1, index=keep_)
        x = x + postion
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_token = cls_token.repeat(B, 1, 1)
        x = torch.cat([cls_token, x], dim=1)
        for block in self.encode_blocks:
            x = block(x)
        x = self.norm(x)
        return x, drop, restore

    def forward_decoder(self, x: Tensor, drop: Tensor, restore: Tensor):
        B, l = drop.shape
        x = self.decode_embed(x)
        mask_tokens = self.mask_token.repeat(B, l, 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        restore = restore.reshape(B, -1, 1).repeat(1, 1, x.shape[-1])
        x_ = x_.gather(dim=1, index=restore)
        x = torch.cat([x[:, :1, :], x_], dim=1)

        x = x + self.decoder_pos
        for block in self.decode_blocks:
            x = block(x)
        x = self.decoder_norm(x)
        x = x[:, 1:, :]
        drop = drop.reshape(B, -1, 1).repeat(1, 1, x.shape[-1])
        x = x.gather(dim=1, index=drop)
        return x

    def forward_loss(self, img_masked: Tensor, pred: Tensor):
        """
        img_masked: (B,l,D)
        pred: (B,l,D)
        """
        if self.norm_pix_loss:
            mean = img_masked.mean(dim=-1, keepdim=True)
            var = img_masked.var(dim=-1, keepdim=True)
            img_masked = (img_masked - mean) / torch.sqrt(var + 1e-6)
        loss = (pred - img_masked) ** 2  # B,l,D
        loss = loss.mean(dim=-2)
        return loss.mean()
