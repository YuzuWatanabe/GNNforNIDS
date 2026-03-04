

# --------------------------------------------------------
# SimMIM
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Zhenda Xie
# --------------------------------------------------------

from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

from modules.swin_transformer.models.swin_transformer import SwinTransformer
#from .swin_transformer_v2 import SwinTransformerV2


def norm_targets(targets, patch_size):
    assert patch_size % 2 == 1
    
    targets_ = targets
    targets_count = torch.ones_like(targets)

    targets_square = targets ** 2.
    
    targets_mean = F.avg_pool2d(targets, kernel_size=patch_size, stride=1, padding=patch_size // 2, count_include_pad=False)
    targets_square_mean = F.avg_pool2d(targets_square, kernel_size=patch_size, stride=1, padding=patch_size // 2, count_include_pad=False)
    targets_count = F.avg_pool2d(targets_count, kernel_size=patch_size, stride=1, padding=patch_size // 2, count_include_pad=True) * (patch_size ** 2)
    
    targets_var = (targets_square_mean - targets_mean ** 2.) * (targets_count / (targets_count - 1))
    targets_var = torch.clamp(targets_var, min=0.)
    
    targets_ = (targets_ - targets_mean) / (targets_var + 1.e-6) ** 0.5
    
    return targets_


class SwinTransformerForSimMIM(SwinTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        assert self.num_classes == 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.w = nn.Parameter(torch.zeros(1, 1, self.embed_dim).to(self.device)) # maskの重みづけ（学習可能なパラメータ化）
        trunc_normal_(self.w, mean=0., std=.02)

    def forward(self, x, mask):
        """
        x: (B, C_in, H_in, W_in)
        z: (B, P_h * P_w, patch_embed_dim) = (128, 55*55, 64)
        mask: (1, 1, 64)
        """
        z = self.patch_embed(x)
        B, L, D = z.shape
        
        w = self.w.expand(B, L, -1)

        if mask is not None:
            # maskかけている個所は重みゼロにし、必要なパラメータなら学習して増加させていく
            mask = mask.expand(-1, L, -1).contiguous().view(1,L,D).to(self.device)
            z = torch.where(mask.bool(), w, z)
        
        if self.ape:
            z = z + self.absolute_pos_embed
        z = self.pos_drop(z)

        for layer in self.layers:
            z = layer(z)

        z = self.norm(z)

        return z

    @torch.jit.ignore
    def no_weight_decay(self):
        return super().no_weight_decay() | {'mask_token'}

class SimMIM(nn.Module):
    def __init__(self, config, encoder, encoder_stride, in_chans, patch_size=8):
        super().__init__()
        self.config = config
        self.encoder = encoder
        self.encoder_stride = encoder_stride
        self.decoder = nn.Linear(config.MODEL.SWIN.EMBED_DIM*config.MODEL.SWIN.PATCH_SIZE, 128)
        #self.decoder_norm = nn.LayerNorm(128)
        self.in_chans = in_chans
        self.patch_size = patch_size

    def forward(self, x, mask=None, head=None):            
        z = self.encoder(x, mask)
        # パッチ次元に沿って平均を計算
        z = z.mean(dim=1)
        x_rec = self.decoder(z)
        if head is not None:
            x_rec = head(x_rec)
        
        return x_rec

    @torch.jit.ignore
    def no_weight_decay(self):
        if hasattr(self.encoder, 'no_weight_decay'):
            return {'encoder.' + i for i in self.encoder.no_weight_decay()}
        return {}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        if hasattr(self.encoder, 'no_weight_decay_keywords'):
            return {'encoder.' + i for i in self.encoder.no_weight_decay_keywords()}
        return {}

def build_simmim(config):
    model_type = config.MODEL.TYPE
    if model_type == 'swin':
        encoder = SwinTransformerForSimMIM(
            img_size=config.DATA.IMG_SIZE,
            patch_size=config.MODEL.SWIN.PATCH_SIZE,
            in_chans=config.MODEL.SWIN.IN_CHANS,
            num_classes=0,
            embed_dim=config.MODEL.SWIN.EMBED_DIM,
            depths=config.MODEL.SWIN.DEPTHS,
            num_heads=config.MODEL.SWIN.NUM_HEADS,
            window_size=config.MODEL.SWIN.WINDOW_SIZE,
            mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
            qkv_bias=config.MODEL.SWIN.QKV_BIAS,
            qk_scale=config.MODEL.SWIN.QK_SCALE,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            ape=config.MODEL.SWIN.APE,
            patch_norm=config.MODEL.SWIN.PATCH_NORM,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT
        )

        encoder_stride = 32
        in_chans = config.MODEL.SWIN.IN_CHANS
        patch_size = config.MODEL.SWIN.PATCH_SIZE
        
    else:
        raise NotImplementedError(f"Unknown pre-train model: {model_type}")

    model = SimMIM(config=config, encoder=encoder, encoder_stride=encoder_stride, in_chans=in_chans)

    return model