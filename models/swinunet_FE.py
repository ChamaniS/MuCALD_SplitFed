### swinunet_FE.py

import torch
import torch.nn as nn
from .swin_transformer_unet_skip_expand_decoder_sys import PatchEmbed
from timm.models.layers import trunc_normal_

class SwinUNet_FE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.patch_embed = PatchEmbed(
            img_size=config.DATA_IMG_SIZE,
            patch_size=config.MODEL_SWIN_PATCH_SIZE,
            in_chans=config.MODEL_SWIN_IN_CHANS,
            embed_dim=config.MODEL_SWIN_EMBED_DIM,
            norm_layer=nn.LayerNorm if config.MODEL_SWIN_PATCH_NORM else None)
        self.ape = config.MODEL_SWIN_APE
        if self.ape:
            num_patches = self.patch_embed.num_patches
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, config.MODEL_SWIN_EMBED_DIM))
            trunc_normal_(self.absolute_pos_embed, std=.02)
        self.pos_drop = nn.Dropout(p=config.MODEL_DROP_RATE)

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x = self.patch_embed(x)
        if self.ape:
            x += self.absolute_pos_embed
        x = self.pos_drop(x)
        return x