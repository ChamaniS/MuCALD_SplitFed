import torch
import torch.nn as nn
from .swin_transformer_unet_skip_expand_decoder_sys import (
    PatchMerging, BasicLayer, PatchExpand, FinalPatchExpand_X4, BasicLayer_up
)


class SwinUNet_SS(nn.Module):
    def __init__(self, config):
        super(SwinUNet_SS, self).__init__()

        embed_dim = config.MODEL_SWIN_EMBED_DIM
        depths = config.MODEL_SWIN_DEPTHS
        num_heads = config.MODEL_SWIN_NUM_HEADS
        window_size = config.MODEL_SWIN_WINDOW_SIZE
        mlp_ratio = config.MODEL_SWIN_MLP_RATIO
        patch_size = config.MODEL_SWIN_PATCH_SIZE
        img_size = config.DATA_IMG_SIZE
        qkv_bias = config.MODEL_SWIN_QKV_BIAS
        drop_path_rate = config.MODEL_DROP_PATH_RATE
        drop_rate = config.MODEL_DROP_RATE
        attn_drop_rate = config.MODEL_DROP_RATE
        use_checkpoint = config.TRAIN_USE_CHECKPOINT

        self.num_layers = len(depths)
        self.patches_resolution = (img_size // patch_size, img_size // patch_size)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # -------------------- Encoder --------------------
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                input_resolution=(self.patches_resolution[0] // (2 ** i_layer),
                                  self.patches_resolution[1] // (2 ** i_layer)),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=nn.LayerNorm,
                downsample=PatchMerging if i_layer < self.num_layers - 1 else None,
                use_checkpoint=use_checkpoint
            )
            self.layers.append(layer)

        self.norm = nn.LayerNorm(int(embed_dim * 2 ** (self.num_layers - 1)))

        # -------------------- Decoder with Skip Connections --------------------
        self.layers_up = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()
        for i_layer in range(self.num_layers):
            cur_dim = int(embed_dim * 2 ** (self.num_layers - 1 - i_layer))
            prev_dim = cur_dim * 2 if i_layer > 0 else cur_dim

            # Linear projection for concatenated features
            concat_linear = nn.Linear(prev_dim, cur_dim) if i_layer > 0 else nn.Identity()
            self.concat_back_dim.append(concat_linear)

            # Decoder layer
            if i_layer == 0:
                layer_up = PatchExpand(
                    input_resolution=(img_size // patch_size // (2 ** (self.num_layers - 1)),
                                      img_size // patch_size // (2 ** (self.num_layers - 1))),
                    dim=cur_dim, dim_scale=2, norm_layer=nn.LayerNorm
                )
            else:
                layer_up = BasicLayer_up(
                    dim=cur_dim,
                    input_resolution=(img_size // patch_size // (2 ** (self.num_layers - 1 - i_layer)),
                                      img_size // patch_size // (2 ** (self.num_layers - 1 - i_layer))),
                    depth=depths[self.num_layers - 1 - i_layer],
                    num_heads=num_heads[self.num_layers - 1 - i_layer],
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[sum(depths[:self.num_layers - 1 - i_layer]):
                                  sum(depths[:self.num_layers - i_layer])],
                    norm_layer=nn.LayerNorm,
                    upsample=PatchExpand if i_layer < self.num_layers - 1 else None,
                    use_checkpoint=use_checkpoint
                )

            self.layers_up.append(layer_up)

        self.norm_up = nn.LayerNorm(embed_dim)

        # Final upsample to full resolution
        self.final_expand = FinalPatchExpand_X4(
            input_resolution=(img_size // patch_size, img_size // patch_size),
            dim=embed_dim,
            dim_scale=4
        )

    def forward(self, x):
        # ---- Encoder ----
        x_downsample = []
        for layer in self.layers:
            x_downsample.append(x)
            x = layer(x)

        x = self.norm(x)

        # ---- Decoder with skip connections ----
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x)
            else:
                x = torch.cat([x, x_downsample[self.num_layers - 1 - inx]], dim=-1)
                x = self.concat_back_dim[inx](x)
                x = layer_up(x)

        x = self.norm_up(x)

        # ---- Final expand to [B, C, H, W] ----
        x = self.final_expand(x)
        B, L, C = x.shape
        H = W = int(L ** 0.5)
        x = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        return x
