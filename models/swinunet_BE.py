import torch
import torch.nn as nn


class SwinUNet_BE(nn.Module):
    def __init__(self, config, num_classes=None):
        super(SwinUNet_BE, self).__init__()

        embed_dim = config.MODEL_SWIN_EMBED_DIM  # 96
        if num_classes is None:
            num_classes = config.MODEL_NUM_CLASSES

        self.adapter = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1)
        )

        self.output = nn.Conv2d(
            in_channels=embed_dim,
            out_channels=num_classes,
            kernel_size=1,
            bias=False
        )

    def forward(self, x):
        x = self.adapter(x)
        return self.output(x)
