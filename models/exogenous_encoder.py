# models/exogenous_encoder.py
import torch
import torch.nn as nn
from typing import List

class ConvGNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1, gn_groups: int = 8):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False)
        # keep groups <= out_ch to avoid invalid GroupNorm configs
        self.gn = nn.GroupNorm(num_groups=min(gn_groups, out_ch), num_channels=out_ch)
        self.act = nn.SiLU(inplace=True)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.gn(self.conv(x)))

class ExogenousEncoder(nn.Module):
    """
    Encodes a feature map (B, C, H, W) into per-node exogenous variables u_i.
    If variational=True, returns reparameterized samples u_i plus (mu_i, logvar_i).
    """
    def __init__(self, in_channels: int, node_dims: List[int], variational: bool = True, hidden: int = 64):
        super().__init__()
        self.node_dims = list(node_dims)
        self.variational = variational

        # lightweight shared trunk
        self.trunk = nn.Sequential(
            ConvGNAct(in_channels, hidden, k=3, s=1, p=1),
            ConvGNAct(hidden,      hidden, k=3, s=1, p=1),
        )

        # one 1x1 head per node for mu and logvar (produces (B, d_i, H, W), then we GAP to (B, d_i))
        self.mu_heads = nn.ModuleList([nn.Conv2d(hidden, d, kernel_size=1, bias=True) for d in self.node_dims])
        self.lv_heads = nn.ModuleList([nn.Conv2d(hidden, d, kernel_size=1, bias=True) for d in self.node_dims])

    def forward(self, x: torch.Tensor):
        F = self.trunk(x)  # (B, hidden, H, W)

        u_list, mu_list, logvar_list = [], [], []
        for mu_h, lv_h in zip(self.mu_heads, self.lv_heads):
            mu_map = mu_h(F)                       # (B, d_i, H, W)
            lv_map = lv_h(F)                       # (B, d_i, H, W)
            mu = mu_map.mean(dim=(2, 3))           # (B, d_i)
            logvar = lv_map.mean(dim=(2, 3))       # (B, d_i)

            if self.variational:
                std = (0.5 * logvar).exp()
                eps = torch.randn_like(std)
                u = mu + std * eps                 # reparameterization
            else:
                u = mu
                logvar = torch.zeros_like(mu)

            u_list.append(u)
            mu_list.append(mu)
            logvar_list.append(logvar)

        # lists of length = num_nodes, each tensor shape (B, d_i)
        return u_list, mu_list, logvar_list
