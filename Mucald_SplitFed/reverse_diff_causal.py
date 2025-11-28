# reverse_diff_causal.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- sinusoidal timestep embedding (standard in diffusion) ----
def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """
    timesteps: (B,) int64
    returns: (B, dim) float
    """
    half = dim // 2
    device = timesteps.device
    freqs = torch.exp(
        -torch.arange(0, half, dtype=torch.float32, device=device)
        * (torch.log(torch.tensor(max_period, dtype=torch.float32, device=device)) / half)
    )  # (half,)
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)  # (B, half)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=1)  # (B, 2*half)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
    return emb

# ---- FiLM utilities ----
class FiLM(nn.Module):
    def __init__(self, cond_dim: int, channels: int):
        super().__init__()
        self.to_gamma = nn.Linear(cond_dim, channels)
        self.to_beta  = nn.Linear(cond_dim, channels)
    def forward(self, h: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # h: (B,C,H,W), c: (B,cond_dim)
        gamma = self.to_gamma(c).unsqueeze(-1).unsqueeze(-1)  # (B,C,1,1)
        beta  = self.to_beta(c).unsqueeze(-1).unsqueeze(-1)   # (B,C,1,1)
        return h * (1 + gamma) + beta

class Block(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, cond_dim: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.gn   = nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch)
        self.act  = nn.SiLU(inplace=True)
        self.film = FiLM(cond_dim, out_ch)
    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        h = self.gn(h)
        h = self.film(h, c)
        h = self.act(h)
        return h

# ---- Conditional denoiser that predicts epsilon ----
class ConditionalDenoiser(nn.Module):
    def __init__(self, input_dim: int, cond_dim: int, hidden_dim: int = 128, t_dim: int = 64):
        super().__init__()
        self.t_dim = t_dim
        # project [z_cond, t_emb] -> cond embedding used by FiLM
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim + t_dim, hidden_dim),
            nn.SiLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(True),
        )
        self.in_conv = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.block1  = Block(hidden_dim, hidden_dim, hidden_dim)
        self.block2  = Block(hidden_dim, hidden_dim, hidden_dim)
        self.out_conv = nn.Conv2d(hidden_dim, input_dim, 3, padding=1)

    def forward(self, x_t: torch.Tensor, t_vec: torch.Tensor, z_cond: torch.Tensor) -> torch.Tensor:
        # t embedding
        if t_vec.dtype != torch.long:
            t_vec = t_vec.long()
        t_emb = timestep_embedding(t_vec, self.t_dim).to(x_t.dtype)  # (B, t_dim)

        # combine with causal conditioning
        cond = torch.cat([z_cond, t_emb], dim=1)      # (B, cond_dim + t_dim)
        cond = self.cond_proj(cond)                   # (B, hidden_dim)

        # predict epsilon
        h = self.in_conv(x_t)
        h = self.block1(h, cond)
        h = self.block2(h, cond)
        eps_pred = self.out_conv(h)
        return eps_pred

def initialize_conditional_denoiser(input_dim: int, cond_dim: int, hidden_dim: int = 128, device: str = "cuda", t_dim: int = 64):
    """
    Returns a denoiser with forward(x_t, t_vec, z_cond) -> eps_pred.
    - input_dim: channels of x_t (e.g., 32)
    - cond_dim:  sum(node_dims)
    """
    model = ConditionalDenoiser(input_dim=input_dim, cond_dim=cond_dim, hidden_dim=hidden_dim, t_dim=t_dim)
    return model.to(device)
