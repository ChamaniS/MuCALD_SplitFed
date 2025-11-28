import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    """Encodes time steps into embeddings using sinusoidal functions."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        # Ensure t is a tensor and has the correct device
        if isinstance(t, int):
            t = torch.tensor([t], dtype=torch.float32,
                             device=t.device if t.device else "cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(t, torch.Tensor) and t.dim() == 0:
            t = t.unsqueeze(0)

        half_dim = self.dim // 2
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=t.device) *
                        (-torch.log(torch.tensor(10000.0, dtype=torch.float32, device=t.device)) / (half_dim - 1)))
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class UNetBlock(nn.Module):
    """Basic UNet Block with Conv + GN + SiLU."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.activation = nn.SiLU()  # Swish activation

    def forward(self, x):
        x = self.activation(self.norm1(self.conv1(x)))
        x = self.activation(self.norm2(self.conv2(x)))
        return x


class ReverseDiffusion(nn.Module):
    """U-Net-based Reverse Diffusion Model with Time Conditioning."""

    def __init__(self, input_dim, hidden_dim, output_dim, time_emb_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim  # Add self.hidden_dim to use later
        self.output_dim = output_dim  # Add self.output_dim to use later

        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, hidden_dim),  # Ensure hidden_dim matches
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 2)  # Ensure hidden_dim * 2 matches
        )

        self.encoder1 = UNetBlock(input_dim, hidden_dim)
        self.encoder2 = UNetBlock(hidden_dim, hidden_dim * 2)
        self.bottleneck = UNetBlock(hidden_dim * 2, hidden_dim * 2)
        self.decoder1 = UNetBlock(hidden_dim * 2, hidden_dim)
        self.decoder2 = UNetBlock(hidden_dim, output_dim)

        self.final_conv = nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1)

    def forward(self, x_t, t):
        if isinstance(t, int):
            t = torch.tensor([t], dtype=torch.float32, device=x_t.device)
        elif isinstance(t, torch.Tensor) and t.dim() == 0:
            t = t.unsqueeze(0)

        t_embedding = self.time_mlp(t)  # Now properly projected to hidden_dim * 2

        # Ensure correct reshaping
        t_emb_1 = t_embedding[:, :self.hidden_dim].view(t.shape[0], self.hidden_dim, 1, 1)
        t_emb_2 = t_embedding[:, :self.hidden_dim * 2].view(t.shape[0], self.hidden_dim * 2, 1, 1)
        t_emb_b = t_embedding[:, :self.hidden_dim * 2].view(t.shape[0], self.hidden_dim * 2, 1, 1)
        t_emb_d1 = t_embedding[:, :self.hidden_dim].view(t.shape[0], self.hidden_dim, 1, 1)
        t_emb_d2 = t_embedding[:, :self.output_dim].view(t.shape[0], self.output_dim, 1, 1)

        # Forward pass with time conditioning
        enc1 = self.encoder1(x_t) + t_emb_1
        enc2 = self.encoder2(enc1) + t_emb_2
        bottleneck = self.bottleneck(enc2) + t_emb_b

        # Decoder with proper skip connections
        dec1 = self.decoder1(bottleneck + enc2) + t_emb_d1  # Add both enc1 and enc2
        dec2 = self.decoder2(dec1 + enc1) + t_emb_d2  # Add both enc1 and enc2 again

        noise_pred = self.final_conv(dec2)
        return noise_pred


def denoise(x_t, denoiser, alpha_cum_t, t, device):
    x_t = x_t.to(device)

    # Ensure t is a tensor with correct type and device
    if isinstance(t, int):
        t = torch.tensor([t], dtype=torch.float32, device=device)
    elif isinstance(t, torch.Tensor) and t.dim() == 0:
        t = t.unsqueeze(0)

    t = t.to(device).long()  # Ensure t is on the correct device and of type long

    # Use `index_select` for proper tensor indexing
    sqrt_alpha_cum_t = torch.index_select(alpha_cum_t, 0, t).to(device)
    sqrt_one_minus_alpha_cum_t = torch.index_select((1 - alpha_cum_t), 0, t).to(device)

    noise_pred = denoiser(x_t, t)  # Predict the noise from the noisy input

    # Calculate the denoised output by subtracting the predicted noise
    x_0_hat = (x_t - sqrt_one_minus_alpha_cum_t * noise_pred) / sqrt_alpha_cum_t
    return x_0_hat


def initialize_denoiser(input_dim, output_dim, hidden_dim, device="cuda"):
    denoiser = ReverseDiffusion(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim).to(device)
    return denoiser
