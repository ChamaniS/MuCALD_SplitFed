'''
import torch
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def forward_diffusion(x_0, beta,alpha_cum,t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_0).to(device)  # Sample noise from N(0, I)

    # Handle batch processing for t
    if isinstance(t, int):
        t = torch.tensor([t] * x_0.size(0), device=device)  # Expand scalar t to batch size
    #t = t.view(-1, 1, 1, 1)  # Reshape to match feature dimensions
    t = t.view(-1, 1, 1, 1).to(dtype=torch.long)  # Ensure t is a long tensor

    # Compute coefficients
    sqrt_alpha_cum_t = torch.sqrt(alpha_cum[t]).to(device)
    sqrt_one_minus_alpha_cum_t = torch.sqrt(1 - alpha_cum[t]).to(device)

    # Apply forward diffusion (Forward diffusion equation in the DDPM paper)
    x_t = sqrt_alpha_cum_t * x_0 + sqrt_one_minus_alpha_cum_t * noise    #linear combination of x_0 and the noise variable as in the DDPM paper
    return x_t
'''
import torch
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def forward_diffusion(x_0, beta, alpha_cum, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_0).to(device)  # Sample noise from N(0, I)

    # Ensure t is a tensor with the correct shape for batch processing
    if isinstance(t, int):
        t = torch.tensor([t] * x_0.size(0), device=device, dtype=torch.long)
    t = t.view(-1, 1, 1, 1)  # Reshape to match feature dimensions

    # Compute coefficients from precomputed alpha_cum
    sqrt_alpha_cum_t = torch.sqrt(alpha_cum[t]).to(device)
    sqrt_one_minus_alpha_cum_t = torch.sqrt(1 - alpha_cum[t]).to(device)

    # Apply forward diffusion equation (from DDPM)
    x_t = sqrt_alpha_cum_t * x_0 + sqrt_one_minus_alpha_cum_t * noise
    return x_t
