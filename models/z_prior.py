import torch
import torch.nn as nn

class ZPrior(nn.Module):
    """
    Takes per-image proxy vector y (len = P) and predicts per-node (mu, logvar)
    for each z_i in z (split into node_dims).
    """
    def __init__(self, y_dim: int, node_dims):
        super().__init__()
        self.node_dims = node_dims
        hid = max(16, y_dim*2)
        self.backbone = nn.Sequential(
            nn.Linear(y_dim, hid), nn.ReLU(),
            nn.Linear(hid, hid), nn.ReLU(),
        )
        self.mu_heads = nn.ModuleList([nn.Linear(hid, d) for d in node_dims])
        self.lv_heads = nn.ModuleList([nn.Linear(hid, d) for d in node_dims])

    def forward(self, y_vec):  # y_vec: (B, y_dim)
        h = self.backbone(y_vec)
        mu_list, logvar_list = [], []
        for mu_h, lv_h in zip(self.mu_heads, self.lv_heads):
            mu_list.append(mu_h(h))
            logvar_list.append(lv_h(h).clamp(min=-10.0, max=10.0))  # numeric safety
        return mu_list, logvar_list

def kl_gaussians(mu_q, logv_q, mu_p, logv_p):
    # KL(q||p) for diag Gaussians, summed over dims, mean over batch
    var_q = logv_q.exp(); var_p = logv_p.exp()
    term = (var_q + (mu_q - mu_p)**2) / (var_p + 1e-8)
    kl = 0.5 * torch.mean(torch.sum(logv_p - logv_q + term - 1.0, dim=1))
    return kl
