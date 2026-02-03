# Multi-task SplitFed Training Script (Refactored to noise-prediction + KL regularizers, with warmup/ramp/hold + cosine beta)
import os, math
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from sklearn.metrics import jaccard_score
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
# from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from models.clientmodel_FE import UNET_FE
from models.servermodel import UNET_server
from models.clientmodel_BE import UNET_BE
from dataset import EmbryoDataset, HAMDataset, CVCDataset, covidCTDataset, FHPsAOPMSBDataset
# NOTE: denoise_cond/forward_diffusion no longer used in training; denoisers now predict epsilon.
from reverse_diff_causal import initialize_conditional_denoiser
from models.exogenous_encoder import ExogenousEncoder
from models.neural_scm import NeuralSCM
from scm_configs import get_scm_config
from proxy_tables import ProxyTable
from dataset_wrappers import WithFilenames
from models.z_prior import ZPrior

# ----------------- Hyperparams & Globals -----------------
DEVICE = "cuda"
NUM_CLIENTS = 5
LOCAL_EPOCHS = 12
COMM_ROUNDS = 10

# Reference (not used directly in loss anymore — see epoch scheduler below)
LAMBDA_PROXY = 1.5
LAMBDA_DIFF  = 0.5
LAMBDA_KLU   = 1e-4
LAMBDA_KLZ   = 1e-4

# ---- NEW: epoch scheduling knobs ----
# Warmup (seg-only) for first 2–3 local epochs, ramp over next 5–7, then hold.
WARMUP_EPOCHS = 3            # set 2 or 3 as desired
RAMP_EPOCHS   = 7            # set 5–7 as desired

# Target weights (choose within requested ranges)
LAMBDA_PROXY_TARGET = 0.40   # [0.2, 0.5]
LAMBDA_DIFF_TARGET  = 0.20   # [0.1, 0.3]
LAMBDA_KLU_TARGET   = 1.0e-4 # [5e-5, 2e-4]
LAMBDA_KLZ_TARGET   = 1.0e-4 # [5e-5, 2e-4]

def get_epoch_lambdas(ep: int):
    """Linear schedule per *local* epoch (0-indexed) inside each round."""
    if ep < WARMUP_EPOCHS:
        return 0.0, 0.0, 0.0, 0.0
    if ep < WARMUP_EPOCHS + RAMP_EPOCHS:
        factor = (ep - WARMUP_EPOCHS + 1) / float(RAMP_EPOCHS)
        factor = max(0.0, min(1.0, factor))
        return (
            LAMBDA_PROXY_TARGET * factor,
            LAMBDA_DIFF_TARGET  * factor,
            LAMBDA_KLU_TARGET   * factor,
            LAMBDA_KLZ_TARGET   * factor,
        )
    return (
        LAMBDA_PROXY_TARGET,
        LAMBDA_DIFF_TARGET,
        LAMBDA_KLU_TARGET,
        LAMBDA_KLZ_TARGET,
    )

# ---- NEW: cosine beta schedule (Nichol & Dhariwal) with T=400 ----
T = 400

def cosine_beta_schedule(T: int, s: float = 0.008, device: torch.device = None):
    # alpha_bar(t) = cos^2((t/T + s)/(1+s) * pi/2)
    # betas_t = 1 - alpha_bar(t+1)/alpha_bar(t)
    device = device or torch.device("cpu")
    t = torch.linspace(0, T, T + 1, device=device) / T
    f = torch.cos(((t + s) / (1 + s)) * torch.pi / 2) ** 2
    f = f / f[0]  # normalize so alpha_bar(0)=1
    betas = 1 - (f[1:] / f[:-1])
    return betas.clamp(1e-8, 0.999)

beta = cosine_beta_schedule(T, device=torch.device(DEVICE))
alpha = 1.0 - beta
alpha_cum = torch.cumprod(alpha, dim=0).to(DEVICE)

DATA_PATH = "XXXXX/Projects/Data/MTS2"
PROXY_DATA_PATH = "XXXXX/Projects/Causal-proxy/Proxy_variables_dir/Final/"

os.makedirs("BestModels", exist_ok=True)
os.makedirs("Outputs", exist_ok=True)

test_iou_wbg_all = {i: [] for i in range(NUM_CLIENTS)}
test_iou_nbg_all = {i: [] for i in range(NUM_CLIENTS)}

# ---- score smoothing + fairness blending (global, persists across rounds) ----
score_ema = [0.0] * NUM_CLIENTS
SCORE_EMA_BETA = 0.7           # higher = smoother
WEIGHT_FLOOR   = 0.25 / NUM_CLIENTS  # 25% mass uniformly reserved
TEMP           = 2.0           # >1 flattens differences

def normalize_scores(raw_scores):
    """EMA-smooth + temperature-scale + floor + renormalize."""
    global score_ema
    clean = []
    for i, sc in enumerate(raw_scores):
        # NaN-robust
        if isinstance(sc, float) and math.isnan(sc):
            sc = 0.0
        sc = float(sc)
        score_ema[i] = SCORE_EMA_BETA * score_ema[i] + (1 - SCORE_EMA_BETA) * sc
        clean.append(max(1e-8, score_ema[i]))

    softened = [s ** (1.0 / TEMP) for s in clean]
    tot = sum(softened) or 1.0
    softened = [s / tot for s in softened]

    K = len(softened)
    leftover = max(1e-8, 1.0 - WEIGHT_FLOOR * K)
    softened = [WEIGHT_FLOOR + leftover * s for s in softened]
    z = sum(softened) or 1.0
    softened = [s / z for s in softened]
    return softened

# ==== Diffusion ε-loss helpers ====
def extract_into(arr_1d, timesteps, x_shape):
    out = arr_1d.gather(0, timesteps.clamp_(0, arr_1d.shape[0]-1))
    return out.view(-1, *([1] * (len(x_shape) - 1)))

@torch.no_grad()
def predict_x0_from_eps(x_t, t_vec, eps_pred, alpha_cum):
    sqrt_ab   = extract_into(alpha_cum.sqrt(), t_vec, x_t.shape)
    sqrt_1mab = extract_into((1.0 - alpha_cum).sqrt(), t_vec, x_t.shape)
    x0_hat = (x_t - sqrt_1mab * eps_pred) / (sqrt_ab + 1e-8)
    return x0_hat

def diffusion_noise_loss(denoiser, x0, t_vec, z_cond, alpha_cum, want_grad_x0=False):
    """
    Train the denoiser with ε-pred MSE. If want_grad_x0=True, also return x0_hat
    that carries gradients to x0 (FE/SS) but NOT to denoiser.
    """
    eps = torch.randn_like(x0)
    sqrt_ab   = extract_into(alpha_cum.sqrt(), t_vec, x0.shape)
    sqrt_1mab = extract_into((1.0 - alpha_cum).sqrt(), t_vec, x0.shape)

    # 1) Build x_t for denoiser training WITHOUT sending grads into x0 path:
    x_t_detached = sqrt_ab * x0.detach() + sqrt_1mab * eps
    eps_pred = denoiser(x_t_detached, t_vec, z_cond)  # grads -> denoiser only
    loss = F.mse_loss(eps_pred, eps)

    if not want_grad_x0:
        with torch.no_grad():
            x0_hat = (x_t_detached - sqrt_1mab * eps_pred) / (sqrt_ab + 1e-8)
        return loss, x0_hat

    # 2) Rebuild x_t but now connected to x0 (so seg can backprop to FE/SS):
    x_t_for_seg = sqrt_ab * x0 + sqrt_1mab * eps
    # Use ε̂ but cut its grads so denoiser doesn't get seg grads:
    eps_pred_ng = eps_pred.detach()
    x0_hat_for_seg = (x_t_for_seg - sqrt_1mab * eps_pred_ng) / (sqrt_ab + 1e-8)
    return loss, x0_hat_for_seg

# ==== KL helpers ====
def kl_standard_normal(mu, logvar):
    # mean over batch, sum over dims
    return 0.5 * torch.mean(torch.sum(torch.exp(logvar) + mu**2 - 1.0 - logvar, dim=1))

def kl_gaussians(mu_q, logv_q, mu_p, logv_p, weight=None):
    device = mu_q.device
    mu_p, logv_p = mu_p.to(device), logv_p.to(device)
    var_q = logv_q.exp(); var_p = logv_p.exp()
    per = 0.5 * torch.sum(logv_p - logv_q + (var_q + (mu_q - mu_p)**2) / (var_p + 1e-8) - 1.0, dim=1)
    if weight is None:
        return per.mean()
    w = weight.to(device).float()
    denom = w.sum().clamp_min(1.0)
    return (per * w).sum() / denom

def ensure_z_logvars(scm_module, node_dims):
    try:
        dev = next(scm_module.parameters()).device
    except StopIteration:
        dev = torch.device(DEVICE)
    if not hasattr(scm_module, "z_logvar_params"):
        scm_module.z_logvar_params = nn.ParameterList([
            nn.Parameter(torch.full((d,), -2.0, device=dev))  # on the right device
            for d in node_dims
        ])
    else:
        new_list = nn.ParameterList()
        for p in scm_module.z_logvar_params:
            if p.device != dev:
                p = nn.Parameter(p.to(dev))
            new_list.append(p)
        scm_module.z_logvar_params = new_list
    return scm_module.z_logvar_params

def normalize_features(features):
    mean = features.mean(dim=(2, 3), keepdim=True)
    std = features.std(dim=(2, 3), keepdim=True)
    normalized_features = (features - mean) / (std + 1e-8)
    return normalized_features

# ----------------- Utils -----------------
def get_loader(img_dir, mask_dir, dataset_class, transform, batch_size=1, num_workers=1, with_names=True, shuffle=True):
    base = dataset_class(img_dir, mask_dir, transform=transform)
    ds = WithFilenames(base, img_dir) if with_names else base
    return DataLoader(ds, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)

def nodewise_scalar(z_list):
    out = {}
    for i, z in enumerate(z_list):
        if z.dim() > 1:
            reduce_dims = tuple(range(1, z.dim()))
            out[i] = z.mean(dim=reduce_dims)
        else:
            out[i] = z
    return out

class ComboLoss(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.dice = smp.losses.DiceLoss(smp.losses.MULTICLASS_MODE, from_logits=True)
        self.ce = nn.CrossEntropyLoss()
    def forward(self, logits, targets):
        # Dice + small CE tends to help boundaries; adjust if desired
        return self.dice(logits, targets) + 0.5 * self.ce(logits, targets)

def make_grad_tracker(model):
    return {name: torch.zeros_like(p, device=p.device)
            for name, p in model.named_parameters() if p.requires_grad}

# --------- NEW: IoU+size weighted fusion (no masks), FedBN-safe ---------
def weighted_fusion(models, weights, fallback_state=None):
    """
    weights: list of floats, normalized to sum=1 (we'll renorm just in case)
    FedBN-safe: skip BN buffers (keep fallback/global copy).
    """
    ref_sd = models[0].state_dict()
    w = torch.tensor(weights, dtype=torch.float32)
    w = w / (w.sum() if w.sum() > 0 else 1.0)

    model_sds = [m.state_dict() for m in models]
    out = {}

    def is_bn_buffer(name: str) -> bool:
        return any(t in name for t in ["running_mean", "running_var", "num_batches_tracked"])

    for key in ref_sd.keys():
        if is_bn_buffer(key):
            out[key] = (fallback_state[key] if fallback_state is not None else ref_sd[key]).to(ref_sd[key].device)
            continue
        acc = None
        for i, sd in enumerate(model_sds):
            val = sd[key]
            wi = w[i].to(val.device).type_as(val)
            acc = wi * val if acc is None else acc + wi * val
        out[key] = acc.to(ref_sd[key].device).type_as(ref_sd[key])
    return out

def plot_iou_curves(round_num):
    rounds = list(range(1, round_num + 1))
    plt.figure(figsize=(10, 5))
    for i in range(NUM_CLIENTS):
        plt.plot(rounds, test_iou_wbg_all[i], label=f"Client {i+1}")
    plt.xlabel("Communication Round"); plt.ylabel("IoU w/b")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig("Outputs/iou_wbg_curve.png"); plt.close()

    plt.figure(figsize=(10, 5))
    for i in range(NUM_CLIENTS):
        plt.plot(rounds, test_iou_nbg_all[i], label=f"Client {i+1}")
    plt.xlabel("Communication Round"); plt.ylabel("IoU n/b")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig("Outputs/iou_nobg_curve.png"); plt.close()

# ----------------- NEW: Dataset-level IoU accumulator -----------------
class IoUAccumulator:
    def __init__(self, num_classes: int):
        self.inter = torch.zeros(num_classes, dtype=torch.float64)
        self.union = torch.zeros(num_classes, dtype=torch.float64)

    @torch.no_grad()
    def update(self, preds_lbl: torch.Tensor, target: torch.Tensor, num_classes: int):
        # preds_lbl, target: (B,H,W) ints
        for c in range(num_classes):
            p = (preds_lbl == c)
            t = (target    == c)
            self.inter[c] += (p & t).sum().item()
            self.union[c] += (p | t).sum().item()

    def per_class_iou(self):
        return self.inter / (self.union + 1e-8)

# ----------------- Train / Eval -----------------
def train_local(loader, FE, SS, BE,
                denoiser1, denoiser2,
                exo1, scm1, exo2, scm2,
                opt, loss_fn, cid, num_classes,
                task_name, proxy_table: ProxyTable,
                mask_thresh=1e-5):

    FE.train(); SS.train(); BE.train()
    denoiser1.train(); denoiser2.train()
    exo1.train(); scm1.train()
    exo2.train(); scm2.train()

    ss_grad = make_grad_tracker(SS)
    d1_grad = make_grad_tracker(denoiser1)
    d2_grad = make_grad_tracker(denoiser2)
    num_updates = 0

    for ep in range(LOCAL_EPOCHS):
        # --- NEW: epoch-wise weights (warmup → ramp → hold) ---
        lam_proxy, lam_diff, lam_klu, lam_klz = get_epoch_lambdas(ep)
        print(f"[Sched] ep={ep+1}/{LOCAL_EPOCHS} | "
              f"λ_proxy={lam_proxy:.3g}, λ_diff={lam_diff:.3g}, "
              f"λ_KLu={lam_klu:.3g}, λ_KLz={lam_klz:.3g}")

        tloss = 0.0
        tcorrect = 0.0
        num_batches = 0

        # NEW: dataset-level IoU accumulator
        iou_acc = IoUAccumulator(num_classes)

        proxy_sq_sum_s1 = defaultdict(float)
        proxy_n_s1      = defaultdict(int)
        proxy_sq_sum_s2 = defaultdict(float)
        proxy_n_s2      = defaultdict(int)

        for batch in tqdm(loader, leave=False):
            if len(batch) == 3:
                data, target, fnames = batch
                fnames = list(fnames)
            else:
                data, target = batch
                fnames = [str(i) for i in range(data.size(0))]
            data, target = data.to(DEVICE), target.long().to(DEVICE)
            B = data.size(0)
            t_vec = torch.randint(1, T, (B,), device=DEVICE)

            # -------- split-1 --------
            x1 = FE(data)
            u_list_1, u_mu_1, u_lv_1 = exo1(x1)
            z_list_1 = scm1(u_list_1)
            z_causal_1 = scm1.as_vector(z_list_1)

            # Train d1; also get x1_hat that allows seg grads to flow into FE later if used
            diff_loss1, x1_hat = diffusion_noise_loss(denoiser1, x1, t_vec, z_causal_1.detach(), alpha_cum, want_grad_x0=True)

            # -------- split-2 --------
            x2 = SS(x1_hat)

            u_list_2, u_mu_2, u_lv_2 = exo2(x2)
            z_list_2 = scm2(u_list_2)
            z_causal_2 = scm2.as_vector(z_list_2)

            diff_loss2, x2_hat = diffusion_noise_loss(denoiser2, x2, t_vec, z_causal_2.detach(), alpha_cum, want_grad_x0=True)

            # segmentation on denoised features (no grads into denoisers)
            preds = BE(x2_hat)
            seg_loss = loss_fn(preds, target)

            # ---------- proxy losses ----------
            proxy_loss1 = 0.0
            proxy_loss2 = 0.0
            y_input = None
            y_mask_per_node = None
            if proxy_table is not None and proxy_table.enabled:
                z1_scalar = nodewise_scalar(z_list_1)
                z2_scalar = nodewise_scalar(z_list_2)
                targets_map = proxy_table.get_batch(fnames, device=DEVICE)

                if num_batches == 0:
                    total_found1 = {k: int(torch.isfinite(v).sum().item()) for k, v in targets_map.items()}
                    print(f"[DEBUG-SPLIT1] {task_name} | Client {cid} | first-batch matches: {total_found1}")
                    print(f"[DEBUG-SPLIT1] sample fnames (canon): {[ProxyTable._canon(f) for f in fnames[:5]]}")

                # build y vector (B, P) in node order + masks
                y_list, mask_list = [], []
                for i_node, nname in enumerate(scm1.node_names):
                    if nname in targets_map:
                        tgt = targets_map[nname]
                        mask = torch.isfinite(tgt)
                        # proxy MSE for each split (scalar z per node)
                        pred1 = z1_scalar[i_node]
                        pred2 = z2_scalar[i_node]
                        if mask.any():
                            err1 = (pred1[mask] - tgt[mask]) ** 2
                            err2 = (pred2[mask] - tgt[mask]) ** 2
                            proxy_loss1 += err1.mean()
                            proxy_loss2 += err2.mean()
                            proxy_sq_sum_s1[nname] += float(err1.sum().item())
                            proxy_sq_sum_s2[nname] += float(err2.sum().item())
                            proxy_n_s1[nname]      += int(mask.sum().item())
                            proxy_n_s2[nname]      += int(mask.sum().item())
                        y_list.append(torch.nan_to_num(tgt, nan=0.0))
                        mask_list.append(mask.float())
                    else:
                        y_list.append(torch.zeros(B, device=DEVICE))
                        mask_list.append(torch.zeros(B, device=DEVICE))
                y_input = torch.stack(y_list, dim=1)            # (B, P)
                y_mask_per_node = torch.stack(mask_list, dim=1) # (B, P)

            # ---------- KL(q(u|x)||N(0,I)) ----------
            kl_u1 = sum(kl_standard_normal(mu, lv) for mu, lv in zip(u_mu_1, u_lv_1))
            kl_u2 = sum(kl_standard_normal(mu, lv) for mu, lv in zip(u_mu_2, u_lv_2))
            kl_u = kl_u1 + kl_u2

            # ---------- KL(q(z|x)||p(z|y)) ----------
            kl_z = torch.tensor(0.0, device=DEVICE)
            if y_input is not None:
                mu_p_list, lv_p_list = scm1.zprior(y_input)  # should already be on DEVICE
                ensure_z_logvars(scm1, [z.shape[1] for z in z_list_1])
                ensure_z_logvars(scm2, [z.shape[1] for z in z_list_2])
                for idx, (z1, zlv1_param, z2, zlv2_param, mu_p, lv_p) in enumerate(
                        zip(z_list_1, scm1.z_logvar_params, z_list_2, scm2.z_logvar_params, mu_p_list, lv_p_list)
                ):
                    dev = z1.device
                    mu_p = mu_p.to(dev)
                    lv_p = lv_p.to(dev)
                    lv_q1 = zlv1_param.to(dev).view(1, -1).expand_as(z1)
                    lv_q2 = zlv2_param.to(dev).view(1, -1).expand_as(z2)
                    w = y_mask_per_node[:, idx].to(dev) if y_mask_per_node is not None else None

                    kl_z += kl_gaussians(z1, lv_q1, mu_p, lv_p, weight=w)
                    kl_z += kl_gaussians(z2, lv_q2, mu_p, lv_p, weight=w)

            # total loss (scheduled)
            loss = (
                seg_loss
                + lam_proxy * (proxy_loss1 + proxy_loss2)
                + lam_diff  * (diff_loss1 + diff_loss2)
                + lam_klu   * kl_u
                + lam_klz   * kl_z
            )

            opt.zero_grad()
            loss.backward()

            with torch.no_grad():
                for name, p in SS.named_parameters():
                    if p.grad is not None:
                        ss_grad[name].add_(p.grad.abs())
                for name, p in denoiser1.named_parameters():
                    if p.grad is not None:
                        d1_grad[name].add_(p.grad.abs())
                for name, p in denoiser2.named_parameters():
                    if p.grad is not None:
                        d2_grad[name].add_(p.grad.abs())

            opt.step()
            num_updates += 1
            num_batches += 1

            preds_lbl = torch.argmax(preds, dim=1)

            # NEW: accumulate dataset-level IoU
            iou_acc.update(preds_lbl, target, num_classes)

            tcorrect += (preds_lbl == target).float().mean().item()
            tloss += float(loss.item())

        # ---- epoch summary ----
        acc = 100.0 * (tcorrect / max(num_batches, 1))

        # NEW: compute with-bg and no-bg mIoU
        per_cls = iou_acc.per_class_iou().cpu().numpy()
        iou_wbg = float(per_cls.mean())
        iou_nbg = float(per_cls[1:].mean()) if num_classes > 1 else float('nan')

        print(
            f"Client {cid} | Epoch {ep + 1} | "
            f"Train Loss: {(tloss / max(num_batches, 1)):.4f} | Acc: {acc:.2f}% | "
            f"IoU w/bg: {iou_wbg:.4f} | IoU no/bg: {iou_nbg:.4f}"
        )

        # ---- per-proxy logging, split-1 & split-2 printed separately ----
        if hasattr(scm1, "node_names") and scm1.node_names:
            parts1, parts2 = [], []
            for nname in scm1.node_names:
                n1 = proxy_n_s1.get(nname, 0)
                n2 = proxy_n_s2.get(nname, 0)
                if n1 > 0:
                    mse1 = proxy_sq_sum_s1[nname] / n1
                    parts1.append(f"{nname}={mse1:.4f} (n={n1})")
                else:
                    parts1.append(f"{nname}=NA (n=0)")
                if n2 > 0:
                    mse2 = proxy_sq_sum_s2[nname] / n2
                    parts2.append(f"{nname}={mse2:.4f} (n={n2})")
                else:
                    parts2.append(f"{nname}=NA (n=0)")
            print(f"[{task_name} | Client {cid}] Proxy MSE (Split-1, epoch {ep+1}): " + ", ".join(parts1))
            print(f"[{task_name} | Client {cid}] Proxy MSE (Split-2, epoch {ep+1}): " + ", ".join(parts2))
        else:
            print(f"[{task_name} | Client {cid}] Proxy MSE per label (epoch {ep+1}): node_names missing")

    ss_mask = {name: (g / max(num_updates, 1) > mask_thresh).float() for name, g in ss_grad.items()}
    d1_mask = {name: (g / max(num_updates, 1) > mask_thresh).float() for name, g in d1_grad.items()}
    d2_mask = {name: (g / max(num_updates, 1) > mask_thresh).float() for name, g in d2_grad.items()}
    return ss_mask, d1_mask, d2_mask

def evaluate(loader, FE, SS, BE, denoiser1, denoiser2, exo1, scm1, exo2, scm2,
             loss_fn, num_classes,
             proxy_table: ProxyTable = None,
             task_name: str = "",
             cid: int = -1):
    # FE.eval(); SS.eval(); BE.eval(); denoiser1.eval(); denoiser2.eval()
    # exo1.eval(); scm1.eval(); exo2.eval(); scm2.eval()
    total_loss, total_correct = 0.0, 0.0

    # NEW: dataset-level IoU accumulator
    iou_acc = IoUAccumulator(num_classes)

    proxy_sq_sum_s1 = defaultdict(float)
    proxy_n_s1      = defaultdict(int)
    proxy_sq_sum_s2 = defaultdict(float)
    proxy_n_s2      = defaultdict(int)
    with torch.no_grad():
        num_batches = 0
        for batch in loader:
            if len(batch) == 3:
                data, target, fnames = batch
                fnames = list(fnames)
            else:
                data, target = batch
                fnames = [str(i) for i in range(data.size(0))]
            data, target = data.to(DEVICE), target.long().to(DEVICE)

            # --- mirror the training forward: FE -> denoise1 -> SS -> denoise2 -> BE ---
            B = data.size(0)
            # fixed mid-step for stable eval; you can randomize if you prefer
            t_vec = torch.full((B,), max(1, T // 2), device=DEVICE, dtype=torch.long)

            # split-1
            x1 = FE(data)
            u_list_1, _, _ = exo1(x1)
            z_list_1 = scm1(u_list_1)
            zc1 = scm1.as_vector(z_list_1)

            # build x_t and reconstruct x1_hat
            eps1 = torch.randn_like(x1)
            sqrt_ab   = extract_into(alpha_cum.sqrt(), t_vec, x1.shape)
            sqrt_1mab = extract_into((1.0 - alpha_cum).sqrt(), t_vec, x1.shape)
            x_t1 = sqrt_ab * x1 + sqrt_1mab * eps1
            eps_pred1 = denoiser1(x_t1, t_vec, zc1)
            x1_hat = (x_t1 - sqrt_1mab * eps_pred1) / (sqrt_ab + 1e-8)

            # split-2
            x2 = SS(x1_hat)
            u_list_2, _, _ = exo2(x2)
            z_list_2 = scm2(u_list_2)
            zc2 = scm2.as_vector(z_list_2)

            eps2 = torch.randn_like(x2)
            x_t2 = sqrt_ab * x2 + sqrt_1mab * eps2  # reuse same t_vec stats
            eps_pred2 = denoiser2(x_t2, t_vec, zc2)
            x2_hat = (x_t2 - sqrt_1mab * eps_pred2) / (sqrt_ab + 1e-8)

            # segmentation on denoised features
            preds = BE(x2_hat)
            seg_loss = loss_fn(preds, target)

            # proxy losses (for logging/metric parity)
            proxy_loss1 = 0.0
            proxy_loss2 = 0.0
            if proxy_table is not None and proxy_table.enabled:
                targets_map = proxy_table.get_batch(fnames, device=DEVICE)
                if hasattr(scm1, "node_names"):
                    z1_scalar = nodewise_scalar(z_list_1)
                    z2_scalar = nodewise_scalar(z_list_2)
                    for i_node, nname in enumerate(scm1.node_names):
                        if nname in targets_map:
                            tgt  = targets_map[nname]
                            mask = torch.isfinite(tgt)
                            if mask.any():
                                err1 = (z1_scalar[i_node][mask] - tgt[mask]) ** 2
                                err2 = (z2_scalar[i_node][mask] - tgt[mask]) ** 2
                                proxy_loss1 += err1.mean()
                                proxy_loss2 += err2.mean()
                                proxy_sq_sum_s1[nname] += float(err1.sum().item())
                                proxy_sq_sum_s2[nname] += float(err2.sum().item())
                                proxy_n_s1[nname]      += int(mask.sum().item())
                                proxy_n_s2[nname]      += int(mask.sum().item())

            loss = seg_loss + 0.0 * (proxy_loss1 + proxy_loss2)  # keep evaluation loss simple; metrics printed below

            # metrics
            preds_lbl = torch.argmax(preds, dim=1)

            # NEW: accumulate dataset-level IoU
            iou_acc.update(preds_lbl, target, num_classes)

            total_correct += (preds_lbl == target).float().mean().item()
            total_loss    += float(loss.item())
            num_batches   += 1

    # NEW: compute per-class IoUs from accumulator
    per_cls = iou_acc.per_class_iou().cpu().numpy()
    iou_wbg = float(per_cls.mean())
    iou_nbg = float(per_cls[1:].mean()) if num_classes > 1 else float('nan')

    acc = 100. * total_correct / max(num_batches, 1)
    print(f"Eval Loss: {total_loss / max(num_batches,1):.4f} | Eval Acc: {acc:.2f}% | "
          f"IoU w/bg: {iou_wbg:.4f} | IoU no/bg: {iou_nbg:.4f}")
    if hasattr(scm1, "node_names") and scm1.node_names and proxy_table is not None and proxy_table.enabled:
        parts1, parts2 = [], []
        for nname in scm1.node_names:
            n1 = proxy_n_s1.get(nname, 0)
            n2 = proxy_n_s2.get(nname, 0)
            if n1 > 0:
                mse1 = proxy_sq_sum_s1[nname] / n1
                parts1.append(f"{nname}={mse1:.4f} (n={n1})")
            else:
                parts1.append(f"{nname}=NA (n=0)")
            if n2 > 0:
                mse2 = proxy_sq_sum_s2[nname] / n2
                parts2.append(f"{nname}={mse2:.4f} (n={n2})")
            else:
                parts2.append(f"{nname}=NA (n=0)")
        tag = f"{task_name} | Client {cid if cid!=-1 else '?'}"
        print(f"[{tag}] Proxy MSE (Split-1): " + ", ".join(parts1))
        print(f"[{tag}] Proxy MSE (Split-2): " + ", ".join(parts2))
    # return per-class arrays (with-bg full list, and 1..C-1 for no-bg)
    return total_loss / max(num_batches,1), acc, per_cls.tolist(), per_cls[1:].tolist()

# ----------------- Main -----------------
def main():
    task_info = {
        0: {"name": "Blastocyst", "num_classes": 5, "path": DATA_PATH, "dataset": EmbryoDataset,
            "proxy_paths": {
                "train": os.path.join(PROXY_DATA_PATH, "Blastocysts", "blastocysts_train.csv"),
                "val":   os.path.join(PROXY_DATA_PATH, "Blastocysts", "blastocysts_val.csv"),
                "test":  os.path.join(PROXY_DATA_PATH, "Blastocysts", "blastocysts_test.csv"),
            }},
        1: {"name": "HAM10K", "num_classes": 2, "path": DATA_PATH, "dataset": HAMDataset,
            "proxy_paths": {
                "train": os.path.join(PROXY_DATA_PATH, "HAM10K", "ham10k_train.csv"),
                "val":   os.path.join(PROXY_DATA_PATH, "HAM10K", "ham10k_val.csv"),
                "test":  os.path.join(PROXY_DATA_PATH, "HAM10K", "ham10k_test.csv"),
            }},
        2: {"name": "Fetal", "num_classes": 3, "path": DATA_PATH, "dataset": FHPsAOPMSBDataset,
            "proxy_paths": {
                "train": os.path.join(PROXY_DATA_PATH, "Fetus", "fetus_train.csv"),
                "val":   os.path.join(PROXY_DATA_PATH, "Fetus", "fetus_val.csv"),
                "test":  os.path.join(PROXY_DATA_PATH, "Fetus", "fetus_test.csv"),
            }},
        3: {"name": "Mosmed", "num_classes": 2, "path": DATA_PATH, "dataset": covidCTDataset,
            "proxy_paths": {
                "train": os.path.join(PROXY_DATA_PATH, "Mosmed", "mosmed_train.csv"),
                "val":   os.path.join(PROXY_DATA_PATH, "Mosmed", "mosmed_val.csv"),
                "test":  os.path.join(PROXY_DATA_PATH, "Mosmed", "mosmed_test.csv"),
            }},
        4: {"name": "Kvasir", "num_classes": 2, "path": DATA_PATH, "dataset": CVCDataset,
            "proxy_paths": {
                "train": os.path.join(PROXY_DATA_PATH, "Kvasir-SEG", "kvasir_train.csv"),
                "val":   os.path.join(PROXY_DATA_PATH, "Kvasir-SEG", "kvasir_val.csv"),
                "test":  os.path.join(PROXY_DATA_PATH, "Kvasir-SEG", "kvasir_test.csv"),
            }},
    }

    tr_tf = A.Compose([A.Resize(256, 256), A.Normalize(mean=[0]*3, std=[1]*3, max_pixel_value=255.0), ToTensorV2()])
    val_tf = test_tf = tr_tf

    global_SS = UNET_server(in_channels=32).to(DEVICE)
    best_loss = float('inf')

    # client-local modules across rounds
    client_exo1 = [None] * NUM_CLIENTS
    client_exo2 = [None] * NUM_CLIENTS
    client_scm1 = [None] * NUM_CLIENTS
    client_scm2 = [None] * NUM_CLIENTS
    client_den1_state = [None] * NUM_CLIENTS
    client_den2_state = [None] * NUM_CLIENTS
    client_FE = [None] * NUM_CLIENTS
    client_BE = [None] * NUM_CLIENTS

    for r in range(COMM_ROUNDS):
        print(f"\n[Comm Round {r+1}/{COMM_ROUNDS}]")

        val_proxies_round = []
        test_proxies_round = []

        local_SS = []
        client_masks_SS = []  # kept for compatibility; not used in aggregation
        client_FEs_round, client_BEs_round = [], []
        client_den1_round, client_den2_round = [], []
        val_nb_iou_round = []
        n_samples = [0] * NUM_CLIENTS  # <-- track per-client training set sizes this round

        for i in range(NUM_CLIENTS):
            task = task_info[i]
            task_name = task["name"]
            num_classes, path = task["num_classes"], task["path"]
            ds_class = task["dataset"]

            if client_FE[i] is None:
                FE = UNET_FE(in_channels=3).to(DEVICE)
                BE = UNET_BE(out_channels=num_classes).to(DEVICE)
                client_FE[i], client_BE[i] = FE, BE
            else:
                FE = client_FE[i].to(DEVICE)
                BE = client_BE[i].to(DEVICE)
            SS = UNET_server(in_channels=32).to(DEVICE)

            cfg = get_scm_config(task_name)
            nodes, parents, node_dims = cfg["nodes"], cfg["parents"], cfg["node_dims"]

            if client_exo1[i] is None:
                exo1 = ExogenousEncoder(in_channels=32, node_dims=node_dims, variational=True).to(DEVICE)
                exo2 = ExogenousEncoder(in_channels=32, node_dims=node_dims, variational=True).to(DEVICE)
                scm1 = NeuralSCM(parents=parents, node_dims=node_dims).to(DEVICE); scm1.node_names = nodes
                scm2 = NeuralSCM(parents=parents, node_dims=node_dims).to(DEVICE); scm2.node_names = nodes
                client_exo1[i], client_exo2[i] = exo1, exo2
                client_scm1[i], client_scm2[i] = scm1, scm2
            else:
                exo1 = client_exo1[i].to(DEVICE)
                exo2 = client_exo2[i].to(DEVICE)
                scm1 = client_scm1[i].to(DEVICE); scm1.node_names = nodes
                scm2 = client_scm2[i].to(DEVICE); scm2.node_names = nodes

            # attach z posterior logvars
            _ = ensure_z_logvars(scm1, node_dims)
            _ = ensure_z_logvars(scm2, node_dims)
            scm1.to(DEVICE); scm2.to(DEVICE)

            # build / cache a ZPrior (shared by both splits for this client)
            if not hasattr(scm1, "zprior"):
                y_dim = len(nodes)
                zprior = ZPrior(y_dim=y_dim, node_dims=node_dims).to(DEVICE)
                scm1.zprior = zprior
                scm2.zprior = zprior
            else:
                zprior = scm1.zprior
                scm2.zprior = zprior  # keep both in sync if objects are re-used

            cond_dim = sum(node_dims)
            denoiser1 = initialize_conditional_denoiser(input_dim=32, cond_dim=cond_dim, hidden_dim=128, device=DEVICE)
            denoiser2 = initialize_conditional_denoiser(input_dim=32, cond_dim=cond_dim, hidden_dim=128, device=DEVICE)
            if client_den1_state[i] is not None:
                denoiser1.load_state_dict(client_den1_state[i])
            if client_den2_state[i] is not None:
                denoiser2.load_state_dict(client_den2_state[i])

            SS.load_state_dict(global_SS.state_dict())

            opt = optim.AdamW([
                {"params": list(FE.parameters()) + list(BE.parameters()) + list(SS.parameters()), "lr": 1e-4},
                {"params": list(exo1.parameters()) + list(scm1.parameters()) +
                           list(exo2.parameters()) + list(scm2.parameters()), "lr": 5e-5},
                {"params": list(denoiser1.parameters()) + list(denoiser2.parameters()), "lr": 1e-4},
            ])

            # scheduler = CosineAnnealingWarmRestarts(opt, T_0=5, eta_min=1e-6)

            loss_fn = ComboLoss(num_classes)
            train_loader = get_loader(os.path.join(path, f"client{i + 1}/train_imgs"),
                                      os.path.join(path, f"client{i + 1}/train_masks"),
                                      ds_class, tr_tf, with_names=True, shuffle=True)
            val_loader = get_loader(os.path.join(path, f"client{i + 1}/val_imgs"),
                                    os.path.join(path, f"client{i + 1}/val_masks"),
                                    ds_class, val_tf, with_names=True, shuffle=False)

            # ---- count samples for fairness blending ----
            n_samples[i] = len(train_loader.dataset)

            ptrain = task["proxy_paths"].get("train") if task.get("proxy_paths") else None
            pval = task["proxy_paths"].get("val") if task.get("proxy_paths") else None
            ptest = task["proxy_paths"].get("test") if task.get("proxy_paths") else None

            alias_map_by_task = {
                "Blastocyst": {
                    "ZPThicknessVar": "ZP Thickness Variation",
                    "TEArea": "TE area",
                    "BubbleArea": "Bubble area",
                    "BlastocoelSym": "Blastocoel symmetry",
                },
                "HAM10K": {
                    "LesionSize": "Lesion Size",
                    "Compactness": "Compactness",
                    "Asymmetry": "Asymmetry",
                    "MeanRGB": "Mean RGB",
                },
                "Fetal": {
                    "HeadPerimeter": "Head perimeter",
                    "HeadMajorAxis": "Head Major axis",
                    "HeadArea": "Head Area",
                    "HeadCircularity": "Head circularity",
                },
                "Mosmed": {
                    "LesionSize": "Lesion size",
                    "Solidity": "Solidity",
                    "Orientation": "Orientation",
                    "Perimeter": "Perimeter",
                },
                "Kvasir": {
                    "MeanRGB": "Mean RGB",
                    "LesionSize": "Lesion size",
                    "Compactness": "Compactness",
                    "Asymmetry": "Asymmetry",
                },
            }
            alias_map = alias_map_by_task.get(task_name, {})

            train_proxy = ProxyTable(ptrain, nodes, zscore=True, alias_map=alias_map) if ptrain else ProxyTable(None, nodes)
            val_proxy   = ProxyTable(pval,   nodes, zscore=True, alias_map=alias_map) if pval   else ProxyTable(None, nodes)
            test_proxy  = ProxyTable(ptest,  nodes, zscore=True, alias_map=alias_map) if ptest  else ProxyTable(None, nodes)

            # ---- local training ----
            ss_mask, _, _ = train_local(
                train_loader, FE, SS, BE,
                denoiser1, denoiser2,
                exo1, scm1, exo2, scm2,
                opt, loss_fn, i + 1, num_classes,
                task_name, train_proxy,
                mask_thresh=1e-5
            )
            # scheduler.step(ep := 0)

            # local validation (per client)
            val_loss, val_acc, val_iou_wb, val_iou_nb = evaluate(
                val_loader, FE, SS, BE,
                denoiser1, denoiser2,
                exo1, scm1, exo2, scm2,
                loss_fn, num_classes,
                proxy_table=val_proxy, task_name=task_name, cid=i + 1
            )
            val_nb_iou_round.append(val_iou_nb)

            client_FE[i] = FE
            client_BE[i] = BE
            client_FEs_round.append(FE)
            client_BEs_round.append(BE)
            local_SS.append(SS)
            client_den1_round.append(denoiser1)
            client_den2_round.append(denoiser2)
            client_masks_SS.append(ss_mask)  # not used later
            val_proxies_round.append(val_proxy)
            test_proxies_round.append(test_proxy)
            client_den1_state[i] = denoiser1.state_dict()
            client_den2_state[i] = denoiser2.state_dict()

        # ----- IoU + size blended weighted fusion for SS (NO MASKS) -----
        # 1) scores from IoU (no-bg), NaN-robust
        client_scores = []
        for i in range(NUM_CLIENTS):
            nb = val_nb_iou_round[i]
            nb_clean = [x for x in nb if not (isinstance(x, float) and math.isnan(x))]
            score = float(sum(nb_clean) / max(1, len(nb_clean)))
            client_scores.append(score)

        # 2) smooth + floor + temperature
        client_weights = normalize_scores(client_scores)

        # 3) data-size fairness blend
        size_weights = [max(1, n) for n in n_samples]
        size_tot = sum(size_weights) or 1
        size_weights = [n / size_tot for n in size_weights]

        alpha = 0.5  # 0=only size, 1=only IoU; try 0.4–0.7
        combined = [alpha * cw + (1 - alpha) * sw for cw, sw in zip(client_weights, size_weights)]
        z = sum(combined) or 1.0
        combined = [c / z for c in combined]

        global_SS.load_state_dict(
            weighted_fusion(local_SS, combined, fallback_state=global_SS.state_dict())
        )

        # ----- Global validation (with global SS) -----
        total_val_loss, total_val_ious = 0.0, [0.0] * 5
        for i in range(NUM_CLIENTS):
            task = task_info[i]
            task_name = task["name"]
            num_classes, path = task["num_classes"], task["path"]
            ds_class = task["dataset"]

            val_loader = get_loader(
                os.path.join(path, f"client{i + 1}/val_imgs"),
                os.path.join(path, f"client{i + 1}/val_masks"),
                ds_class, val_tf, with_names=True, shuffle=False
            )
            loss_fn = ComboLoss(num_classes)

            val_loss, val_acc, val_iou_wb, val_iou_nb = evaluate(
                val_loader,
                client_FEs_round[i], global_SS, client_BEs_round[i],
                client_den1_round[i], client_den2_round[i],
                client_exo1[i], client_scm1[i],
                client_exo2[i], client_scm2[i],
                loss_fn, num_classes,
                proxy_table=val_proxies_round[i], task_name=task_name, cid=i + 1
            )
            print(f"[Client {i + 1} Global Validation] Loss: {val_loss:.4f} | Acc: {val_acc:.2f}% | "
                  f"IoU wb: {sum(val_iou_wb) / num_classes:.4f} | IoU nb: {sum(val_iou_nb) / (num_classes-1):.4f}")
            total_val_loss += val_loss
            for j in range(num_classes): total_val_ious[j] += val_iou_wb[j]

        avg_val_loss = total_val_loss / NUM_CLIENTS
        avg_ious = [v / NUM_CLIENTS for v in total_val_ious]
        print(f"[Global Validation] Loss: {avg_val_loss:.4f} | IoU w/bg: {sum(avg_ious) / 5:.4f} | IoU no/bg: {sum(avg_ious[1:]) / 4:.4f}")

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(global_SS.state_dict(), "BestModels/best_SS.pth")
            for i in range(NUM_CLIENTS):
                torch.save(client_den1_state[i], f"BestModels/denoiser1_client{i+1}.pth")
                torch.save(client_den2_state[i], f"BestModels/denoiser2_client{i+1}.pth")
                torch.save(client_FEs_round[i].state_dict(), f"BestModels/best_FE_client{i + 1}.pth")
                torch.save(client_BEs_round[i].state_dict(), f"BestModels/best_BE_client{i + 1}.pth")
            print("[Best Global SS and Client Models Saved]")

        # ----- Testing -----
        for i in range(NUM_CLIENTS):
            task = task_info[i]
            task_name = task["name"]
            num_classes, path = task["num_classes"], task["path"]
            ds_class = task["dataset"]

            test_loader = get_loader(
                os.path.join(path, f"client{i + 1}/test_imgs"),
                os.path.join(path, f"client{i + 1}/test_masks"),
                ds_class, test_tf, with_names=True, shuffle=False
            )
            loss_fn = ComboLoss(num_classes)

            test_loss, test_acc, test_iou_wb, test_iou_nb = evaluate(
                test_loader,
                client_FEs_round[i], global_SS, client_BEs_round[i],
                client_den1_round[i], client_den2_round[i],
                client_exo1[i], client_scm1[i],
                client_exo2[i], client_scm2[i],
                loss_fn, num_classes,
                proxy_table=test_proxies_round[i], task_name=task_name, cid=i + 1
            )
            print(f"[Client {i + 1} Testing] Loss: {test_loss:.4f} | Acc: {test_acc:.2f}% | "
                  f"IoU wb: {sum(test_iou_wb) / num_classes:.4f} | IoU nb: {sum(test_iou_nb) / (num_classes-1):.4f}")
            test_iou_wbg_all[i].append(sum(test_iou_wb) / num_classes)
            test_iou_nbg_all[i].append(sum(test_iou_nb) / (num_classes - 1))

        plot_iou_curves(r + 1)

if __name__ == "__main__":
    main()
