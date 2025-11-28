# full corrected code (copy-paste)
import os
CUDA_LAUNCH_BLOCKING = 1
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
from torch.autograd import Function
from models.clientmodel_FE import UNET_FE
from models.servermodel import UNET_server
from models.clientmodel_BE import UNET_BE
from dataset import EmbryoDataset, HAMDataset, CVCDataset, covidCTDataset, FHPsAOPMSBDataset
from reverse_diff_causal import initialize_conditional_denoiser
from models.exogenous_encoder import ExogenousEncoder
from models.neural_scm import NeuralSCM
from scm_configs import get_scm_config
from proxy_tables import ProxyTable
from dataset_wrappers import WithFilenames
from models.z_prior import ZPrior
import sys
output_file = "/lustre06/project/6008975/csj5/causalenv/mucald_final_unet.txt"
sys.stdout = open(output_file, "w")
from torch.utils.data import Subset
import copy
import torchvision

# --- NEW imports for saving predictions and metrics ---
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.metrics import structural_similarity as sk_ssim
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
import math
try:
    import lpips
    LPIPS_AVAILABLE = True
except Exception:
    LPIPS_AVAILABLE = False
    # Will skip LPIPS if not installed.

# -----------------------------------------

DEVICE = "cuda"
NUM_CLIENTS = 5
LOCAL_EPOCHS = 5
COMM_ROUNDS = 24

LAMBDA_PROXY = 1.5
LAMBDA_DIFF = 0.5
LAMBDA_KLU = 1e-4
LAMBDA_KLZ = 1e-4
MU = 0.001  # fedprox coefficient
WARMUP_EPOCHS = 2
RAMP_EPOCHS = 3

LAMBDA_PROXY_TARGET = 0.40
LAMBDA_DIFF_TARGET = 0.40
LAMBDA_KLU_TARGET = 1.0e-4
LAMBDA_KLZ_TARGET = 1.0e-4

LAMBDA_ADV = 0.10
LAMBDA_DOM = 0.1

# --- Per-client adversarial scaling (cid starts at 1) ---
PER_CLIENT_ADV_SCALE = {
    1: 1.0,   # Blastocyst
    2: 0.4,   # HAM10K  -> reduce adversarial pressure
    3: 1.0,   # Fetal
    4: 1.8,   # MosMed  -> increase alignment pressure
    5: 1.0    # Kvasir
}


class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambd=1.0):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambd, None


def grad_reverse(x, lambd=1.0):
    return GradReverse.apply(x, lambd)


class DomainDiscriminator(nn.Module):
    def __init__(self, in_dim=32, num_domains=NUM_CLIENTS, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, num_domains)
        )

    def forward(self, x):
        return self.net(x)


def get_epoch_lambdas(ep: int):
    if ep < WARMUP_EPOCHS:
        return 0.0, 0.0, 0.0, 0.0
    if ep < WARMUP_EPOCHS + RAMP_EPOCHS:
        factor = (ep - WARMUP_EPOCHS + 1) / float(RAMP_EPOCHS)
        factor = max(0.0, min(1.0, factor))
        return (
            LAMBDA_PROXY_TARGET * factor,
            LAMBDA_DIFF_TARGET * factor,
            LAMBDA_KLU_TARGET * factor,
            LAMBDA_KLZ_TARGET * factor,
        )
    return (
        LAMBDA_PROXY_TARGET,
        LAMBDA_DIFF_TARGET,
        LAMBDA_KLU_TARGET,
        LAMBDA_KLZ_TARGET,
    )


T = 400


def cosine_beta_schedule(T: int, s: float = 0.008, device: torch.device = None):
    device = device or torch.device("cpu")
    t = torch.linspace(0, T, T + 1, device=device) / T
    f = torch.cos(((t + s) / (1 + s)) * torch.pi / 2) ** 2
    f = f / f[0]
    betas = 1 - (f[1:] / f[:-1])
    return betas.clamp(1e-8, 0.999)


beta = cosine_beta_schedule(T, device=torch.device(DEVICE))
alpha = 1.0 - beta
alpha_cum = torch.cumprod(alpha, dim=0).to(DEVICE)

PROXY_DATA_PATH = "/lustre06/project/6008975/csj5/causalenv/Proxy_variables_dir/Final/"
DATA_PATH = "/lustre06/project/6008975/csj5/MTS2"

os.makedirs("BestModels", exist_ok=True)
os.makedirs("Outputs", exist_ok=True)

test_iou_wbg_all = {i: [] for i in range(NUM_CLIENTS)}
test_iou_nbg_all = {i: [] for i in range(NUM_CLIENTS)}
#test_proxy_mse_all = {i: [] for i in range(NUM_CLIENTS)}
#test_dice_all = {i: [] for i in range(NUM_CLIENTS)}


def extract_into(arr_1d, timesteps, x_shape):
    out = arr_1d.gather(0, timesteps.clamp_(0, arr_1d.shape[0] - 1))
    return out.view(-1, *([1] * (len(x_shape) - 1)))


@torch.no_grad()
def predict_x0_from_eps(x_t, t_vec, eps_pred, alpha_cum):
    sqrt_ab = extract_into(alpha_cum.sqrt(), t_vec, x_t.shape)
    sqrt_1mab = extract_into((1.0 - alpha_cum).sqrt(), t_vec, x_t.shape)
    x0_hat = (x_t - sqrt_1mab * eps_pred) / (sqrt_ab + 1e-8)
    return x0_hat


def diffusion_noise_loss(denoiser, x0, t_vec, z_cond, alpha_cum, want_grad_x0=False):
    eps = torch.randn_like(x0)
    sqrt_ab = extract_into(alpha_cum.sqrt(), t_vec, x0.shape)
    sqrt_1mab = extract_into((1.0 - alpha_cum).sqrt(), t_vec, x0.shape)
    x_t_detached = sqrt_ab * x0.detach() + sqrt_1mab * eps
    eps_pred = denoiser(x_t_detached, t_vec, z_cond)
    loss = F.mse_loss(eps_pred, eps)
    if not want_grad_x0:
        with torch.no_grad():
            x0_hat = (x_t_detached - sqrt_1mab * eps_pred) / (sqrt_ab + 1e-8)
        return loss, x0_hat
    x_t_for_seg = sqrt_ab * x0 + sqrt_1mab * eps
    eps_pred_ng = eps_pred.detach()
    x0_hat_for_seg = (x_t_for_seg - sqrt_1mab * eps_pred_ng) / (sqrt_ab + 1e-8)
    return loss, x0_hat_for_seg


def kl_standard_normal(mu, logvar):
    return 0.5 * torch.mean(torch.sum(torch.exp(logvar) + mu ** 2 - 1.0 - logvar, dim=1))


def kl_gaussians(mu_q, logv_q, mu_p, logv_p, weight=None):
    device = mu_q.device
    mu_p, logv_p = mu_p.to(device), logv_p.to(device)
    var_q = logv_q.exp();
    var_p = logv_p.exp()
    per = 0.5 * torch.sum(logv_p - logv_q + (var_q + (mu_q - mu_p) ** 2) / (var_p + 1e-8) - 1.0, dim=1)
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
            nn.Parameter(torch.full((d,), -2.0, device=dev))
            for d in node_dims
        ])
    return scm_module.z_logvar_params


def normalize_features(features):
    mean = features.mean(dim=(2, 3), keepdim=True)
    std = features.std(dim=(2, 3), keepdim=True)
    return (features - mean) / (std + 1e-8)


def get_loader(img_dir, mask_dir, dataset_class, transform,
               batch_size=1, num_workers=1, with_names=True,
               shuffle=True, fraction=1.0):
    base = dataset_class(img_dir, mask_dir, transform=transform)
    ds = WithFilenames(base, img_dir) if with_names else base
    # take only first fraction of dataset
    if fraction < 1.0:
        n = int(len(ds) * fraction)
        indices = list(range(n))
        ds = Subset(ds, indices)
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
        return self.dice(logits, targets) + 0.5 * self.ce(logits, targets)


def make_grad_tracker(model):
    return {name: torch.zeros_like(p, device=p.device)
            for name, p in model.named_parameters() if p.requires_grad}


# -------------- Metrics helpers (segmentation & reconstruction) ----------------

def _flatten_for_fid(tensor):  # tensor: (N, C, H, W) -> (N, D)
    N = tensor.shape[0]
    return tensor.reshape(N, -1).astype(np.float64)


def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    # compute Frechet distance between two Gaussians (numpy arrays)
    from scipy.linalg import sqrtm
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    diff = mu1 - mu2
    covmean, _ = None, None
    try:
        covmean = sqrtm(sigma1.dot(sigma2))
    except Exception:
        covmean = None
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    tr_covmean = np.trace(covmean)
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2.0 * tr_covmean)


def compute_fid_from_arrays(arr1, arr2):
    # arrX: (N, D) numpy arrays
    mu1 = np.mean(arr1, axis=0)
    mu2 = np.mean(arr2, axis=0)
    sigma1 = np.cov(arr1, rowvar=False)
    sigma2 = np.cov(arr2, rowvar=False)
    return frechet_distance(mu1, sigma1, mu2, sigma2)


def compute_recon_metrics(clean_list, hat_list):
    """
    clean_list / hat_list: Python lists of numpy arrays with shape (C,H,W) per sample
    returns dict: mse, psnr, ssim, fid
    """
    if len(clean_list) == 0:
        return {"mse": None, "psnr": None, "ssim": None, "fid": None, "lpips": None}

    clean = np.stack(clean_list, axis=0)  # (N, C, H, W)
    hat = np.stack(hat_list, axis=0)

    # per-sample MSE
    mse_per_sample = np.mean((clean - hat) ** 2, axis=(1, 2, 3))
    mse = float(mse_per_sample.mean())

    # PSNR: use dynamic range per-sample (max - min) fallback to 1.0
    psnrs = []
    ssims = []
    for i in range(clean.shape[0]):
        c = clean[i]
        h = hat[i]
        data_range = float(c.max() - c.min())
        if data_range <= 0:
            data_range = 1.0
        # compute PSNR on flattened channels
        psnrs.append(sk_psnr(c, h, data_range=data_range))

        # SSIM: compute per-channel and average (sk_ssim expects 2D or multichannel last dim)
        # We compute per-channel SSIM and average:
        ch_ssims = []
        for ch in range(c.shape[0]):
            try:
                ch_ssim = sk_ssim(c[ch], h[ch], data_range=(c[ch].max() - c[ch].min()) if (c[ch].max() - c[ch].min()) != 0 else 1.0)
            except Exception:
                ch_ssim = 0.0
            ch_ssims.append(ch_ssim)
        ssims.append(float(np.mean(ch_ssims)))
    psnr = float(np.mean(psnrs))
    ssim = float(np.mean(ssims))

    # FID on flattened latent vectors
    arr1 = _flatten_for_fid(clean)
    arr2 = _flatten_for_fid(hat)
    try:
        fid = compute_fid_from_arrays(arr1, arr2)
    except Exception:
        fid = None

    # LPIPS (only if available and if channels==3 and small H/W; otherwise skip)
    lpips_score = None
    if LPIPS_AVAILABLE:
        try:
            loss_fn = lpips.LPIPS(net='alex').to("cpu")
            # LPIPS expects tensors in range [-1,1], 3-channel. We attempt conversion per sample if possible.
            if clean.shape[1] == 3:
                scores = []
                for i in range(clean.shape[0]):
                    cimg = torch.tensor(clean[i]).unsqueeze(0)  # (1,C,H,W)
                    himg = torch.tensor(hat[i]).unsqueeze(0)
                    # move to [-1,1] approx by scaling based on min/max
                    def norm01(t):
                        t = t - float(t.min())
                        denom = float(t.max() - t.min())
                        if denom <= 0:
                            denom = 1.0
                        return 2.0 * (t / denom) - 1.0
                    c01 = norm01(cimg)
                    h01 = norm01(himg)
                    with torch.no_grad():
                        s = loss_fn(c01, h01).item()
                    scores.append(s)
                lpips_score = float(np.mean(scores))
        except Exception:
            lpips_score = None

    return {"mse": mse, "psnr": psnr, "ssim": ssim, "fid": fid, "lpips": lpips_score}


def _mask_to_surface(mask):
    """
    Return boolean surface pixels for a binary mask (2D numpy array)
    """
    # mask resized at 256x256 probably, assume connectivity 1
    if mask.sum() == 0:
        return np.zeros_like(mask, dtype=bool)
    eroded = ndi.binary_erosion(mask, structure=np.ones((3, 3)))
    surface = mask.astype(bool) & (~eroded)
    return surface


def _surface_distances(mask_gt, mask_pred):
    """
    Get distance array from surface points of mask_gt to nearest surface point in mask_pred.
    Both inputs are binary 2D numpy arrays in the same grid.
    Returns distances (1D array) from each GT surface point to nearest pred surface.
    """
    if mask_gt.sum() == 0:
        return np.array([])
    # distance transform on predicted mask's complement (distance to nearest foreground)
    pred_dist = ndi.distance_transform_edt(~mask_pred)
    gt_surface = _mask_to_surface(mask_gt)
    distances = pred_dist[gt_surface]
    return distances


def hd95_assd_for_pair(mask_gt, mask_pred):
    """
    Compute HD95 and ASSD between two binary masks (2D numpy arrays).
    Returns hd95, assd (float or np.nan if not computable)
    """
    if mask_gt.sum() == 0 and mask_pred.sum() == 0:
        return 0.0, 0.0

    # distances from gt surface to pred surface
    d_gt_to_pred = _surface_distances(mask_gt, mask_pred)
    d_pred_to_gt = _surface_distances(mask_pred, mask_gt)

    if d_gt_to_pred.size == 0:
        hd1 = 0.0
        asd1 = 0.0
    else:
        hd1 = np.percentile(d_gt_to_pred, 95)
        asd1 = d_gt_to_pred.mean()

    if d_pred_to_gt.size == 0:
        hd2 = 0.0
        asd2 = 0.0
    else:
        hd2 = np.percentile(d_pred_to_gt, 95)
        asd2 = d_pred_to_gt.mean()

    hd95 = max(hd1, hd2)
    assd = 0.5 * (asd1 + asd2)
    return float(hd95), float(assd)


def compute_segmentation_metrics_all(preds_lbl_np, target_np, num_classes):
    """
    preds_lbl_np, target_np: numpy arrays (N, H, W) int labels
    returns dict of per-class metrics arrays:
       iou: length num_classes
       dice: ...
       precision/recall/f1: ...
       hd95: ...
       assd: ...
    """
    N = preds_lbl_np.shape[0]
    iou_per_class = np.zeros((num_classes,), dtype=float)
    dice_per_class = np.zeros((num_classes,), dtype=float)
    prec_per_class = np.zeros((num_classes,), dtype=float)
    rec_per_class = np.zeros((num_classes,), dtype=float)
    f1_per_class = np.zeros((num_classes,), dtype=float)
    hd95_per_class = np.zeros((num_classes,), dtype=float)
    assd_per_class = np.zeros((num_classes,), dtype=float)

    # accumulate counts for precision/recall
    tp = np.zeros((num_classes,), dtype=float)
    fp = np.zeros((num_classes,), dtype=float)
    fn = np.zeros((num_classes,), dtype=float)

    # compute IoU using flatten (sklearn jaccard expects 1D)
    try:
        ious_all = jaccard_score(
            target_np.flatten(),
            preds_lbl_np.flatten(),
            average=None,
            labels=list(range(num_classes)),
            zero_division=0
        )
    except Exception:
        # fallback compute per class manually
        ious_all = np.zeros((num_classes,), dtype=float)
        for c in range(num_classes):
            inter = np.logical_and(target_np == c, preds_lbl_np == c).sum()
            union = np.logical_or(target_np == c, preds_lbl_np == c).sum()
            ious_all[c] = inter / union if union > 0 else 0.0

    iou_per_class[:] = ious_all

    # per-class counts, dice, precision etc.
    for c in range(num_classes):
        for i in range(N):
            gt_mask = (target_np[i] == c)
            pred_mask = (preds_lbl_np[i] == c)
            inter = float((gt_mask & pred_mask).sum())
            p_area = float(pred_mask.sum())
            g_area = float(gt_mask.sum())
            tp[c] += inter
            fp[c] += max(0.0, p_area - inter)
            fn[c] += max(0.0, g_area - inter)

            # HD95 / ASSD per-sample -> accumulate average
            hd95_val, assd_val = hd95_assd_for_pair(gt_mask.astype(bool), pred_mask.astype(bool))
            # accumulate sums; we'll later average over images where class present in GT OR pred
            hd95_per_class[c] += hd95_val
            assd_per_class[c] += assd_val

    # finalize precision/recall/f1 and dice
    for c in range(num_classes):
        precision = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) > 0 else 0.0
        recall = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        prec_per_class[c] = precision
        rec_per_class[c] = recall
        f1_per_class[c] = f1
        # Dice from tp, fp, fn: Dice = 2TP / (2TP + FP + FN)
        denom = (2.0 * tp[c] + fp[c] + fn[c])
        dice_per_class[c] = (2.0 * tp[c] / denom) if denom > 0 else 0.0

        # hd95_per_class and assd_per_class are sums across images; average by N
        hd95_per_class[c] = float(hd95_per_class[c] / max(1, N))
        assd_per_class[c] = float(assd_per_class[c] / max(1, N))

    return {
        "iou": iou_per_class,
        "dice": dice_per_class,
        "precision": prec_per_class,
        "recall": rec_per_class,
        "f1": f1_per_class,
        "hd95": hd95_per_class,
        "assd": assd_per_class
    }


# ------------------------------------------------------------------------------------


def _try_resolve_original_path(loader, fname):
    # Try several fallbacks to find original image path given a filename from the dataset.
    # Returns a path if exists, else None.
    if os.path.isabs(fname) and os.path.exists(fname):
        return fname
    # If fname is already a path relative to cwd
    if os.path.exists(fname):
        return fname

    ds = loader.dataset
    # if Subset wrapper, unwrap
    if hasattr(ds, "dataset"):
        ds = ds.dataset

    candidate_dirs = []
    # common attribute names that might hold the image base dir
    for attr in ("root", "img_dir", "image_dir", "base_dir", "data_dir", "images_dir"):
        base = getattr(ds, attr, None)
        if base:
            candidate_dirs.append(base)

    # also try the dataset initialization argument passed into our WithFilenames wrapper (we set it in get_loader)
    try:
        if hasattr(loader.dataset, "base_dir") and loader.dataset.base_dir:
            candidate_dirs.append(loader.dataset.base_dir)
    except Exception:
        pass

    for d in candidate_dirs:
        # try joining
        candidate = os.path.join(d, fname)
        if os.path.exists(candidate):
            return candidate
        # maybe fname is basename, try basename join
        candidate2 = os.path.join(d, os.path.basename(fname))
        if os.path.exists(candidate2):
            return candidate2

    # still not found
    return None


# ------------------- NEW: save_test_images helper (corrected + resize to original) -------------------
@torch.no_grad()
def save_test_images(loader, FE, SS, BE, denoiser1, denoiser2, exo1, scm1, exo2, scm2,
                     num_classes, out_dir_root="Outputs/unet_mucald_test_preds", cid=1):
    """
    Saves predicted masks for each image in loader.
    Overwrites existing files (so each communication round replaces them).
    Filenames: if input is "1.png" -> saved as "1_pred.png" inside
      Outputs/test_preds/client{cid}/
    Resizes masks to original image size when the original can be located.
    """
    out_dir = os.path.join(out_dir_root, f"client{cid}")
    os.makedirs(out_dir, exist_ok=True)

    # simple deterministic middle timestep
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                data, target, fnames = batch
                fnames = list(fnames)
            else:
                data, target = batch
                fnames = [str(i) for i in range(data.size(0))]
            data = data.to(DEVICE)
            B = data.size(0)
            t_vec = torch.full((B,), max(1, T // 2), device=DEVICE, dtype=torch.long)

            x1 = FE(data)
            u_list_1, _, _ = exo1(x1)
            z_list_1 = scm1(u_list_1)
            zc1 = scm1.as_vector(z_list_1)

            eps1 = torch.randn_like(x1)
            sqrt_ab = extract_into(alpha_cum.sqrt(), t_vec, x1.shape)
            sqrt_1mab = extract_into((1.0 - alpha_cum).sqrt(), t_vec, x1.shape)
            x_t1 = sqrt_ab * x1 + sqrt_1mab * eps1
            eps_pred1 = denoiser1(x_t1, t_vec, zc1)
            x1_hat = (x_t1 - sqrt_1mab * eps_pred1) / (sqrt_ab + 1e-8)

            x2 = SS(x1_hat)
            u_list_2, _, _ = exo2(x2)
            z_list_2 = scm2(u_list_2)
            zc2 = scm2.as_vector(z_list_2)

            eps2 = torch.randn_like(x2)
            x_t2 = sqrt_ab * x2 + sqrt_1mab * eps2
            eps_pred2 = denoiser2(x_t2, t_vec, zc2)
            x2_hat = (x_t2 - sqrt_1mab * eps_pred2) / (sqrt_ab + 1e-8)

            preds = BE(x2_hat)
            preds_lbl = torch.argmax(preds, dim=1).cpu().numpy()  # (B, H, W)

            for b in range(preds_lbl.shape[0]):
                fname = fnames[b]
                base = os.path.basename(fname)
                name_noext = os.path.splitext(base)[0]
                out_name = f"{name_noext}_pred.png"
                out_path = os.path.join(out_dir, out_name)

                arr = preds_lbl[b].astype(np.uint8)
                # map class indices to 0-255 for visualization (spread across range)
                if num_classes > 1:
                    scale = 255 // max(1, (num_classes - 1))
                    vis = (arr * scale).astype(np.uint8)
                else:
                    vis = (arr * 255).astype(np.uint8)

                # Try to resize to original image size if possible
                orig_path = _try_resolve_original_path(loader, fname)
                img = Image.fromarray(vis)
                img = img.convert("L")  # single-channel

                if orig_path is not None:
                    try:
                        orig_img = Image.open(orig_path)
                        orig_size = orig_img.size  # (width, height)
                        # use NEAREST to preserve mask labels precisely
                        img = img.resize(orig_size, resample=Image.NEAREST)
                    except Exception as e:
                        # fallback: keep transform-size mask (already at 256x256)
                        print(f"[save_test_images] Warning: couldn't open/resize original {orig_path}: {e}")

                img.save(out_path)
# --------------------------------------------------------------------

def causal_invariant_fusion(
    models, proxy_mse_scores, client_sizes=None, val_losses=None,
    eps=1e-8, min_fraction=0.25, blend_equal=0.5
):
    ref_sd = models[0].state_dict()
    model_sds = [m.state_dict() for m in models]
    out = {}
    n_clients = len(models)

    proxy_mse_scores = [max(mse, 1e-3) for mse in proxy_mse_scores]

    if client_sizes is None:
        client_sizes = [1.0] * n_clients
    if val_losses is None:
        val_losses = [1.0] * n_clients

    causal_scores = [
        (1.0 / (mse + eps)) * (size / (val_loss + eps))
        for mse, size, val_loss in zip(proxy_mse_scores, client_sizes, val_losses)
    ]

    total = sum(causal_scores) + eps
    weights = [cs / total for cs in causal_scores]

    min_w = min_fraction / n_clients
    weights = [w + min_w for w in weights]
    total = sum(weights)
    weights = [w / total for w in weights]

    if blend_equal > 0:
        equal_w = [1.0 / n_clients] * n_clients
        weights = [
            (1 - blend_equal) * w + blend_equal * ew
            for w, ew in zip(weights, equal_w)
        ]

    print(f"Aggregation weights: {weights}")

    for key in ref_sd.keys():
        out[key] = sum(sd[key] * w for sd, w in zip(model_sds, weights))

    return out
    
    
# ----------- NEW: Equal Weight Fusion -------------
def equal_weight_fusion(models):
    ref_sd = models[0].state_dict()
    model_sds = [m.state_dict() for m in models]
    n_clients = len(models)
    weights = [1.0 / n_clients] * n_clients
    print(f"Equal aggregation weights: {weights}")
    out = {}
    for key in ref_sd.keys():
        out[key] = sum(sd[key] * w for sd, w in zip(model_sds, weights))
    return out



def get_transforms(task_name):
    if task_name == "Blastocyst":
        return A.Compose([
            A.Resize(256, 256),
            A.Normalize(mean=[0]*3, std=[1]*3, max_pixel_value=255.0),
            ToTensorV2()
        ])
    elif task_name == "HAM10K":
        return A.Compose([
            A.Resize(256, 256),
            A.Normalize(mean=[0]*3, std=[1]*3, max_pixel_value=255.0),
            ToTensorV2()
        ])
    elif task_name == "Fetal":
        return A.Compose([
            A.Resize(256, 256),
            A.Normalize(mean=[0]*3, std=[1]*3, max_pixel_value=255.0),
            ToTensorV2()
        ])
    elif task_name == "Mosmed":
        return A.Compose([
            A.Resize(256, 256),
            A.Normalize(mean=[0]*3, std=[1]*3, max_pixel_value=255.0),
            ToTensorV2()
        ])
    elif task_name == "Kvasir":
        return A.Compose([
            A.Resize(256, 256),
            A.Normalize(mean=[0]*3, std=[1]*3, max_pixel_value=255.0),
            ToTensorV2()
        ])
    return A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=[0]*3, std=[1]*3, max_pixel_value=255.0),
        ToTensorV2()
    ])



def plot_curves(round_num):
    dataset_names = ["Blastocysts", "HAM10K", "Fetus", "MosMed", "Kvasir"]
    rounds = list(range(1, round_num + 1))
    plt.figure(figsize=(10, 5))
    for i in range(NUM_CLIENTS):
        plt.plot(rounds, test_iou_wbg_all[i], label=dataset_names[i])
    plt.xlabel("Communication Round")
    plt.ylabel("IoU w/b")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Outputs/mucald_unet_final_wb.png")
    plt.close()

    plt.figure(figsize=(10, 5))
    for i in range(NUM_CLIENTS):
        plt.plot(rounds, test_iou_nbg_all[i], label=dataset_names[i])
    plt.xlabel("Communication Round")
    plt.ylabel("IoU n/b")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Outputs/mucald_unet_final_nb.png")
    plt.close()


def train_local(loader, FE, SS, BE,
                denoiser1, denoiser2,
                exo1, scm1, exo2, scm2,
                dom_disc1, dom_disc2,
                opt, dom_opt, loss_fn, cid, num_classes,
                task_name, proxy_table: ProxyTable,
                fallback_state=None, mask_thresh=1e-5, client_size=None,max_size=None):

    FE.train(); SS.train(); BE.train()
    denoiser1.train(); denoiser2.train()
    exo1.train(); scm1.train()
    exo2.train(); scm2.train()
    dom_disc1.train(); dom_disc2.train()

    ss_grad = make_grad_tracker(SS)
    d1_grad = make_grad_tracker(denoiser1)
    d2_grad = make_grad_tracker(denoiser2)
    num_updates = 0
    
    if client_size is not None and max_size is not None:
        seg_weight = max_size / float(client_size)
        adv_weight = client_size / float(max_size)
    else:
        seg_weight = 1.0
        adv_weight = 1.0
    

    for ep in range(LOCAL_EPOCHS):
        lam_proxy, lam_diff, lam_klu, lam_klz = get_epoch_lambdas(ep)

        if task_name in ["Blastocyst", "Fetal", "Mosmed"]:
            lam_proxy *= 0.25

        if ep < WARMUP_EPOCHS + RAMP_EPOCHS:
            lam_adv = 0.0
        else:
            lam_adv = LAMBDA_ADV

        # --- apply per-client adversarial scaling ---
        scale = PER_CLIENT_ADV_SCALE.get(cid, 1.0)
        lam_adv = lam_adv * scale
        # -------------------------------------------------

        print(f"[Sched] ep={ep+1}/{LOCAL_EPOCHS} | ?_proxy={lam_proxy:.3g}, ?_diff={lam_diff:.3g}, ?_KLu={lam_klu:.3g}, ?_KLz={lam_klz:.3g}")

        tloss = 0.0
        tcorrect = 0.0
        num_batches = 0
        iou_c = [0.0] * num_classes

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

            # ---------- Forward FE ? SS ----------
            x1 = FE(data)
            u_list_1, u_mu_1, u_lv_1 = exo1(x1)
            z_list_1 = scm1(u_list_1)
            z_causal_1 = scm1.as_vector(z_list_1)
            diff_loss1, x1_hat = diffusion_noise_loss(denoiser1, x1, t_vec, z_causal_1.detach(), alpha_cum, want_grad_x0=True)

            x2 = SS(x1_hat)
            u_list_2, u_mu_2, u_lv_2 = exo2(x2)
            z_list_2 = scm2(u_list_2)
            z_causal_2 = scm2.as_vector(z_list_2)
            diff_loss2, x2_hat = diffusion_noise_loss(denoiser2, x2, t_vec, z_causal_2.detach(), alpha_cum, want_grad_x0=True)

            preds = BE(x2_hat)
            seg_loss = loss_fn(preds, target)

            # ---------- Proxy losses ----------
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

                y_list, mask_list = [], []
                for i_node, nname in enumerate(scm1.node_names):
                    if nname in targets_map:
                        tgt = targets_map[nname]
                        mask = torch.isfinite(tgt)
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
                y_input = torch.stack(y_list, dim=1)
                y_mask_per_node = torch.stack(mask_list, dim=1)

            # ---------- KL losses ----------
            kl_u1 = sum(kl_standard_normal(mu, lv) for mu, lv in zip(u_mu_1, u_lv_1))
            kl_u2 = sum(kl_standard_normal(mu, lv) for mu, lv in zip(u_mu_2, u_lv_2))
            kl_u = kl_u1 + kl_u2

            kl_z = torch.tensor(0.0, device=DEVICE)
            if y_input is not None:
                mu_p_list, lv_p_list = scm1.zprior(y_input)
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

            # ---------- Domain adversarial + discriminator losses ----------
            domain_labels = torch.full((B,), cid-1, dtype=torch.long, device=DEVICE)

            # Adversarial branch (SS/FE training via GRL)
            adv_logits1 = dom_disc1(grad_reverse(x1_hat))
            adv_loss1 = F.cross_entropy(adv_logits1, domain_labels)
            adv_logits2 = dom_disc2(grad_reverse(x2_hat))
            adv_loss2 = F.cross_entropy(adv_logits2, domain_labels)

            # Discriminator branch (train discriminator directly)
            dom_logits1 = dom_disc1(x1_hat.detach())
            dom_loss1 = F.cross_entropy(dom_logits1, domain_labels)

            dom_logits2 = dom_disc2(x2_hat.detach())
            dom_loss2 = F.cross_entropy(dom_logits2, domain_labels)
   
            
            # ---------- Total loss for FE/SS/BE/SCM ----------
            loss = (
                    seg_weight *seg_loss
                    + lam_proxy * (proxy_loss1 + proxy_loss2)
                    + lam_diff * (diff_loss1 + diff_loss2)
                    + lam_klu * kl_u
                    + lam_klz * kl_z
                    + (adv_weight * lam_adv) * (adv_loss1 + adv_loss2)
            )


            prox_term = 0.0
            if fallback_state is not None:  # fallback_state is global params
                global_params = fallback_state
                for name, p in SS.named_parameters():
                    if p.requires_grad:
                        prox_term += ((p - global_params[name].to(p.device)) ** 2).sum()
            prox_term = 0.5 * MU * prox_term
            loss = loss + prox_term

           # ---------- Backward ----------
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

            # ---------- Backward: discriminator update ----------
            dom_opt.zero_grad()
            (dom_loss1 + dom_loss2).backward()
            dom_opt.step()

            num_updates += 1
            num_batches += 1

            # ---------- Metrics ----------
            preds_lbl = torch.argmax(preds, dim=1)
            tcorrect += (preds_lbl == target).float().mean().item()
            tloss += float(loss.item())

            ious = jaccard_score(
                target.detach().cpu().flatten(),
                preds_lbl.detach().cpu().flatten(),
                average=None,
                labels=list(range(num_classes)),
                zero_division=0
            )
            for i_c in range(num_classes):
                iou_c[i_c] += float(ious[i_c])

        acc = 100.0 * (tcorrect / max(num_batches, 1))
        avg_iou = [v / max(num_batches, 1) for v in iou_c]

        print(f"Client {cid} | Epoch {ep + 1} | "
              f"Train Loss: {(tloss / max(num_batches, 1)):.4f} | Acc: {acc:.2f}% | "
              f"IoU w/bg: {sum(avg_iou) / num_classes:.4f} | IoU no/bg: {sum(avg_iou[1:]) / (num_classes - 1):.4f}")

        if hasattr(scm1, "node_names") and scm1.node_names:
            parts1, parts2 = [], []
            for nname in scm1.node_names:
                n1 = proxy_n_s1.get(nname, 0)
                n2 = proxy_n_s2.get(nname, 0)
                mse1 = proxy_sq_sum_s1[nname] / n1 if n1 > 0 else None
                mse2 = proxy_sq_sum_s2[nname] / n2 if n2 > 0 else None
                parts1.append(f"{nname}={mse1:.4f} (n={n1})" if mse1 is not None else f"{nname}=NA (n=0)")
                parts2.append(f"{nname}={mse2:.4f} (n={n2})" if mse2 is not None else f"{nname}=NA (n=0)")
            print(f"[{task_name} | Client {cid}] Proxy MSE (Split-1, epoch {ep+1}): " + ", ".join(parts1))
            print(f"[{task_name} | Client {cid}] Proxy MSE (Split-2, epoch {ep+1}): " + ", ".join(parts2))

    # ---------- Compute masks for aggregation ----------
    ss_mask = {name: (g / max(num_updates, 1) > mask_thresh).float() for name, g in ss_grad.items()}
    d1_mask = {name: (g / max(num_updates, 1) > mask_thresh).float() for name, g in d1_grad.items()}
    d2_mask = {name: (g / max(num_updates, 1) > mask_thresh).float() for name, g in d2_grad.items()}

    return ss_mask, d1_mask, d2_mask


def evaluate(loader, FE, SS, BE, denoiser1, denoiser2, exo1, scm1, exo2, scm2,
             loss_fn, num_classes,
             proxy_table: ProxyTable,
             task_name: str = "",
             cid: int = -1):
    """
    Corrected evaluate():
      - accumulate per-batch IoU and divide by number of batches (not dataset size)
      - use zero_division=0 for jaccard_score
      - compute acc by averaging per-batch accuracies
    """
    total_loss = 0.0
    total_correct = 0.0
    iou_c = [0.0] * num_classes
    proxy_sq_sum_s1 = defaultdict(float)
    proxy_n_s1 = defaultdict(int)
    num_batches = 0

    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                data, target, fnames = batch
                fnames = list(fnames)
            else:
                data, target = batch
                fnames = [str(i) for i in range(data.size(0))]
            data, target = data.to(DEVICE), target.long().to(DEVICE)
            B = data.size(0)
            t_vec = torch.full((B,), max(1, T // 2), device=DEVICE, dtype=torch.long)

            x1 = FE(data)
            u_list_1, _, _ = exo1(x1)
            z_list_1 = scm1(u_list_1)
            zc1 = scm1.as_vector(z_list_1)

            eps1 = torch.randn_like(x1)
            sqrt_ab = extract_into(alpha_cum.sqrt(), t_vec, x1.shape)
            sqrt_1mab = extract_into((1.0 - alpha_cum).sqrt(), t_vec, x1.shape)
            x_t1 = sqrt_ab * x1 + sqrt_1mab * eps1
            eps_pred1 = denoiser1(x_t1, t_vec, zc1)
            x1_hat = (x_t1 - sqrt_1mab * eps_pred1) / (sqrt_ab + 1e-8)

            x2 = SS(x1_hat)
            u_list_2, _, _ = exo2(x2)
            z_list_2 = scm2(u_list_2)
            zc2 = scm2.as_vector(z_list_2)

            eps2 = torch.randn_like(x2)
            x_t2 = sqrt_ab * x2 + sqrt_1mab * eps2
            eps_pred2 = denoiser2(x_t2, t_vec, zc2)
            x2_hat = (x_t2 - sqrt_1mab * eps_pred2) / (sqrt_ab + 1e-8)

            preds = BE(x2_hat)
            seg_loss = loss_fn(preds, target)

            if proxy_table is not None and proxy_table.enabled:
                targets_map = proxy_table.get_batch(fnames, device=DEVICE)
                if hasattr(scm1, "node_names"):
                    z1_scalar = nodewise_scalar(z_list_1)
                    for i_node, nname in enumerate(scm1.node_names):
                        if nname in targets_map:
                            tgt = targets_map[nname]
                            mask = torch.isfinite(tgt)
                            if mask.any():
                                err1 = (z1_scalar[i_node][mask] - tgt[mask]) ** 2
                                proxy_sq_sum_s1[nname] += float(err1.sum().item())
                                proxy_n_s1[nname] += int(mask.sum().item())

            loss = seg_loss

            preds_lbl = torch.argmax(preds, dim=1)
            total_correct += (preds_lbl == target).float().mean().item()
            total_loss += float(loss.item())

            # use zero_division=0 to avoid warnings/NaNs when a class is absent
            ious = jaccard_score(
                target.detach().cpu().flatten(),
                preds_lbl.detach().cpu().flatten(),
                average=None, labels=list(range(num_classes)), zero_division=0
            )
            for i in range(num_classes):
                iou_c[i] += float(ious[i])

            num_batches += 1

    # Use number of batches to compute averages (correct)
    if num_batches == 0:
        num_batches = 1

    acc = 100. * (total_correct / num_batches)
    avg_iou = [v / num_batches for v in iou_c]

    # ---- Compute average proxy MSE per client ----
    proxy_mse = 0.0
    total_count = 0
    for nname in proxy_sq_sum_s1:
        if proxy_n_s1[nname] > 0:
            proxy_mse += proxy_sq_sum_s1[nname] / proxy_n_s1[nname]
            total_count += 1
    proxy_mse = proxy_mse / max(1, total_count)
    if total_count == 0:   # no proxy matches
        proxy_mse = 10.0   # large penalty
    proxy_mse = max(proxy_mse, 1e-3)

    N = len(loader.dataset)
    print(f"Eval Loss: {total_loss / max(num_batches, 1):.4f} | Eval Acc: {acc:.2f}% | "
          f"IoU w/bg: {sum(avg_iou) / num_classes:.4f} | IoU no/bg: {sum(avg_iou[1:]) / (num_classes - 1):.4f} | "
          f"Proxy MSE: {proxy_mse:.4f}")

    return total_loss / max(num_batches, 1), acc, avg_iou, avg_iou[1:], proxy_mse


# ------------------- NEW: full metrics collector used at test time -------------------
# Replace the previous compute_full_metrics with this streaming, low-memory implementation
@torch.no_grad()
def compute_full_metrics(loader, FE, SS, BE, denoiser1, denoiser2, exo1, scm1, exo2, scm2,
                         num_classes, device=DEVICE):
    """
    Streaming computation of segmentation and reconstruction (latent) metrics without storing
    all latent tensors. Uses diagonal-covariance approximation for FID (fast, memory-light).
    Returns a dict with seg and recon summaries.
    """
    preds_all = []
    targets_all = []

    # Streaming stats for latent recon metrics
    # We'll compute per-sample MSE/PSNR/SSIM sums and averages
    mse_sum_x1 = 0.0
    mse_sum_x2 = 0.0
    psnr_sum_x1 = 0.0
    psnr_sum_x2 = 0.0
    ssim_sum_x1 = 0.0
    ssim_sum_x2 = 0.0
    lpips_sum_x1 = 0.0
    lpips_sum_x2 = 0.0
    lpips_count = 0

    n_samples = 0

    # For diagonal-FID we keep sum and sumsq (elementwise)
    sum_feat_x1 = None
    sumsq_feat_x1 = None
    sum_feat_x1hat = None
    sumsq_feat_x1hat = None

    sum_feat_x2 = None
    sumsq_feat_x2 = None
    sum_feat_x2hat = None
    sumsq_feat_x2hat = None

    # LPIPS model (if available) compute on CPU to avoid GPU blowup
    lpips_fn = None
    if LPIPS_AVAILABLE:
        try:
            lpips_fn = lpips.LPIPS(net='alex').to("cpu")
        except Exception:
            lpips_fn = None

    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                data, target, fnames = batch
            else:
                data, target = batch
            data = data.to(device)
            target = target.long().to(device)

            B = data.size(0)
            t_vec = torch.full((B,), max(1, T // 2), device=device, dtype=torch.long)

            # forward
            x1 = FE(data)                    # (B, C1, H, W)
            u_list_1, _, _ = exo1(x1)
            z_list_1 = scm1(u_list_1)
            zc1 = scm1.as_vector(z_list_1)

            eps1 = torch.randn_like(x1)
            sqrt_ab = extract_into(alpha_cum.sqrt(), t_vec, x1.shape)
            sqrt_1mab = extract_into((1.0 - alpha_cum).sqrt(), t_vec, x1.shape)
            x_t1 = sqrt_ab * x1 + sqrt_1mab * eps1
            eps_pred1 = denoiser1(x_t1, t_vec, zc1)
            x1_hat = (x_t1 - sqrt_1mab * eps_pred1) / (sqrt_ab + 1e-8)

            x2 = SS(x1_hat)
            u_list_2, _, _ = exo2(x2)
            z_list_2 = scm2(u_list_2)
            zc2 = scm2.as_vector(z_list_2)

            eps2 = torch.randn_like(x2)
            x_t2 = sqrt_ab * x2 + sqrt_1mab * eps2
            eps_pred2 = denoiser2(x_t2, t_vec, zc2)
            x2_hat = (x_t2 - sqrt_1mab * eps_pred2) / (sqrt_ab + 1e-8)

            preds = BE(x2_hat)
            preds_lbl = torch.argmax(preds, dim=1).cpu().numpy()  # (B, H, W)
            targets_np = target.cpu().numpy()  # (B, H, W)

            preds_all.append(preds_lbl)
            targets_all.append(targets_np)

            # move latent tensors to CPU and numpy immediately (to free GPU)
            x1_cpu = x1.detach().cpu().numpy()
            x1hat_cpu = x1_hat.detach().cpu().numpy()
            x2_cpu = x2.detach().cpu().numpy()
            x2hat_cpu = x2_hat.detach().cpu().numpy()

            # sample-wise metrics (MSE / PSNR / SSIM)
            for i in range(x1_cpu.shape[0]):
                n_samples += 1
                # x1
                c_clean = x1_cpu[i]   # (C,H,W)
                c_hat = x1hat_cpu[i]
                mse_val = float(np.mean((c_clean - c_hat) ** 2))
                mse_sum_x1 += mse_val
                # PSNR: data-range per-sample
                dr = float(c_clean.max() - c_clean.min())
                if dr <= 0:
                    dr = 1.0
                try:
                    psnr_sum_x1 += float(sk_psnr(c_clean, c_hat, data_range=dr))
                except Exception:
                    psnr_sum_x1 += 0.0
                # SSIM: per-channel average
                try:
                    ch_ssims = []
                    for ch in range(c_clean.shape[0]):
                        denom = (c_clean[ch].max() - c_clean[ch].min())
                        denom = denom if denom != 0 else 1.0
                        ch_ssims.append(float(sk_ssim(c_clean[ch], c_hat[ch], data_range=denom)))
                    ssim_sum_x1 += float(np.mean(ch_ssims))
                except Exception:
                    ssim_sum_x1 += 0.0

                # x2
                c2_clean = x2_cpu[i]
                c2_hat = x2hat_cpu[i]
                mse_val2 = float(np.mean((c2_clean - c2_hat) ** 2))
                mse_sum_x2 += mse_val2
                dr2 = float(c2_clean.max() - c2_clean.min())
                if dr2 <= 0:
                    dr2 = 1.0
                try:
                    psnr_sum_x2 += float(sk_psnr(c2_clean, c2_hat, data_range=dr2))
                except Exception:
                    psnr_sum_x2 += 0.0
                try:
                    ch_ssims2 = []
                    for ch in range(c2_clean.shape[0]):
                        denom = (c2_clean[ch].max() - c2_clean[ch].min())
                        denom = denom if denom != 0 else 1.0
                        ch_ssims2.append(float(sk_ssim(c2_clean[ch], c2_hat[ch], data_range=denom)))
                    ssim_sum_x2 += float(np.mean(ch_ssims2))
                except Exception:
                    ssim_sum_x2 += 0.0

                # LPIPS (optional) - compute on CPU if net available and channels==3
                if lpips_fn is not None and c_clean.shape[0] == 3:
                    try:
                        # LPIPS expects (N,3,H,W) in [-1,1]
                        def norm01(t):
                            t = t - float(t.min())
                            denom = float(t.max() - t.min())
                            if denom <= 0:
                                denom = 1.0
                            return 2.0 * (t / denom) - 1.0
                        tc = torch.from_numpy(norm01(c_clean)).unsqueeze(0).float()
                        th = torch.from_numpy(norm01(c_hat)).unsqueeze(0).float()
                        with torch.no_grad():
                            s = lpips_fn(tc, th).item()
                        lpips_sum_x1 += s
                        # x2
                        tc2 = torch.from_numpy(norm01(c2_clean)).unsqueeze(0).float()
                        th2 = torch.from_numpy(norm01(c2_hat)).unsqueeze(0).float()
                        with torch.no_grad():
                            s2 = lpips_fn(tc2, th2).item()
                        lpips_sum_x2 += s2
                        lpips_count += 1
                    except Exception:
                        pass

            # update diagonal FID approx stats: sum and sumsq per element
            # shapes: (B, C, H, W) -> flatten to (B, D)
            def _update_flat_stats(arr_np, sum_ref, sumsq_ref):
                # arr_np: (B, C, H, W)
                flat = arr_np.reshape(arr_np.shape[0], -1).astype(np.float64)  # (B, D)
                if sum_ref is None:
                    sum_ref = np.sum(flat, axis=0)
                    sumsq_ref = np.sum(flat * flat, axis=0)
                else:
                    sum_ref += np.sum(flat, axis=0)
                    sumsq_ref += np.sum(flat * flat, axis=0)
                return sum_ref, sumsq_ref

            sum_feat_x1, sumsq_feat_x1 = _update_flat_stats(x1_cpu, sum_feat_x1, sumsq_feat_x1)
            sum_feat_x1hat, sumsq_feat_x1hat = _update_flat_stats(x1hat_cpu, sum_feat_x1hat, sumsq_feat_x1hat)
            sum_feat_x2, sumsq_feat_x2 = _update_flat_stats(x2_cpu, sum_feat_x2, sumsq_feat_x2)
            sum_feat_x2hat, sumsq_feat_x2hat = _update_flat_stats(x2hat_cpu, sumsq_feat_x2hat, sumsq_feat_x2hat) if False else (sum_feat_x2hat, sumsq_feat_x2hat)
            # note: above line for x2hat was intentionally kept safe to not error if not initialized; we update below properly
            if sum_feat_x2hat is None:
                sum_feat_x2hat, sumsq_feat_x2hat = _update_flat_stats(x2hat_cpu, sum_feat_x2hat, sumsq_feat_x2hat)
            else:
                sum_feat_x2hat, sumsq_feat_x2hat = _update_flat_stats(x2hat_cpu, sum_feat_x2hat, sumsq_feat_x2hat)

            # free GPU memory explicitly
            del x1, x1_hat, x2, x2_hat, preds, data, target
            torch.cuda.empty_cache()

    # concatenate preds / targets (kept as small numpy arrays in the lists)
    preds_all_np = np.concatenate(preds_all, axis=0) if len(preds_all) > 0 else np.zeros((0, 256, 256), dtype=np.int32)
    targets_all_np = np.concatenate(targets_all, axis=0) if len(targets_all) > 0 else np.zeros((0, 256, 256), dtype=np.int32)

    # segmentation metrics
    seg_metrics = compute_segmentation_metrics_all(preds_all_np, targets_all_np, num_classes)

    # finalize recon metrics averages
    if n_samples == 0:
        n_samples = 1
    recon_x1 = {
        "mse": mse_sum_x1 / n_samples,
        "psnr": psnr_sum_x1 / n_samples,
        "ssim": ssim_sum_x1 / n_samples,
        "lpips": (lpips_sum_x1 / lpips_count) if lpips_count > 0 else None,
        "fid": None
    }
    recon_x2 = {
        "mse": mse_sum_x2 / n_samples,
        "psnr": psnr_sum_x2 / n_samples,
        "ssim": ssim_sum_x2 / n_samples,
        "lpips": (lpips_sum_x2 / lpips_count) if lpips_count > 0 else None,
        "fid": None
    }

    # compute diagonal-FID (approx) from sum and sumsq
    def _compute_diag_fid(sum_a, sumsq_a, sum_b, sumsq_b, n):
        # mean vectors
        mu_a = sum_a / n
        mu_b = sum_b / n
        var_a = (sumsq_a / n) - (mu_a ** 2)
        var_b = (sumsq_b / n) - (mu_b ** 2)
        # numeric safety
        var_a = np.maximum(var_a, 1e-6)
        var_b = np.maximum(var_b, 1e-6)
        diff = mu_a - mu_b
        # Frechet for diagonal cov: sum(diff^2) + sum(var_a + var_b - 2*sqrt(var_a*var_b))
        term1 = np.sum(diff ** 2)
        term2 = np.sum(var_a + var_b - 2.0 * np.sqrt(var_a * var_b))
        return float(term1 + term2)

    try:
        # use n_samples as count
        if sum_feat_x1 is not None and sum_feat_x1hat is not None:
            recon_x1['fid'] = _compute_diag_fid(sum_feat_x1, sumsq_feat_x1, sum_feat_x1hat, sumsq_feat_x1hat, n_samples)
        if sum_feat_x2 is not None and sum_feat_x2hat is not None:
            recon_x2['fid'] = _compute_diag_fid(sum_feat_x2, sumsq_feat_x2, sum_feat_x2hat, sumsq_feat_x2hat, n_samples)
    except Exception:
        recon_x1['fid'] = None
        recon_x2['fid'] = None

    summary = {
        "seg": seg_metrics,
        "recon_x1": recon_x1,
        "recon_x2": recon_x2,
        "n_samples": n_samples
    }
    return summary



# ------------------------------------------------------------------------------------


def main():
    task_info = {
        0: {"name": "Blastocyst", "num_classes": 5, "path": DATA_PATH, "dataset": EmbryoDataset,
            "proxy_paths": {"train": os.path.join(PROXY_DATA_PATH, "Blastocysts", "blastocysts_train.csv"),
                            "val": os.path.join(PROXY_DATA_PATH, "Blastocysts", "blastocysts_val.csv"),
                            "test": os.path.join(PROXY_DATA_PATH, "Blastocysts", "blastocysts_test.csv")}},
        1: {"name": "HAM10K", "num_classes": 2, "path": DATA_PATH, "dataset": HAMDataset,
            "proxy_paths": {"train": os.path.join(PROXY_DATA_PATH, "HAM10K", "ham10k_train.csv"),
                            "val": os.path.join(PROXY_DATA_PATH, "HAM10K", "ham10k_val.csv"),
                            "test": os.path.join(PROXY_DATA_PATH, "HAM10K", "ham10k_test.csv")}},
        2: {"name": "Fetal", "num_classes": 3, "path": DATA_PATH, "dataset": FHPsAOPMSBDataset,
            "proxy_paths": {"train": os.path.join(PROXY_DATA_PATH, "Fetus", "fetus_train.csv"),
                            "val": os.path.join(PROXY_DATA_PATH, "Fetus", "fetus_val.csv"),
                            "test": os.path.join(PROXY_DATA_PATH, "Fetus", "fetus_test.csv")}},
        3: {"name": "Mosmed", "num_classes": 2, "path": DATA_PATH, "dataset": covidCTDataset,
            "proxy_paths": {"train": os.path.join(PROXY_DATA_PATH, "Mosmed", "mosmed_train.csv"),
                            "val": os.path.join(PROXY_DATA_PATH, "Mosmed", "mosmed_val.csv"),
                            "test": os.path.join(PROXY_DATA_PATH, "Mosmed", "mosmed_test.csv")}},
        4: {"name": "Kvasir", "num_classes": 2, "path": DATA_PATH, "dataset": CVCDataset,
            "proxy_paths": {"train": os.path.join(PROXY_DATA_PATH, "Kvasir-SEG", "kvasir_train.csv"),
                            "val": os.path.join(PROXY_DATA_PATH, "Kvasir-SEG", "kvasir_val.csv"),
                            "test": os.path.join(PROXY_DATA_PATH, "Kvasir-SEG", "kvasir_test.csv")}}
    }

    global_SS = UNET_server(in_channels=32).to(DEVICE)
    global_exo1 = None
    global_exo2 = None
    global_scm1 = None
    global_scm2 = None
    global_den1 = None
    global_den2 = None

    # Global templates (for round 1 init of all clients)
    global_exo_template1, global_exo_template2 = None, None
    global_scm_template1, global_scm_template2 = None, None
    global_den_template1, global_den_template2 = None, None

    client_FE = [None] * NUM_CLIENTS
    client_BE = [None] * NUM_CLIENTS
    client_dom1 = [None] * NUM_CLIENTS
    client_dom2 = [None] * NUM_CLIENTS
    client_scm1 = [None] * NUM_CLIENTS   # NEW: per-client SCM1
    client_scm2 = [None] * NUM_CLIENTS   # NEW: per-client SCM2

    best_loss = float("inf")

    # alias map for tasks (used later inside loop)
    alias_map_by_task = {
        "Blastocyst": {
            "ZPThicknessVar": "ZPThicknessVar",
            "TEArea": "TEArea",
            "BubbleArea": "BubbleArea",
            "BlastocoelSym": "BlastocoelSym",
        },
        "HAM10K": {
            "LesionSize": "LesionSize",
            "Compactness": "Compactness",
            "Asymmetry": "Asymmetry",
            "MeanRGB": "MeanRGB",
        },
        "Fetal": {
            "HeadPerimeter": "HeadPerimeter",
            "HeadMajorAxis": "HeadMajorAxis",
            "HeadArea": "HeadArea",
            "HeadCircularity": "HeadCircularity",
        },
        "Mosmed": {
            "LesionSize": "LesionSize",
            "Solidity": "Solidity",
            "Orientation": "Orientation",
            "Perimeter": "Perimeter",
        },
        "Kvasir": {
            "MeanRGB": "MeanRGB",
            "LesionSize": "LesionSize",
            "Compactness": "Compactness",
            "Asymmetry": "Asymmetry",
        },
    }

    for r in range(COMM_ROUNDS):
        print(f"\n[Comm Round {r + 1}/{COMM_ROUNDS}]")
        client_sizes, val_losses, proxy_mse_scores = [], [], []

        for i in range(NUM_CLIENTS):
            path, ds_class = task_info[i]["path"], task_info[i]["dataset"]
            loader = get_loader(
                os.path.join(path, f"client{i + 1}/train_imgs"),
                os.path.join(path, f"client{i + 1}/train_masks"),
                ds_class, get_transforms(task_info[i]["name"]),
                with_names=True, shuffle=True
            )
            client_sizes.append(len(loader.dataset))
        max_size = max(client_sizes)

        local_SS, local_exo1, local_exo2 = [], [], []
        local_scm1, local_scm2 = [], []
        local_den1, local_den2 = [], []
        client_FEs_round, client_BEs_round = [], []
        val_proxies_round, test_proxies_round = [], []

        for i in range(NUM_CLIENTS):
            task = task_info[i]
            task_name, num_classes, path, ds_class = (
                task["name"], task["num_classes"], task["path"], task["dataset"]
            )

            cfg = get_scm_config(task_name)
            nodes, parents, node_dims = cfg["nodes"], cfg["parents"], cfg["node_dims"]
            cond_dim = sum(node_dims)

            # Initialize global templates ONCE (round 1, client 0)
            if global_exo1 is None:
                global_exo_template1 = ExogenousEncoder(in_channels=32, node_dims=node_dims, variational=True).to(
                    DEVICE)
                global_exo_template2 = ExogenousEncoder(in_channels=32, node_dims=node_dims, variational=True).to(
                    DEVICE)
                global_scm_template1 = NeuralSCM(parents=parents, node_dims=node_dims).to(DEVICE);
                global_scm_template1.node_names = nodes
                global_scm_template2 = NeuralSCM(parents=parents, node_dims=node_dims).to(DEVICE);
                global_scm_template2.node_names = nodes
                global_den_template1 = initialize_conditional_denoiser(32, cond_dim, 128, DEVICE)
                global_den_template2 = initialize_conditional_denoiser(32, cond_dim, 128, DEVICE)

                # also set "current global" copies (will get updated after aggregation)
                global_exo1, global_exo2 = copy.deepcopy(global_exo_template1), copy.deepcopy(global_exo_template2)
                global_scm1, global_scm2 = copy.deepcopy(global_scm_template1), copy.deepcopy(global_scm_template2)
                global_den1, global_den2 = copy.deepcopy(global_den_template1), copy.deepcopy(global_den_template2)

            # FE/BE stay local (but same arch)
            if client_FE[i] is None:
                FE, BE = UNET_FE(in_channels=3).to(DEVICE), UNET_BE(out_channels=num_classes).to(DEVICE)
                client_FE[i], client_BE[i] = FE, BE
            else:
                FE, BE = client_FE[i].to(DEVICE), client_BE[i].to(DEVICE)

            # SS + denoisers + exo + SCM are shared (all start same at round 1)
            SS = copy.deepcopy(global_SS)
            exo1 = copy.deepcopy(global_exo1)
            exo2 = copy.deepcopy(global_exo2)
            scm1 = copy.deepcopy(global_scm1);
            scm1.node_names = nodes
            scm2 = copy.deepcopy(global_scm2);
            scm2.node_names = nodes
            denoiser1 = copy.deepcopy(global_den1)
            denoiser2 = copy.deepcopy(global_den2)

            # Save SCMs per client (NEW)
            client_scm1[i] = scm1
            client_scm2[i] = scm2

            # Discriminators (local)
            if client_dom1[i] is None:
                dom_disc1, dom_disc2 = DomainDiscriminator().to(DEVICE), DomainDiscriminator().to(DEVICE)
                client_dom1[i], client_dom2[i] = dom_disc1, dom_disc2
            else:
                dom_disc1, dom_disc2 = client_dom1[i].to(DEVICE), client_dom2[i].to(DEVICE)

            dom_opt = optim.Adam(list(dom_disc1.parameters()) + list(dom_disc2.parameters()), lr=1e-4)
            opt = optim.AdamW([
                {"params": list(FE.parameters()) + list(BE.parameters()) + list(SS.parameters()), "lr": 1e-4},
                {"params": list(exo1.parameters()) + list(scm1.parameters()) +
                           list(exo2.parameters()) + list(scm2.parameters()), "lr": 5e-5},
                {"params": list(denoiser1.parameters()) + list(denoiser2.parameters()), "lr": 1e-4},
            ])

            loss_fn = ComboLoss(num_classes)

            tr_tf, val_tf = get_transforms(task_name), A.Compose([
                A.Resize(256, 256), A.Normalize(mean=[0] * 3, std=[1] * 3, max_pixel_value=255.0), ToTensorV2()
            ])
            train_loader = get_loader(os.path.join(path, f"client{i + 1}/train_imgs"),
                                      os.path.join(path, f"client{i + 1}/train_masks"),
                                      ds_class, tr_tf, with_names=True, shuffle=True, fraction=1.0)
            val_loader = get_loader(os.path.join(path, f"client{i + 1}/val_imgs"),
                                    os.path.join(path, f"client{i + 1}/val_masks"),
                                    ds_class, val_tf, with_names=True, shuffle=False, fraction=1.0)
            test_loader = get_loader(os.path.join(path, f"client{i + 1}/test_imgs"),
                                     os.path.join(path, f"client{i + 1}/test_masks"),
                                     ds_class, val_tf, with_names=True, shuffle=False, fraction=1.0)

            ptrain, pval, ptest = task["proxy_paths"]["train"], task["proxy_paths"]["val"], task["proxy_paths"]["test"]
            alias_map = alias_map_by_task.get(task_name, {})

            train_proxy = ProxyTable(ptrain, nodes, zscore=True, alias_map=alias_map) if ptrain else ProxyTable(None, nodes)
            val_proxy = ProxyTable(pval, nodes, zscore=True, alias_map=alias_map) if pval else ProxyTable(None, nodes)
            test_proxy = ProxyTable(ptest, nodes, zscore=True, alias_map=alias_map) if ptest else ProxyTable(None, nodes)
            train_local(train_loader, FE, SS, BE,
                        denoiser1, denoiser2, exo1, scm1, exo2, scm2,
                        dom_disc1, dom_disc2,
                        opt, dom_opt, loss_fn, i + 1, num_classes,
                        task_name, train_proxy,
                        fallback_state=global_SS.state_dict(),
                        client_size=len(train_loader.dataset), max_size=max_size)

            val_loss, _, _, _, proxy_mse = evaluate(
                val_loader, FE, SS, BE,
                denoiser1, denoiser2, exo1, scm1, exo2, scm2,
                loss_fn, num_classes,
                proxy_table=val_proxy,
                task_name=task_name,
                cid=i + 1
            )

            val_losses.append(val_loss)
            proxy_mse_scores.append(proxy_mse)
            client_FE[i], client_BE[i] = FE, BE
            client_FEs_round.append(FE)
            client_BEs_round.append(BE)
            local_SS.append(SS)
            local_exo1.append(exo1)
            local_exo2.append(exo2)
            local_scm1.append(scm1)
            local_scm2.append(scm2)
            local_den1.append(denoiser1)
            local_den2.append(denoiser2)
            val_proxies_round.append(val_proxy)
            test_proxies_round.append(test_proxy)

        # --- Aggregation step (only SS + exos + denoisers aggregated) ---

        if r < 24:  # first rounds
            global_SS.load_state_dict(equal_weight_fusion(local_SS))
            global_exo1.load_state_dict(equal_weight_fusion(local_exo1))
            global_exo2.load_state_dict(equal_weight_fusion(local_exo2))
            global_den1.load_state_dict(equal_weight_fusion(local_den1))
            global_den2.load_state_dict(equal_weight_fusion(local_den2))

        else:
            global_SS.load_state_dict(causal_invariant_fusion(local_SS, proxy_mse_scores, client_sizes, val_losses, min_fraction=0.25, blend_equal=0.5))
            global_exo1.load_state_dict(causal_invariant_fusion(local_exo1, proxy_mse_scores, client_sizes, val_losses, min_fraction=0.25, blend_equal=0.5))
            global_exo2.load_state_dict(causal_invariant_fusion(local_exo2, proxy_mse_scores, client_sizes, val_losses, min_fraction=0.25, blend_equal=0.5))
            global_den1.load_state_dict(causal_invariant_fusion(local_den1, proxy_mse_scores, client_sizes, val_losses, min_fraction=0.25, blend_equal=0.5))
            global_den2.load_state_dict(causal_invariant_fusion(local_den2, proxy_mse_scores, client_sizes, val_losses, min_fraction=0.25, blend_equal=0.5))

        total_val_loss, total_val_ious = 0.0, [0.0] * 5
        for i in range(NUM_CLIENTS):
            task = task_info[i]
            task_name = task["name"]
            num_classes = task["num_classes"]
            ds_class = task["dataset"]
            cfg = get_scm_config(task_name)
            nodes = cfg["nodes"]
            alias_map = alias_map_by_task.get(task_name, {})

            val_proxy = ProxyTable(task["proxy_paths"]["val"], nodes, zscore=True, alias_map=alias_map)
            val_loader = get_loader(os.path.join(task["path"], f"client{i + 1}/val_imgs"),
                                    os.path.join(task["path"], f"client{i + 1}/val_masks"),
                                    ds_class, val_tf, with_names=True, shuffle=False, fraction=1.0)
            loss_fn = ComboLoss(num_classes)
            val_loss, val_acc, val_iou_wb, val_iou_nb, _ = evaluate(
                val_loader, client_FEs_round[i], global_SS, client_BEs_round[i],
                global_den1, global_den2, global_exo1, client_scm1[i], global_exo2, client_scm1[i],
                loss_fn, num_classes,
                proxy_table=val_proxy,
                task_name=task_name
            )
            total_val_loss += val_loss
            for j in range(num_classes):
                total_val_ious[j] += val_iou_wb[j]

        avg_val_loss = total_val_loss / NUM_CLIENTS
        avg_ious = [v / NUM_CLIENTS for v in total_val_ious]
        print(f"[Global Validation] Loss: {avg_val_loss:.4f} | IoU w/bg: {sum(avg_ious) / 5:.4f}")

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(global_SS.state_dict(), "BestModels/MTS5_best_old_SS_modi2.pth")
            torch.save(global_exo1.state_dict(), "BestModels/MTS5_best_old_exo1_modi2.pth")
            torch.save(global_exo2.state_dict(), "BestModels/MTS5_best_old_exo2_modi2.pth")
            torch.save(global_scm1.state_dict(), "BestModels/MTS5_best_old_scm1_modi2.pth")
            torch.save(global_scm2.state_dict(), "BestModels/MTS5_best_old_scm2_modi2.pth")
            torch.save(global_den1.state_dict(), "BestModels/MTS5_best_old_den1_modi2.pth")
            torch.save(global_den2.state_dict(), "BestModels/MTS5_best_old_den2_modi2.pth")
            print("[Best Global Models Saved]")

        # TEST & METRICS: compute metrics per client and print summary
        for i in range(NUM_CLIENTS):
            task = task_info[i]
            task_name = task["name"]
            num_classes = task["num_classes"]
            ds_class = task["dataset"]
            cfg = get_scm_config(task_name)
            nodes = cfg["nodes"]
            alias_map = alias_map_by_task.get(task_name, {})

            # ensure we use the correct task path for each client
            path = task["path"]

            test_proxy = ProxyTable(task["proxy_paths"]["test"], nodes, zscore=True, alias_map=alias_map)
            test_loader = get_loader(os.path.join(path, f"client{i + 1}/test_imgs"),
                                     os.path.join(path, f"client{i + 1}/test_masks"),
                                     ds_class, val_tf, with_names=True, shuffle=False, fraction=1.0)
            loss_fn = ComboLoss(num_classes)
            test_loss, test_acc, test_iou_wb, test_iou_nb, proxy_mse = evaluate(
                test_loader, client_FEs_round[i], global_SS, client_BEs_round[i],
                global_den1, global_den2, global_exo1, client_scm1[i], global_exo2, client_scm2[i],
                loss_fn, num_classes,
                proxy_table=test_proxy,
                task_name=task_name
            )
            print(f"[Client {i + 1} Testing] Loss: {test_loss:.4f} | IoU wb: {sum(test_iou_wb) / num_classes:.4f}")
            test_iou_wbg_all[i].append(sum(test_iou_wb) / num_classes)
            test_iou_nbg_all[i].append(sum(test_iou_nb) / (num_classes - 1))
            #test_proxy_mse_all[i].append(sum(proxy_mse) / (num_classes - 1))


            # --- SAVE test predictions for this client (overwrites previous round) ---
            save_test_images(
                test_loader,
                client_FEs_round[i],
                global_SS,
                client_BEs_round[i],
                global_den1,
                global_den2,
                global_exo1,
                client_scm1[i],
                global_exo2,
                client_scm2[i],
                num_classes=num_classes,
                out_dir_root="Outputs/unet_mucald_test_preds",
                cid=i + 1
            )

            # --- NEW: compute detailed metrics (seg + reconstruction on latent) ---
            print(f"[Client {i + 1}] Computing detailed metrics (segmentation + latent reconstructions)...")
            metrics = compute_full_metrics(test_loader,
                                           client_FEs_round[i], global_SS, client_BEs_round[i],
                                           global_den1, global_den2, global_exo1, client_scm1[i],
                                           global_exo2, client_scm2[i],
                                           num_classes=num_classes, device=DEVICE)

            seg = metrics["seg"]
            recon1 = metrics["recon_x1"]
            recon2 = metrics["recon_x2"]

            # Print segmentation summary
            print(f"--- Client {i+1} ({task_name}) Segmentation summary ---")
            for c in range(num_classes):
                print(f"Class {c}: IoU={seg['iou'][c]:.4f} | Dice={seg['dice'][c]:.4f} | "
                      f"P={seg['precision'][c]:.4f} R={seg['recall'][c]:.4f} F1={seg['f1'][c]:.4f} | "
                      f"HD95={seg['hd95'][c]:.2f} px | ASSD={seg['assd'][c]:.3f} px")
            mean_iou_wbg = float(np.mean(seg['iou']))
            mean_iou_nbg = float(np.mean(seg['iou'][1:])) if num_classes > 1 else mean_iou_wbg
            mean_dice = float(np.mean(seg['dice']))
            mean_P = float(np.mean(seg['precision']))
            mean_R = float(np.mean(seg['recall']))
            mean_f1 = float(np.mean(seg['f1']))
            mean_HD95 = float(np.mean(seg['hd95']))
            mean_ASSD = float(np.mean(seg['assd']))
            
            print(f"Summary: mean IoU (w/bg)={mean_iou_wbg:.4f}, mean IoU (no/bg)={mean_iou_nbg:.4f}, mean Dice={mean_dice:.4f}, mean P={mean_P:.4f}, mean R={mean_R:.4f}, mean F1={mean_f1:.4f}, mean hd95={mean_HD95:.4f}, mean ASSD={mean_ASSD:.4f}, ")

            # Print reconstruction summary
            print(f"--- Client {i+1} ({task_name}) Reconstruction (latent) summary ---")
            print(f"x1 (split1)   : MSE={recon1['mse']:.6f} | PSNR={recon1['psnr']:.4f} | SSIM={recon1['ssim']:.4f} | FID={recon1['fid']}")
            print(f"x2 (split2)   : MSE={recon2['mse']:.6f} | PSNR={recon2['psnr']:.4f} | SSIM={recon2['ssim']:.4f} | FID={recon2['fid']}")
            if LPIPS_AVAILABLE:
                print(f"LPIPS (x1): {recon1['lpips']}, LPIPS (x2): {recon2['lpips']}")
            else:
                print("LPIPS: package not available, skipped.")

        plot_curves(r + 1)
        #plot_proxy_mse_curves(r+1)

if __name__ == "__main__":
    main()
