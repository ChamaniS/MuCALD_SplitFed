# save as visualize_fe_and_lddm.py and run with your project PYTHONPATH so that
# models.* and dataset.* imports resolve (same environment as your main training script)
import os
CUDA_LAUNCH_BLOCKING = 1
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.autograd import Function
import copy
import numpy as np
from PIL import Image
import os.path as osp

# ---- import your project modules (assumed available in PYTHONPATH) ----
from models.clientmodel_FE import UNET_FE
from models.servermodel import UNET_server
from dataset import EmbryoDataset, HAMDataset, CVCDataset, covidCTDataset, FHPsAOPMSBDataset
from reverse_diff_causal import initialize_conditional_denoiser
from models.exogenous_encoder import ExogenousEncoder
from models.neural_scm import NeuralSCM
from scm_configs import get_scm_config
from dataset_wrappers import WithFilenames

# ---- basic config (adapt paths as necessary) ----
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLIENTS = 5
T = 400

PROXY_DATA_PATH = "C:/Users/csj5/Projects/Causal-proxy/Proxy_variables_dir/Final/"
DATA_PATH = "C:/Users/csj5/Projects/Data/MTS2"

os.makedirs("Outputs/feature_maps_compare", exist_ok=True)

# ---- diffusion schedule utilities (copied from your script) ----
def cosine_beta_schedule(T:int, s:float=0.008, device:torch.device=None):
    device = device or torch.device("cpu")
    t = torch.linspace(0, T, T+1, device=device) / T
    f = torch.cos(((t + s) / (1 + s)) * torch.pi / 2) ** 2
    f = f / f[0]
    betas = 1 - (f[1:] / f[:-1])
    return betas.clamp(1e-8, 0.999)

beta = cosine_beta_schedule(T, device=torch.device(DEVICE))
alpha = 1.0 - beta
alpha_cum = torch.cumprod(alpha, dim=0).to(DEVICE)

def extract_into(arr_1d, timesteps, x_shape):
    out = arr_1d.gather(0, timesteps.clamp_(0, arr_1d.shape[0]-1))
    return out.view(-1, *([1] * (len(x_shape)-1)))

# ---- simple transform used for visualization (same as your pipeline) ----
def get_transforms():
    return A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=[0]*3, std=[1]*3, max_pixel_value=255.0),
        ToTensorV2()
    ])

# ---- dataset loader helper ----
def get_loader(img_dir, mask_dir, dataset_class, transform, batch_size=1, num_workers=1, with_names=True, shuffle=False, fraction=1.0):
    base = dataset_class(img_dir, mask_dir, transform=transform)
    ds = WithFilenames(base, img_dir) if with_names else base
    if fraction < 1.0:
        n = int(len(ds) * fraction)
        ds = Subset(ds, list(range(n)))
    return DataLoader(ds, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)

# ---- Visualization: compare raw FE output vs LDDM1 denoised x1_hat ----
@torch.no_grad()
def visualize_fe_vs_lddm(task_info, FE_list, ss_template, denoiser_template,
                          exo_template, scm_template,
                          device="cuda",
                          samples_per_client=18,   # how many images to collect per client (will be split for channels)
                          samples_per_channel=9,   # must be 9 for 3x3 grids
                          batch_size=1,
                          cmap="cubehelix",
                          out_root="Outputs/feature_maps_compare"):
    """
    For each client/task:
      - Collect up to `samples_per_client` test images
      - Compute:
          raw FE output x1 (direct output from CS-FE)
          denoised x1_hat using exo->scm->denoiser (LDDM1)
      - For each channel in x1 (C channels):
          - Create a 3x6 figure: left 3x3 = that channel across 9 images (raw FE),
                            right 3x3 = same channel across 9 images (x1_hat denoised).
          - Save as PNG named client{idx}_{task}_ch{ch:03d}.png
    """
    assert samples_per_channel == 9, "samples_per_channel must be 9 (3x3)"
    os.makedirs(out_root, exist_ok=True)

    for i in range(len(task_info)):
        info = task_info[i]
        task_name = info["name"]
        path = info["path"]
        ds_class = info["dataset"]

        # FE: either provided or instantiate a fresh one
        if FE_list is not None and FE_list[i] is not None:
            FE = FE_list[i].to(device).eval()
        else:
            FE = UNET_FE(in_channels=3).to(device).eval()

        SS = copy.deepcopy(ss_template).to(device).eval()            # not used directly for visualization but kept for completeness
        denoiser = copy.deepcopy(denoiser_template).to(device).eval()
        exo = copy.deepcopy(exo_template).to(device).eval()
        scm = copy.deepcopy(scm_template).to(device).eval()
        if hasattr(scm_template, "node_names"):
            scm.node_names = getattr(scm_template, "node_names", [])

        tf = get_transforms()
        loader = get_loader(os.path.join(path, f"client{i+1}/test_imgs"),
                            os.path.join(path, f"client{i+1}/test_masks"),
                            ds_class, tf, batch_size=batch_size, num_workers=1, with_names=True, shuffle=False, fraction=1.0)

        # collect up to samples_per_client items
        collected_raw = []   # list of (C,H,W) numpy arrays
        collected_denoised = []
        fnames = []
        seen = 0
        for batch in loader:
            if seen >= samples_per_client:
                break
            if len(batch) == 3:
                data, target, bfnames = batch
                bfnames = list(bfnames)
            else:
                data, target = batch
                bfnames = [f"sample{seen}"]

            data = data.to(device)
            B = data.size(0)
            # forward raw FE
            with torch.no_grad():
                x1 = FE(data)                       # (B, C, H, W)
            # compute denoised x1_hat via conditional denoiser (LDDM1) with exo+scm
            with torch.no_grad():
                u_list, _, _ = exo(x1)
                z_list = scm(u_list)
                zc = scm.as_vector(z_list)
                eps = torch.randn_like(x1)
                t_vec = torch.full((B,), max(1, T//2), device=device, dtype=torch.long)
                sqrt_ab = extract_into(alpha_cum.sqrt(), t_vec, x1.shape)
                sqrt_1mab = extract_into((1.0 - alpha_cum).sqrt(), t_vec, x1.shape)
                x_t = sqrt_ab * x1 + sqrt_1mab * eps
                eps_pred = denoiser(x_t, t_vec, zc)
                x1_hat = (x_t - sqrt_1mab * eps_pred) / (sqrt_ab + 1e-8)

            x1_cpu = x1.detach().cpu().numpy()
            x1hat_cpu = x1_hat.detach().cpu().numpy()

            for b in range(x1_cpu.shape[0]):
                if seen >= samples_per_client:
                    break
                collected_raw.append(x1_cpu[b])        # (C,H,W)
                collected_denoised.append(x1hat_cpu[b])
                fnames.append(os.path.basename(bfnames[b] if b < len(bfnames) else f"sample{seen}"))
                seen += 1

        if len(collected_raw) == 0:
            print(f"[Vis] No samples for client {i+1} ({task_name}), skipping.")
            continue

        collected_raw = np.stack(collected_raw, axis=0)         # (N, C, H, W)
        collected_denoised = np.stack(collected_denoised, axis=0)
        N, C, H, W = collected_raw.shape
        print(f"[Vis] Client {i+1} ({task_name}) collected N={N}, channels={C}, HxW={H}x{W}")

        out_dir = osp.join(out_root, f"client{i+1}_{task_name}")
        os.makedirs(out_dir, exist_ok=True)

        # For each channel produce a 3x6 grid (left raw, right denoised)
        for ch in range(C):
            # pick up to samples_per_channel indices from collected N
            idxs = list(range(min(N, samples_per_channel)))
            if len(idxs) < samples_per_channel:
                # pad by repeating last index if necessary
                last = idxs[-1] if len(idxs) > 0 else 0
                idxs += [last] * (samples_per_channel - len(idxs))

            fig, axes = plt.subplots(3, 6, figsize=(12, 6))
            fig.suptitle(f"Client {i+1} - {task_name} - channel {ch}", fontsize=12)

            # left 3 cols = raw FE (3x3), right 3 cols = denoised (3x3)
            for j, sidx in enumerate(idxs):
                # column for raw: col = j%3
                r = j // 3
                ccol = j % 3
                ax_raw = axes[r, ccol]
                fmap_raw = collected_raw[sidx, ch]
                vmin_r = float(np.nanmin(fmap_raw))
                vmax_r = float(np.nanmax(fmap_raw))
                if vmax_r - vmin_r <= 1e-8:
                    im_raw = np.zeros_like(fmap_raw)
                else:
                    im_raw = (fmap_raw - vmin_r) / (vmax_r - vmin_r)
                ax_raw.imshow(im_raw, cmap=cmap, aspect='equal')
                ax_raw.axis('off')
                ax_raw.set_title(f"BS: {fnames[sidx]}", fontsize=6)

                # column for denoised: placed in columns 3..5
                ax_den = axes[r, 3 + ccol]
                fmap_den = collected_denoised[sidx, ch]
                vmin_d = float(np.nanmin(fmap_den))
                vmax_d = float(np.nanmax(fmap_den))
                if vmax_d - vmin_d <= 1e-8:
                    im_den = np.zeros_like(fmap_den)
                else:
                    im_den = (fmap_den - vmin_d) / (vmax_d - vmin_d)
                ax_den.imshow(im_den, cmap=cmap, aspect='equal')
                ax_den.axis('off')
                ax_den.set_title(f"MuCALD: {fnames[sidx]}", fontsize=6)

            plt.tight_layout(rect=[0, 0, 1, 0.96])
            save_path = osp.join(out_dir, f"client{i+1}_{task_name}_channel_{ch:03d}.png")
            plt.savefig(save_path, dpi=150)
            plt.close(fig)
            print(f"[Vis] saved {save_path}")

        print(f"[Vis] Finished client {i+1} ({task_name}) — created {C} channel figures.")

# -----------------------------
# ENTRYPOINT: run visualization
# -----------------------------
if __name__ == "__main__":
    # Minimal task_info mapping (same as your main)
    task_info_local = {
        0: {"name": "Blastocyst", "num_classes": 5, "path": DATA_PATH, "dataset": EmbryoDataset},
        1: {"name": "HAM10K",    "num_classes": 2, "path": DATA_PATH, "dataset": HAMDataset},
        2: {"name": "Fetal",     "num_classes": 3, "path": DATA_PATH, "dataset": FHPsAOPMSBDataset},
        3: {"name": "Mosmed",    "num_classes": 2, "path": DATA_PATH, "dataset": covidCTDataset},
        4: {"name": "Kvasir",    "num_classes": 2, "path": DATA_PATH, "dataset": CVCDataset},
    }

    # Prepare templates roughly as in your main() (we only need shapes & modules for forward)
    sample_task_name = task_info_local[0]["name"]
    cfg0 = get_scm_config(sample_task_name)
    node_dims0 = cfg0["node_dims"]
    cond_dim0 = sum(node_dims0)

    ss_template = UNET_server(in_channels=32).to(DEVICE)
    exo_template = ExogenousEncoder(in_channels=32, node_dims=node_dims0, variational=True).to(DEVICE)
    scm_template = NeuralSCM(parents=cfg0["parents"], node_dims=node_dims0).to(DEVICE)
    if hasattr(scm_template, "node_names"):
        scm_template.node_names = cfg0["nodes"]
    denoiser_template = initialize_conditional_denoiser(32, cond_dim0, 128, DEVICE)

    # If you have trained FE instances available in memory, place them here (one per client).
    FE_placeholders = [None] * NUM_CLIENTS

    # Run visualization: this will create per-channel comparison PNGs under Outputs/feature_maps_compare
    visualize_fe_vs_lddm(
        task_info_local,
        FE_list=FE_placeholders,
        ss_template=ss_template,
        denoiser_template=denoiser_template,
        exo_template=exo_template,
        scm_template=scm_template,
        device=DEVICE,
        samples_per_client=18,    # collects 18 images per client (9 used per channel visualized)
        samples_per_channel=9,    # must be 9 -> 3x3 per left/right
        batch_size=1,
        cmap="cubehelix",
        out_root="Outputs/feature_maps_compare"
    )

    print("All visualizations done.")
