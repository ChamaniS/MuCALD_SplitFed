# Multi-task SplitFed + MOCHA-inspired aggregation (copy-pasteable)
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from sklearn.metrics import jaccard_score
from models.clientmodel_FE import UNET_FE
from models.servermodel import UNET_server
from models.clientmodel_BE import UNET_BE
from tqdm import tqdm
import segmentation_models_pytorch as smp
from dataset import EmbryoDataset, HAMDataset, CVCDataset, covidCTDataset, FHPsAOPMSBDataset
import matplotlib.pyplot as plt
from PIL import Image
import scipy.ndimage as ndi
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim

# NOTE: This integration is MOCHA-inspired: MOCHA is a convex primal-dual algorithm for MTL.
# Here we implement a practical MOCHA-like aggregation for deep nets where each client returns
# a local update (delta) and the server aggregates them using per-client θ parameters:
# θ in [0,1] (θ=1 -> dropped client, θ=0 -> full contribution). This is a heuristic
# adaptation (practical) rather than a formal MOCHA solver. See the MOCHA paper for details. :contentReference[oaicite:1]{index=1}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLIENTS = 5
LOCAL_EPOCHS = 1
COMM_ROUNDS = 24
DATA_PATH="C:/Users/csj5/Projects/Data/MTS2"

test_iou_wbg_all = {i: [] for i in range(NUM_CLIENTS)}
test_iou_nbg_all = {i: [] for i in range(NUM_CLIENTS)}

def get_loader(img_dir, mask_dir, dataset_class, transform, batch_size=1, num_workers=1):
    ds = dataset_class(img_dir, mask_dir, transform=transform)
    return DataLoader(ds, batch_size=batch_size, num_workers=num_workers)

class ComboLoss(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.dice = smp.losses.DiceLoss(smp.losses.MULTICLASS_MODE, from_logits=True)
        self.ce = nn.CrossEntropyLoss()
    def forward(self, logits, targets):
        return self.dice(logits, targets) + 0.0 * self.ce(logits, targets)

# -----------------------
# Local training / eval (kept mostly as you provided)
# -----------------------
def train_local(loader, FE, SS, BE, opt, loss_fn, cid, num_classes):
    FE.train(); SS.train(); BE.train()
    for ep in range(LOCAL_EPOCHS):
        tloss, tcorrect = 0.0, 0.0
        iou_c = [0.0] * num_classes
        for data, target in tqdm(loader, leave=False):
            data, target = data.to(DEVICE), target.long().to(DEVICE)
            x1 = FE(data)
            x2 = SS(x1)
            preds = BE(x2)
            loss = loss_fn(preds, target)
            opt.zero_grad(); loss.backward(); opt.step()
            preds_lbl = torch.argmax(preds, dim=1)
            tcorrect += (preds_lbl == target).float().mean().item()
            tloss += loss.item()
            # compute per-batch IoU per class
            try:
                ious = jaccard_score(target.cpu().flatten(), preds_lbl.cpu().flatten(), average=None, labels=list(range(num_classes)))
            except Exception:
                ious = []
                tar = target.cpu().numpy().flatten()
                predf = preds_lbl.cpu().numpy().flatten()
                for c in range(num_classes):
                    inter = np.logical_and(tar == c, predf == c).sum()
                    union = np.logical_or(tar == c, predf == c).sum()
                    ious.append(inter/union if union>0 else 0.0)
                ious = np.array(ious)
            for i in range(num_classes):
                iou_c[i] += ious[i]
        acc = 100.0 * tcorrect / len(loader.dataset)
        avg_iou = [v / len(loader.dataset) for v in iou_c]
        print(f"Client {cid} | Epoch {ep+1} | Train Loss: {tloss:.4f} | Train Acc: {acc:.2f}% | "
              f"Train IoU w/bg: {sum(avg_iou)/num_classes:.4f} | "
              f"Train IoU no/bg: {sum(avg_iou[1:])/(num_classes-1):.4f}")

def evaluate(loader, FE, SS, BE, loss_fn, num_classes):
    total_loss, total_correct = 0.0, 0.0
    iou_c = [0.0] * num_classes
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(DEVICE), target.long().to(DEVICE)
            x1 = FE(data)
            x2 = SS(x1)
            preds = BE(x2)
            loss = loss_fn(preds, target)
            preds_lbl = torch.argmax(preds, dim=1)
            total_correct += (preds_lbl == target).float().mean().item()
            total_loss += loss.item()
            ious = jaccard_score(target.cpu().flatten(), preds_lbl.cpu().flatten(), average=None, labels=list(range(num_classes)))
            for i in range(num_classes):
                iou_c[i] += ious[i]
    N = len(loader.dataset)
    acc = 100.0 * total_correct / N
    avg_iou = [v / N for v in iou_c]
    print(f"Eval Loss: {total_loss / N:.4f} | Eval Acc: {acc:.2f}% | "
          f"Eval IoU w/bg: {sum(avg_iou)/num_classes:.4f} | "
          f"Eval IoU no/bg: {sum(avg_iou[1:])/(num_classes-1):.4f}")
    return total_loss / N, acc, avg_iou, avg_iou[1:]

# -----------------------
# Metric helpers (kept unchanged)
# -----------------------
def _mask_to_surface(mask):
    if mask.sum() == 0:
        return np.zeros_like(mask, dtype=bool)
    eroded = ndi.binary_erosion(mask, structure=np.ones((3, 3)))
    surface = mask.astype(bool) & (~eroded)
    return surface

def _surface_distances(mask_gt, mask_pred):
    if mask_gt.sum() == 0:
        return np.array([])
    pred_dist = ndi.distance_transform_edt(~mask_pred)
    gt_surface = _mask_to_surface(mask_gt)
    distances = pred_dist[gt_surface]
    return distances

def hd95_assd_for_pair(mask_gt, mask_pred):
    if mask_gt.sum() == 0 and mask_pred.sum() == 0:
        return 0.0, 0.0
    d_gt_to_pred = _surface_distances(mask_gt, mask_pred)
    d_pred_to_gt = _surface_distances(mask_pred, mask_gt)
    if d_gt_to_pred.size == 0:
        hd1 = 0.0; asd1 = 0.0
    else:
        hd1 = np.percentile(d_gt_to_pred, 95)
        asd1 = d_gt_to_pred.mean()
    if d_pred_to_gt.size == 0:
        hd2 = 0.0; asd2 = 0.0
    else:
        hd2 = np.percentile(d_pred_to_gt, 95)
        asd2 = d_pred_to_gt.mean()
    hd95 = max(hd1, hd2)
    assd = 0.5 * (asd1 + asd2)
    return float(hd95), float(assd)

def compute_segmentation_metrics_all(preds_lbl_np, target_np, num_classes):
    # (unchanged) --- keep your function as-is
    N = preds_lbl_np.shape[0]
    iou_per_class = np.zeros((num_classes,), dtype=float)
    dice_per_class = np.zeros((num_classes,), dtype=float)
    prec_per_class = np.zeros((num_classes,), dtype=float)
    rec_per_class = np.zeros((num_classes,), dtype=float)
    f1_per_class = np.zeros((num_classes,), dtype=float)
    hd95_per_class = np.zeros((num_classes,), dtype=float)
    assd_per_class = np.zeros((num_classes,), dtype=float)

    tp = np.zeros((num_classes,), dtype=float)
    fp = np.zeros((num_classes,), dtype=float)
    fn = np.zeros((num_classes,), dtype=float)

    try:
        ious_all = jaccard_score(target_np.flatten(), preds_lbl_np.flatten(), average=None, labels=list(range(num_classes)), zero_division=0)
    except Exception:
        ious_all = np.zeros((num_classes,), dtype=float)
        for c in range(num_classes):
            inter = np.logical_and(target_np == c, preds_lbl_np == c).sum()
            union = np.logical_or(target_np == c, preds_lbl_np == c).sum()
            ious_all[c] = inter / union if union > 0 else 0.0

    iou_per_class[:] = ious_all

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
            hd95_val, assd_val = hd95_assd_for_pair(gt_mask.astype(bool), pred_mask.astype(bool))
            hd95_per_class[c] += hd95_val
            assd_per_class[c] += assd_val

    for c in range(num_classes):
        precision = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) > 0 else 0.0
        recall = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        prec_per_class[c] = precision
        rec_per_class[c] = recall
        f1_per_class[c] = f1
        denom = (2.0 * tp[c] + fp[c] + fn[c])
        dice_per_class[c] = (2.0 * tp[c] / denom) if denom > 0 else 0.0
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

def save_pred_as_image(pred_label_np, orig_size, save_path):
    maxval = pred_label_np.max() if pred_label_np.max() > 0 else 1
    img = (pred_label_np.astype(np.float32) / float(maxval)) * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    pil = Image.fromarray(img)
    pil = pil.resize(orig_size, resample=Image.NEAREST)
    pil.save(save_path)

def try_get_filename_from_dataset(dataset, idx):
    candidates = ['img_files', 'image_paths', 'images', 'files', 'img_list', 'paths', 'filenames', 'file_list']
    for c in candidates:
        if hasattr(dataset, c):
            arr = getattr(dataset, c)
            try:
                elem = arr[idx]
                if isinstance(elem, (list, tuple)):
                    elem = elem[0]
                base = os.path.basename(elem)
                return base, elem
            except Exception:
                continue
    if hasattr(dataset, 'imgs'):
        try:
            elem = dataset.imgs[idx][0]
            base = os.path.basename(elem)
            return base, elem
        except Exception:
            pass
    try:
        sample = dataset[idx]
        if isinstance(sample, (list, tuple)) and len(sample) >= 3:
            fname = sample[2]
            if isinstance(fname, str):
                return os.path.basename(fname), fname
    except Exception:
        pass
    return None, None

def evaluate_and_save_predictions(loader, FE, SS, BE, loss_fn, num_classes, client_id, save_dir, save_preds=True):
    # (kept same as your original implementation - unchanged)
    FE.eval(); SS.eval(); BE.eval()
    if save_preds:
        os.makedirs(save_dir, exist_ok=True)

    total_loss = 0.0
    total_correct = 0.0
    N = 0

    tp = np.zeros((num_classes,), dtype=float)
    fp = np.zeros((num_classes,), dtype=float)
    fn = np.zeros((num_classes,), dtype=float)
    hd95_sum = np.zeros((num_classes,), dtype=float)
    assd_sum = np.zeros((num_classes,), dtype=float)
    inter_sum = np.zeros((num_classes,), dtype=float)
    union_sum = np.zeros((num_classes,), dtype=float)

    mse_sum1 = 0.0; psnr_sum1 = 0.0; ssim_sum1 = 0.0
    mse_sum2 = 0.0; psnr_sum2 = 0.0; ssim_sum2 = 0.0

    with torch.no_grad():
        dataset = loader.dataset
        for idx, batch in enumerate(tqdm(loader, leave=False)):
            if len(batch) == 3:
                data, target, fname = batch
            else:
                data, target = batch
                fname = None

            bsize = data.shape[0]
            data_cuda = data.to(DEVICE)
            target_cuda = target.long().to(DEVICE)

            x1 = FE(data_cuda)
            x2 = SS(x1)
            preds = BE(x2)

            loss = loss_fn(preds, target_cuda)
            preds_lbl = torch.argmax(preds, dim=1)

            preds_lbl_cpu = preds_lbl.cpu().numpy()
            preds_soft_cpu = torch.softmax(preds, dim=1).cpu().numpy()
            target_cpu = target.cpu().numpy()

            # free GPU tensors early
            del x1, x2, preds, data_cuda, target_cuda, preds_lbl
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            total_loss += float(loss.item())
            total_correct += float(((preds_lbl_cpu == target_cpu).mean()))
            for bi in range(bsize):
                N += 1
                basename, fullpath = try_get_filename_from_dataset(dataset, idx * loader.batch_size + bi)
                if basename is None and fname is not None:
                    if isinstance(fname, (list, tuple)):
                        basename = os.path.basename(fname[bi] if bi < len(fname) else fname[0])
                        fullpath = fname[bi] if bi < len(fname) else fname[0]
                    else:
                        basename = os.path.basename(fname)
                        fullpath = fname
                if basename is None:
                    basename = f"client{client_id}_idx{idx*loader.batch_size + bi}.png"
                    fullpath = None

                name_root, ext = os.path.splitext(basename)
                out_name = f"{name_root}_pred.png"
                out_path = os.path.join(save_dir, out_name) if save_preds else None

                orig_size = None
                if fullpath is not None and os.path.exists(fullpath):
                    try:
                        with Image.open(fullpath) as im:
                            orig_size = im.size
                    except Exception:
                        orig_size = None
                H_r, W_r = preds_lbl_cpu.shape[1], preds_lbl_cpu.shape[2]
                if orig_size is None:
                    orig_size = (W_r, H_r)

                if save_preds:
                    save_pred_as_image(preds_lbl_cpu[bi], orig_size, out_path)

                # segmentation accumulators
                gt = target_cpu[bi]
                pred_lbl = preds_lbl_cpu[bi]
                for c in range(num_classes):
                    gt_mask = (gt == c)
                    pred_mask = (pred_lbl == c)
                    inter = float((gt_mask & pred_mask).sum())
                    union = float((gt_mask | pred_mask).sum())
                    inter_sum[c] += inter
                    union_sum[c] += union
                    tp[c] += inter
                    p_area = float(pred_mask.sum())
                    g_area = float(gt_mask.sum())
                    fp[c] += max(0.0, p_area - inter)
                    fn[c] += max(0.0, g_area - inter)
                    hd95_val, assd_val = hd95_assd_for_pair(gt_mask.astype(bool), pred_mask.astype(bool))
                    hd95_sum[c] += hd95_val
                    assd_sum[c] += assd_val

                # reconstruction metrics: build clean image from data (CPU tensor)
                dat_np = data[bi].cpu().numpy() if isinstance(data, torch.Tensor) else np.array(data[bi])
                cmin, cmax = dat_np.min(), dat_np.max()
                if cmax > 2.0:
                    dat_np = dat_np / 255.0
                else:
                    dat_np = (dat_np - cmin) / (cmax - cmin + 1e-8)
                pil_clean = Image.fromarray((np.transpose(dat_np, (1,2,0)) * 255.0).astype(np.uint8))
                pil_clean = pil_clean.resize(orig_size, resample=Image.BILINEAR)
                clean_arr = np.array(pil_clean).astype(np.float32) / 255.0
                if clean_arr.ndim == 2:
                    clean_arr = np.stack([clean_arr]*3, axis=-1)
                if clean_arr.shape[2] == 4:
                    clean_arr = clean_arr[..., :3]
                clean_chw = np.transpose(clean_arr, (2,0,1))

                # split1: argmax -> grayscale -> 3ch
                pred_lbl_resized = Image.fromarray(pred_lbl.astype(np.uint8)).resize(orig_size, resample=Image.NEAREST)
                pred_lbl_arr = np.array(pred_lbl_resized).astype(np.float32)
                denom = float(max(1, pred_lbl_arr.max()))
                pred_rgb1 = (pred_lbl_arr / denom).astype(np.float32)
                pred_rgb1_3 = np.stack([pred_rgb1]*3, axis=-1)
                pred1_chw = np.transpose(pred_rgb1_3, (2,0,1))

                # split2: expectation from softmax -> continuous -> 3ch
                probs = preds_soft_cpu[bi]
                classes = np.arange(probs.shape[0]).reshape((-1,1,1))
                expected = (probs * classes).sum(axis=0)
                pil_expected = Image.fromarray(((expected / max(1.0, expected.max())) * 255.0).astype(np.uint8))
                pil_expected = pil_expected.resize(orig_size, resample=Image.BILINEAR)
                expected_arr = np.array(pil_expected).astype(np.float32) / 255.0
                expected_rgb3 = np.stack([expected_arr]*3, axis=-1)
                pred2_chw = np.transpose(expected_rgb3, (2,0,1))

                # per-sample metrics for split1
                mse1 = float(np.mean((clean_chw - pred1_chw)**2))
                try:
                    psnr1 = sk_psnr(clean_chw, pred1_chw, data_range=float(clean_chw.max() - clean_chw.min()) if clean_chw.max() - clean_chw.min() != 0 else 1.0)
                except Exception:
                    psnr1 = 0.0
                def ssim_per_sample(a, b):
                    ch_ssims = []
                    for ch in range(a.shape[0]):
                        try:
                            ch_ssim = sk_ssim(a[ch], b[ch], data_range=(a[ch].max() - a[ch].min()) if (a[ch].max() - a[ch].min()) != 0 else 1.0)
                        except Exception:
                            ch_ssim = 0.0
                        ch_ssims.append(ch_ssim)
                    return float(np.mean(ch_ssims))
                ssim1 = ssim_per_sample(clean_chw, pred1_chw)
                mse_sum1 += mse1
                psnr_sum1 += psnr1
                ssim_sum1 += ssim1

                # per-sample metrics for split2
                mse2 = float(np.mean((clean_chw - pred2_chw)**2))
                try:
                    psnr2 = sk_psnr(clean_chw, pred2_chw, data_range=float(clean_chw.max() - clean_chw.min()) if clean_chw.max() - clean_chw.min() != 0 else 1.0)
                except Exception:
                    psnr2 = 0.0
                ssim2 = ssim_per_sample(clean_chw, pred2_chw)
                mse_sum2 += mse2
                psnr_sum2 += psnr2
                ssim_sum2 += ssim2

                # free temporaries
                del clean_chw, pred1_chw, pred2_chw, pred_lbl_resized, pred_lbl_arr, pil_expected, expected_arr, dat_np
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    if N == 0:
        return None

    # finalize segmentation metrics
    iou_per_class = np.zeros((num_classes,), dtype=float)
    dice_per_class = np.zeros((num_classes,), dtype=float)
    precision_per_class = np.zeros((num_classes,), dtype=float)
    recall_per_class = np.zeros((num_classes,), dtype=float)
    f1_per_class = np.zeros((num_classes,), dtype=float)
    hd95_per_class = np.zeros((num_classes,), dtype=float)
    assd_per_class = np.zeros((num_classes,), dtype=float)

    for c in range(num_classes):
        if union_sum[c] > 0:
            iou_per_class[c] = inter_sum[c] / union_sum[c]
        else:
            iou_per_class[c] = 0.0
        precision = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) > 0 else 0.0
        recall = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        precision_per_class[c] = precision
        recall_per_class[c] = recall
        f1_per_class[c] = f1
        denom = (2.0 * tp[c] + fp[c] + fn[c])
        dice_per_class[c] = (2.0 * tp[c] / denom) if denom > 0 else 0.0
        hd95_per_class[c] = float(hd95_sum[c] / max(1, N))
        assd_per_class[c] = float(assd_sum[c] / max(1, N))

    seg_metrics = {
        "iou": iou_per_class,
        "dice": dice_per_class,
        "precision": precision_per_class,
        "recall": recall_per_class,
        "f1": f1_per_class,
        "hd95": hd95_per_class,
        "assd": assd_per_class
    }

    # finalize reconstruction metrics per split (no LPIPS, no FID)
    recon_split1 = {
        "mse": float(mse_sum1 / N),
        "psnr": float(psnr_sum1 / N),
        "ssim": float(ssim_sum1 / N)
    }
    recon_split2 = {
        "mse": float(mse_sum2 / N),
        "psnr": float(psnr_sum2 / N),
        "ssim": float(ssim_sum2 / N)
    }

    acc = 100.0 * total_correct / N
    avg_iou_wb = iou_per_class.tolist()
    avg_iou_nb = avg_iou_wb[1:] if num_classes > 1 else []

    print(f"Eval Loss: {total_loss / max(1, N):.4f} | Eval Acc: {acc:.2f}% | "
          f"Eval IoU w/bg: {float(np.mean(avg_iou_wb)):.4f} | "
          f"Eval IoU no/bg: {float(np.mean(avg_iou_nb)) if len(avg_iou_nb)>0 else 0.0:.4f}")

    return {
        "loss": total_loss / max(1, N),
        "acc": acc,
        "avg_iou_wb": avg_iou_wb,
        "avg_iou_nb": avg_iou_nb,
        "seg_metrics": seg_metrics,
        "recon_split1": recon_split1,
        "recon_split2": recon_split2
    }

# -----------------------
# MOCHA-like aggregation helpers
# -----------------------
def state_dict_to_cpu(sd):
    """Return a CPU copy of state_dict with detached tensors (float)."""
    cpu_sd = {}
    for k, v in sd.items():
        cpu_sd[k] = v.detach().cpu().clone()
    return cpu_sd

def compute_model_delta(global_sd, local_sd):
    """Compute local - global (on CPU) for each key; returns dict of tensors on CPU."""
    g_cpu = state_dict_to_cpu(global_sd)
    l_cpu = state_dict_to_cpu(local_sd)
    delta = {}
    for k in g_cpu.keys():
        # ensure shapes match
        if k in l_cpu and l_cpu[k].shape == g_cpu[k].shape:
            delta[k] = (l_cpu[k] - g_cpu[k]).float()
        else:
            # fallback: zero
            delta[k] = torch.zeros_like(g_cpu[k]).float()
    return delta

def aggregate_deltas(deltas, norm_weights, thetas, gamma=1.0):
    """
    MOCHA-inspired aggregation of deltas:
      aggregated_delta[k] = gamma * sum_i norm_weights[i] * (1 - theta[i]) * delta_i[k]
    where theta in [0,1] is per-client approximation (1 -> drop).
    deltas: list of dicts (CPU tensors)
    norm_weights: list normalized by total size
    thetas: list of floats in [0,1]
    """
    assert len(deltas) == len(norm_weights) == len(thetas)
    aggregated = {}
    keys = deltas[0].keys()
    for k in keys:
        agg = None
        for i in range(len(deltas)):
            coeff = norm_weights[i] * (1.0 - float(thetas[i]))
            if coeff == 0.0:
                continue
            contrib = deltas[i][k].float() * coeff
            if agg is None:
                agg = contrib.clone()
            else:
                agg += contrib
        if agg is None:
            # no clients contributed -> zero
            aggregated[k] = torch.zeros_like(deltas[0][k])
        else:
            aggregated[k] = gamma * agg
    return aggregated

def apply_delta_to_state(global_sd, aggregated_delta):
    """Return a new state dict where global + aggregated_delta is computed (on CPU)."""
    new_sd = {}
    g_cpu = state_dict_to_cpu(global_sd)
    for k in g_cpu.keys():
        if k in aggregated_delta:
            new_sd[k] = (g_cpu[k] + aggregated_delta[k]).clone()
        else:
            new_sd[k] = g_cpu[k].clone()
    return new_sd

# -----------------------
# plotting helper (kept same)
# -----------------------
def plot_iou_curves(round_num):
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
    plt.savefig("Outputs/MTS1BS_iou_wbg.png")
    plt.close()

    plt.figure(figsize=(10, 5))
    for i in range(NUM_CLIENTS):
        plt.plot(rounds, test_iou_nbg_all[i], label=dataset_names[i])
    plt.xlabel("Communication Round")
    plt.ylabel("IoU n/b")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Outputs/MTS1BS_iou_nobg_curve.png")
    plt.close()

# -----------------------
# main: integrated MOCHA-style
# -----------------------
def main():
    task_info = {
        0: {"name": "Blastocyst", "num_classes": 5, "path": DATA_PATH, "dataset": EmbryoDataset},
        1: {"name": "HAM10K", "num_classes": 2, "path": DATA_PATH, "dataset": HAMDataset},
        2: {"name": "Fetal", "num_classes": 3, "path": DATA_PATH, "dataset": FHPsAOPMSBDataset},
        3: {"name": "Mosmed", "num_classes": 2, "path": DATA_PATH, "dataset": covidCTDataset},
        4: {"name": "Kvasir", "num_classes": 2, "path": DATA_PATH, "dataset": CVCDataset}
    }

    tr_tf = A.Compose([A.Resize(224, 224), A.Normalize(mean=[0]*3, std=[1]*3, max_pixel_value=255.0), ToTensorV2()])
    val_tf = test_tf = tr_tf

    # initialize global server split (SS)
    global_SS = UNET_server(in_channels=32).to(DEVICE)

    best_loss = float('inf')

    # MOCHA hyperparams (practical choices)
    gamma = 1.0  # aggregation multiplier (MOCHA uses gamma in [0,1], we use 1.0 by default)
    # You can tune theta behavior to simulate stragglers: here it's configurable per-round/per-client
    theta_schedule = None  # if None, we'll simulate small random thetas each round; set to a list of lists to control exactly

    for r in range(COMM_ROUNDS):
        print(f"\n[Comm Round {r+1}/{COMM_ROUNDS}]")
        local_SS_list = []
        client_FEs = []
        client_BEs = []
        weights = []
        total_sz = 0

        # optional: set theta for this round (simulate)
        if theta_schedule is not None and r < len(theta_schedule):
            thetas = list(theta_schedule[r])
            if len(thetas) != NUM_CLIENTS:
                raise ValueError("theta_schedule round entry must have NUM_CLIENTS values")
        else:
            # by default simulate mild stragglers: random theta in [0.0, 0.3], occasionally drop someone
            thetas = []
            for i in range(NUM_CLIENTS):
                p_drop = 0.05  # 5% chance to drop entirely this round
                if random.random() < p_drop:
                    thetas.append(1.0)
                else:
                    thetas.append(float(np.clip(np.random.beta(2, 20), 0.0, 0.6)))  # most values small

        deltas = []
        client_sizes = []

        # Run local training on each client (parallelism is not implemented; loop sequentially)
        for i in range(NUM_CLIENTS):
            task = task_info[i]
            num_classes, path = task["num_classes"], task["path"]
            # instantiate fresh FE, copy global SS into local SS (MOCHA provides local solver flexibility)
            FE = UNET_FE(in_channels=3).to(DEVICE)
            SS = UNET_server(in_channels=32).to(DEVICE)
            BE = UNET_BE(out_channels=num_classes).to(DEVICE)
            # load global SS weights into local SS to begin local solve
            SS.load_state_dict(global_SS.state_dict())
            opt = optim.AdamW(list(FE.parameters()) + list(SS.parameters()) + list(BE.parameters()), lr=1e-4)
            loss_fn = ComboLoss(num_classes)
            ds_class = task["dataset"]
            train_loader = get_loader(os.path.join(path, f"client{i + 1}/train_imgs"),
                                      os.path.join(path, f"client{i + 1}/train_masks"),
                                      ds_class, tr_tf)
            val_loader = get_loader(os.path.join(path, f"client{i + 1}/val_imgs"),
                                    os.path.join(path, f"client{i + 1}/val_masks"),
                                    ds_class, val_tf)

            # Local training (your original function)
            train_local(train_loader, FE, SS, BE, opt, loss_fn, i + 1, num_classes)
            evaluate(val_loader, FE, SS, BE, loss_fn, num_classes)

            # collect local copies
            client_FEs.append(FE)
            local_SS_list.append(SS)
            client_BEs.append(BE)

            sz = len(train_loader.dataset)
            client_sizes.append(sz)
            weights.append(sz)
            total_sz += sz

        # normalize weights as you already did
        if total_sz == 0:
            norm_weights = [1.0 / NUM_CLIENTS] * NUM_CLIENTS
        else:
            norm_weights = [w / total_sz for w in weights]

        # Build deltas (local_SS - global_SS) for each client (on CPU)
        global_sd = global_SS.state_dict()
        for ss in local_SS_list:
            local_sd = ss.state_dict()
            delta = compute_model_delta(global_sd, local_sd)
            deltas.append(delta)

        # Aggregate deltas using MOCHA-style per-client thetas
        aggregated_delta = aggregate_deltas(deltas, norm_weights, thetas, gamma=gamma)

        # Apply aggregated delta to global state dict
        new_global_cpu_sd = apply_delta_to_state(global_sd, aggregated_delta)

        # Load new global state into global_SS (move to DEVICE)
        # Ensure dtype/device match
        new_global_sd = {}
        for k, v in new_global_cpu_sd.items():
            new_global_sd[k] = v.to(DEVICE)
        global_SS.load_state_dict(new_global_sd)

        # Global validation: evaluate each client FE + updated global_SS + BE on each client's val set
        total_val_loss = 0.0
        total_val_ious = [0.0] * 5  # aggregated across clients into 5-length vector (max classes 5)
        for i in range(NUM_CLIENTS):
            task = task_info[i]
            num_classes, path = task["num_classes"], task["path"]
            ds_class = task["dataset"]

            val_loader = get_loader(os.path.join(path, f"client{i + 1}/val_imgs"),
                                    os.path.join(path, f"client{i + 1}/val_masks"),
                                    ds_class, val_tf)
            loss_fn = ComboLoss(num_classes)
            # use client's FE and BE, but global SS (updated) as the server split
            val_loss, val_acc, val_iou_wb, val_iou_nb = evaluate(val_loader, client_FEs[i], global_SS, client_BEs[i], loss_fn, num_classes)
            print(f"[Client {i + 1} Global Validation] Loss: {val_loss:.4f} | Acc: {val_acc:.2f}% | IoU wb: {sum(val_iou_wb) / num_classes:.4f} | IoU nb: {sum(val_iou_nb) / num_classes if len(val_iou_nb)>0 else 0.0:.4f}")
            total_val_loss += val_loss
            # accumulate into vector; handle different num_classes by filling prefix
            for j in range(min(len(val_iou_wb), len(total_val_ious))):
                total_val_ious[j] += val_iou_wb[j]

        avg_val_loss = total_val_loss / NUM_CLIENTS
        avg_ious = [v / NUM_CLIENTS for v in total_val_ious]
        print(f"[Global Validation] Loss: {avg_val_loss:.4f} | IoU w/bg: {sum(avg_ious) / 5:.4f} | IoU no/bg: {sum(avg_ious[1:]) / 4:.4f}")

        # Save best global server split + client models
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            os.makedirs("BestModels", exist_ok=True)
            torch.save(global_SS.state_dict(), "BestModels/best_SS.pth")
            for i in range(NUM_CLIENTS):
                torch.save(client_FEs[i].state_dict(), f"BestModels/best_FE_client{i + 1}.pth")
                torch.save(client_BEs[i].state_dict(), f"BestModels/best_BE_client{i + 1}.pth")
            print("[Best Global and Client BE Models Saved]")

        # TESTING: evaluate, save preds (overwrite previous round's preds), compute segmentation & recon metrics
        for i in range(NUM_CLIENTS):
            task = task_info[i]
            num_classes, path = task["num_classes"], task["path"]
            ds_class = task["dataset"]

            test_loader = get_loader(os.path.join(path, f"client{i + 1}/test_imgs"),
                                     os.path.join(path, f"client{i + 1}/test_masks"),
                                     ds_class, test_tf)
            loss_fn = ComboLoss(num_classes)

            # Fixed save_dir per client so files are overwritten each round
            save_dir = os.path.join("Outputs/test_preds_BS", f"client{i+1}_preds")
            results = evaluate_and_save_predictions(
                test_loader,
                client_FEs[i],
                global_SS,
                client_BEs[i],
                loss_fn,
                num_classes,
                i+1,
                save_dir,
                save_preds=True
            )

            if results is None:
                print(f"[Client {i+1} Testing] No test predictions (empty loader).")
                continue

            test_loss = results["loss"]
            test_acc = results["acc"]
            test_iou_wb = results["avg_iou_wb"]
            test_iou_nb = results["avg_iou_nb"]
            print(f"[Client {i + 1} Testing] Loss: {test_loss:.4f} | Acc: {test_acc:.2f}% | "
                  f"IoU wb: {float(np.mean(test_iou_wb)):.4f} | "
                  f"IoU nb: {float(np.mean(test_iou_nb)) if len(test_iou_nb)>0 else 0.0:.4f}")

            # append IoUs for plotting
            test_iou_wbg_all[i].append(float(np.mean(test_iou_wb)))
            if len(test_iou_nb) > 0:
                test_iou_nbg_all[i].append(float(np.mean(test_iou_nb)))
            else:
                test_iou_nbg_all[i].append(0.0)

            # print detailed segmentation per-class metrics
            seg = results["seg_metrics"]
            print(f"[Client {i+1}] Segmentation per-class IoU: {seg['iou']}")
            print(f"[Client {i+1}] Segmentation per-class Dice: {seg['dice']}")
            print(f"[Client {i+1}] Segmentation precision: {seg['precision']}")
            print(f"[Client {i+1}] Segmentation recall: {seg['recall']}")
            print(f"[Client {i+1}] Segmentation F1: {seg['f1']}")
            print(f"[Client {i+1}] Segmentation HD95 per class: {seg['hd95']}")
            print(f"[Client {i+1}] Segmentation ASSD per class: {seg['assd']}")

            mean_iou_wbg = float(np.mean(seg['iou']))
            mean_iou_nbg = float(np.mean(seg['iou'][1:])) if num_classes > 1 else mean_iou_wbg
            mean_dice = float(np.mean(seg['dice']))
            mean_P = float(np.mean(seg['precision']))
            mean_R = float(np.mean(seg['recall']))
            mean_f1 = float(np.mean(seg['f1']))
            mean_HD95 = float(np.mean(seg['hd95']))
            mean_ASSD = float(np.mean(seg['assd']))

            print(f"Summary: mean IoU (w/bg)={mean_iou_wbg:.4f}, mean IoU (no/bg)={mean_iou_nbg:.4f}, "
                  f"mean Dice={mean_dice:.4f}, mean P={mean_P:.4f}, mean R={mean_R:.4f}, mean F1={mean_f1:.4f}, "
                  f"mean hd95={mean_HD95:.4f}, mean ASSD={mean_ASSD:.4f}")

            # print reconstruction metrics separately for split1 and split2 (no LPIPS/FID)
            recon1 = results["recon_split1"]
            recon2 = results["recon_split2"]
            print(f"[Client {i+1}] Reconstruction metrics (split1 - argmax->grayscale): {recon1}")
            print(f"[Client {i+1}] Reconstruction metrics (split2 - softmax expectation): {recon2}")

        plot_iou_curves(r + 1)

if __name__ == "__main__":
    main()
