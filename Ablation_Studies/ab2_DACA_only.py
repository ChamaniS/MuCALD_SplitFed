# =========================
# MuCALD-SplitFed Ablation:
# DACA block only
# Corrected for FE returning tuple output
# =========================

import os
import sys
import copy
from collections import defaultdict

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Function
from torch.utils.data import DataLoader, Subset

import albumentations as A
from albumentations.pytorch import ToTensorV2

import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from sklearn.metrics import jaccard_score
from tqdm import tqdm
from scipy import ndimage as ndi

from models.clientmodel_FE import UNET_FE
from models.servermodel import UNET_server
from models.clientmodel_BE import UNET_BE
from dataset import EmbryoDataset, HAMDataset, CVCDataset, covidCTDataset, FHPsAOPMSBDataset
from dataset_wrappers import WithFilenames

# =========================
# Basic setup
# =========================
CUDA_LAUNCH_BLOCKING = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_CLIENTS = 5
LOCAL_EPOCHS =1
COMM_ROUNDS = 24

LAMBDA_ADV = 0.10

PER_CLIENT_ADV_SCALE = {
    1: 1.0,   # Blastocyst
    2: 0.4,   # HAM10K
    3: 1.0,   # Fetal
    4: 1.8,   # MosMed
    5: 1.0    # Kvasir
}

DATA_PATH = "/xxxx/MTS2"
BASE_OUT = "/xxxx/"
OUTPUT_LOG = os.path.join(BASE_OUT, "mucald_final_unet_daca_ablation.txt")
PROD_OUT_DIR = os.path.join(BASE_OUT, "Outputs")
BEST_DIR = os.path.join(BASE_OUT, "BestModels")

os.makedirs(PROD_OUT_DIR, exist_ok=True)
os.makedirs(BEST_DIR, exist_ok=True)

sys.stdout = open(OUTPUT_LOG, "w")

# =========================
# Helpers
# =========================

def unwrap_output(x):
    """
    UNET_FE in your code returns a tuple: (feature, aux_output)
    This helper safely extracts the tensor we need.
    """
    if isinstance(x, (tuple, list)):
        return x[0]
    return x


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
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, num_domains)
        )

    def forward(self, x):
        return self.net(x)


class ComboLoss(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.dice = smp.losses.DiceLoss(smp.losses.MULTICLASS_MODE, from_logits=True)
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        return self.dice(logits, targets) + 0.5 * self.ce(logits, targets)


def get_transforms(task_name):
    return A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=[0] * 3, std=[1] * 3, max_pixel_value=255.0),
        ToTensorV2()
    ])


def get_loader(img_dir, mask_dir, dataset_class, transform,
               batch_size=1, num_workers=1, with_names=True,
               shuffle=True, fraction=1.0):
    base = dataset_class(img_dir, mask_dir, transform=transform)
    ds = WithFilenames(base, img_dir) if with_names else base
    if fraction < 1.0:
        n = int(len(ds) * fraction)
        ds = Subset(ds, list(range(n)))
    return DataLoader(ds, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)


def _try_resolve_original_path(loader, fname):
    if os.path.isabs(fname) and os.path.exists(fname):
        return fname
    if os.path.exists(fname):
        return fname

    ds = loader.dataset
    if hasattr(ds, "dataset"):
        ds = ds.dataset

    candidate_dirs = []
    for attr in ("root", "img_dir", "image_dir", "base_dir", "data_dir", "images_dir"):
        base = getattr(ds, attr, None)
        if base:
            candidate_dirs.append(base)

    for d in candidate_dirs:
        c1 = os.path.join(d, fname)
        c2 = os.path.join(d, os.path.basename(fname))
        if os.path.exists(c1):
            return c1
        if os.path.exists(c2):
            return c2

    return None


def make_grad_tracker(model):
    return {
        name: torch.zeros_like(p, device=p.device)
        for name, p in model.named_parameters() if p.requires_grad
    }


# =========================
# Segmentation metrics
# =========================

def _mask_to_surface(mask):
    if mask.sum() == 0:
        return np.zeros_like(mask, dtype=bool)
    eroded = ndi.binary_erosion(mask, structure=np.ones((3, 3)))
    return mask.astype(bool) & (~eroded)


def _surface_distances(mask_gt, mask_pred):
    if mask_gt.sum() == 0:
        return np.array([])
    pred_dist = ndi.distance_transform_edt(~mask_pred)
    gt_surface = _mask_to_surface(mask_gt)
    return pred_dist[gt_surface]


def hd95_assd_for_pair(mask_gt, mask_pred):
    if mask_gt.sum() == 0 and mask_pred.sum() == 0:
        return 0.0, 0.0

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

    return float(max(hd1, hd2)), float(0.5 * (asd1 + asd2))


def compute_segmentation_metrics_all(preds_lbl_np, target_np, num_classes):
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
        ious_all = jaccard_score(
            target_np.flatten(),
            preds_lbl_np.flatten(),
            average=None,
            labels=list(range(num_classes)),
            zero_division=0
        )
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

        denom = 2.0 * tp[c] + fp[c] + fn[c]
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


# =========================
# Aggregation
# =========================

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


def causal_invariant_fusion(models, proxy_mse_scores, client_sizes=None, val_losses=None,
                            eps=1e-8, min_fraction=0.25, blend_equal=0.5):
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
        weights = [(1 - blend_equal) * w + blend_equal * ew for w, ew in zip(weights, equal_w)]

    print(f"Aggregation weights: {weights}")
    for key in ref_sd.keys():
        out[key] = sum(sd[key] * w for sd, w in zip(model_sds, weights))
    return out


# =========================
# Curves
# =========================

test_iou_wbg_all = {i: [] for i in range(NUM_CLIENTS)}
test_iou_nbg_all = {i: [] for i in range(NUM_CLIENTS)}


def plot_curves(round_num):
    dataset_names = ["Blastocysts", "HAM10K", "Fetus", "MosMed", "Kvasir"]
    rounds = list(range(1, round_num + 1))

    plt.figure(figsize=(10, 5))
    for i in range(NUM_CLIENTS):
        plt.plot(rounds, test_iou_wbg_all[i], label=dataset_names[i])
    plt.xlabel("Communication Round")
    plt.ylabel("IoU w/bg")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PROD_OUT_DIR, "mucald_daca_ablation_iou_wbg.png"))
    plt.close()

    plt.figure(figsize=(10, 5))
    for i in range(NUM_CLIENTS):
        plt.plot(rounds, test_iou_nbg_all[i], label=dataset_names[i])
    plt.xlabel("Communication Round")
    plt.ylabel("IoU n/bg")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PROD_OUT_DIR, "mucald_daca_ablation_iou_nbg.png"))
    plt.close()


# =========================
# Train / Eval
# =========================

def train_local(loader, FE, SS, BE, dom_disc, opt, dom_opt, loss_fn, cid, num_classes, task_name,
                mask_thresh=1e-5, client_size=None, max_size=None):
    FE.train()
    SS.train()
    BE.train()
    dom_disc.train()

    ss_grad = make_grad_tracker(SS)
    num_updates = 0

    for ep in range(LOCAL_EPOCHS):
        lam_adv = 0.0 if ep < 2 else LAMBDA_ADV
        lam_adv = lam_adv * PER_CLIENT_ADV_SCALE.get(cid, 1.0)

        if client_size is not None and max_size is not None:
            seg_weight = max_size / float(client_size)
            adv_weight = client_size / float(max_size)
        else:
            seg_weight = 1.0
            adv_weight = 1.0

        print(f"[Sched] ep={ep+1}/{LOCAL_EPOCHS} | adv={lam_adv:.3g}")

        tloss = 0.0
        tcorrect = 0.0
        num_batches = 0
        iou_c = [0.0] * num_classes

        for batch in tqdm(loader, leave=False):
            if len(batch) == 3:
                data, target, fnames = batch
            else:
                data, target = batch

            data = data.to(DEVICE)
            target = target.long().to(DEVICE)
            B = data.size(0)

            domain_labels = torch.full((B,), cid - 1, dtype=torch.long, device=DEVICE)

            feat_raw = FE(data)
            feat = unwrap_output(feat_raw)   # FIX: handle tuple output from FE
            ss_feat = SS(feat)
            preds = BE(ss_feat)

            seg_loss = loss_fn(preds, target)

            adv_logits = dom_disc(grad_reverse(ss_feat))
            adv_loss = F.cross_entropy(adv_logits, domain_labels)

            dom_logits = dom_disc(ss_feat.detach())
            dom_loss = F.cross_entropy(dom_logits, domain_labels)

            loss = seg_weight * seg_loss + (adv_weight * lam_adv) * adv_loss

            opt.zero_grad()
            loss.backward()

            with torch.no_grad():
                for name, p in SS.named_parameters():
                    if p.grad is not None:
                        ss_grad[name].add_(p.grad.abs())

            opt.step()

            dom_opt.zero_grad()
            dom_loss.backward()
            dom_opt.step()

            num_updates += 1
            num_batches += 1

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

        print(
            f"Client {cid} | Epoch {ep + 1} | "
            f"Train Loss: {(tloss / max(num_batches, 1)):.4f} | Acc: {acc:.2f}% | "
            f"IoU w/bg: {sum(avg_iou) / num_classes:.4f} | "
            f"IoU no/bg: {sum(avg_iou[1:]) / (num_classes - 1):.4f}"
        )

    ss_mask = {name: (g / max(num_updates, 1) > mask_thresh).float() for name, g in ss_grad.items()}
    return ss_mask


@torch.no_grad()
def evaluate(loader, FE, SS, BE, loss_fn, num_classes):
    total_loss = 0.0
    total_correct = 0.0
    iou_c = [0.0] * num_classes
    num_batches = 0

    for batch in loader:
        if len(batch) == 3:
            data, target, fnames = batch
        else:
            data, target = batch

        data = data.to(DEVICE)
        target = target.long().to(DEVICE)

        feat_raw = FE(data)
        feat = unwrap_output(feat_raw)   # FIX
        preds = BE(SS(feat))
        seg_loss = loss_fn(preds, target)

        preds_lbl = torch.argmax(preds, dim=1)
        total_correct += (preds_lbl == target).float().mean().item()
        total_loss += float(seg_loss.item())

        ious = jaccard_score(
            target.detach().cpu().flatten(),
            preds_lbl.detach().cpu().flatten(),
            average=None,
            labels=list(range(num_classes)),
            zero_division=0
        )
        for i in range(num_classes):
            iou_c[i] += float(ious[i])

        num_batches += 1

    if num_batches == 0:
        num_batches = 1

    acc = 100.0 * (total_correct / num_batches)
    avg_iou = [v / num_batches for v in iou_c]

    print(
        f"Eval Loss: {total_loss / num_batches:.4f} | Eval Acc: {acc:.2f}% | "
        f"IoU w/bg: {sum(avg_iou) / num_classes:.4f} | "
        f"IoU no/bg: {sum(avg_iou[1:]) / (num_classes - 1):.4f}"
    )

    return total_loss / num_batches, acc, avg_iou, avg_iou[1:]


@torch.no_grad()
def save_test_images(loader, FE, SS, BE, out_dir_root="Outputs/unet_mucald_test_preds", cid=1):
    out_dir = os.path.join(out_dir_root, f"client{cid}")
    os.makedirs(out_dir, exist_ok=True)

    for batch in loader:
        if len(batch) == 3:
            data, target, fnames = batch
            fnames = list(fnames)
        else:
            data, target = batch
            fnames = [str(i) for i in range(data.size(0))]

        data = data.to(DEVICE)
        feat_raw = FE(data)
        feat = unwrap_output(feat_raw)   # FIX
        preds = BE(SS(feat))
        preds_lbl = torch.argmax(preds, dim=1).cpu().numpy()

        for b in range(preds_lbl.shape[0]):
            fname = fnames[b]
            base = os.path.basename(fname)
            name_noext = os.path.splitext(base)[0]
            out_path = os.path.join(out_dir, f"{name_noext}_pred.png")

            arr = preds_lbl[b].astype(np.uint8)
            if arr.max() > 1:
                scale = max(1, 255 // max(1, int(arr.max())))
                vis = (arr * scale).astype(np.uint8)
            else:
                vis = (arr * 255).astype(np.uint8)

            img = Image.fromarray(vis).convert("L")
            orig_path = _try_resolve_original_path(loader, fname)
            if orig_path is not None:
                try:
                    orig_img = Image.open(orig_path)
                    img = img.resize(orig_img.size, resample=Image.NEAREST)
                except Exception as e:
                    print(f"[save_test_images] Could not resize using {orig_path}: {e}")

            img.save(out_path)


@torch.no_grad()
def compute_full_metrics(loader, FE, SS, BE, num_classes, device=DEVICE):
    preds_all = []
    targets_all = []

    for batch in loader:
        if len(batch) == 3:
            data, target, fnames = batch
        else:
            data, target = batch

        data = data.to(device)
        target = target.long().to(device)

        feat_raw = FE(data)
        feat = unwrap_output(feat_raw)   # FIX
        preds = BE(SS(feat))
        preds_lbl = torch.argmax(preds, dim=1).cpu().numpy()
        targets_np = target.cpu().numpy()

        preds_all.append(preds_lbl)
        targets_all.append(targets_np)

    preds_all_np = np.concatenate(preds_all, axis=0) if len(preds_all) > 0 else np.zeros((0, 256, 256), dtype=np.int32)
    targets_all_np = np.concatenate(targets_all, axis=0) if len(targets_all) > 0 else np.zeros((0, 256, 256), dtype=np.int32)

    seg_metrics = compute_segmentation_metrics_all(preds_all_np, targets_all_np, num_classes)

    return {
        "seg": seg_metrics,
        "n_samples": int(preds_all_np.shape[0])
    }


# =========================
# Main
# =========================

def main():
    task_info = {
        0: {
            "name": "Blastocyst",
            "num_classes": 5,
            "path": DATA_PATH,
            "dataset": EmbryoDataset
        },
        1: {
            "name": "HAM10K",
            "num_classes": 2,
            "path": DATA_PATH,
            "dataset": HAMDataset
        },
        2: {
            "name": "Fetal",
            "num_classes": 3,
            "path": DATA_PATH,
            "dataset": FHPsAOPMSBDataset
        },
        3: {
            "name": "Mosmed",
            "num_classes": 2,
            "path": DATA_PATH,
            "dataset": covidCTDataset
        },
        4: {
            "name": "Kvasir",
            "num_classes": 2,
            "path": DATA_PATH,
            "dataset": CVCDataset
        }
    }

    global_SS = UNET_server(in_channels=32).to(DEVICE)

    client_FE = [None] * NUM_CLIENTS
    client_BE = [None] * NUM_CLIENTS
    client_dom = [None] * NUM_CLIENTS

    best_loss = float("inf")

    for r in range(COMM_ROUNDS):
        print(f"\n[Comm Round {r + 1}/{COMM_ROUNDS}]")

        client_sizes = []
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

        local_SS = []
        client_FEs_round = []
        client_BEs_round = []

        for i in range(NUM_CLIENTS):
            task = task_info[i]
            task_name = task["name"]
            num_classes = task["num_classes"]
            path = task["path"]
            ds_class = task["dataset"]

            if client_FE[i] is None:
                FE = UNET_FE(in_channels=3).to(DEVICE)
                BE = UNET_BE(out_channels=num_classes).to(DEVICE)
                client_FE[i], client_BE[i] = FE, BE
            else:
                FE = client_FE[i].to(DEVICE)
                BE = client_BE[i].to(DEVICE)

            SS = copy.deepcopy(global_SS)

            if client_dom[i] is None:
                dom_disc = DomainDiscriminator().to(DEVICE)
                client_dom[i] = dom_disc
            else:
                dom_disc = client_dom[i].to(DEVICE)

            opt = optim.AdamW(
                list(FE.parameters()) + list(SS.parameters()) + list(BE.parameters()),
                lr=1e-4
            )
            dom_opt = optim.Adam(dom_disc.parameters(), lr=1e-4)
            loss_fn = ComboLoss(num_classes)

            tr_tf = get_transforms(task_name)
            val_tf = get_transforms(task_name)

            train_loader = get_loader(
                os.path.join(path, f"client{i + 1}/train_imgs"),
                os.path.join(path, f"client{i + 1}/train_masks"),
                ds_class, tr_tf,
                with_names=True, shuffle=True, fraction=1.0
            )
            val_loader = get_loader(
                os.path.join(path, f"client{i + 1}/val_imgs"),
                os.path.join(path, f"client{i + 1}/val_masks"),
                ds_class, val_tf,
                with_names=True, shuffle=False, fraction=1.0
            )
            test_loader = get_loader(
                os.path.join(path, f"client{i + 1}/test_imgs"),
                os.path.join(path, f"client{i + 1}/test_masks"),
                ds_class, val_tf,
                with_names=True, shuffle=False, fraction=1.0
            )

            train_local(
                train_loader,
                FE, SS, BE, dom_disc,
                opt, dom_opt, loss_fn,
                i + 1, num_classes, task_name,
                client_size=len(train_loader.dataset),
                max_size=max_size
            )

            val_loss, val_acc, val_iou_wb, val_iou_nb = evaluate(
                val_loader, FE, SS, BE, loss_fn, num_classes
            )

            client_FE[i], client_BE[i] = FE, BE
            client_FEs_round.append(FE)
            client_BEs_round.append(BE)
            local_SS.append(SS)

        # Aggregation: only SS in this DACA-only ablation
        global_SS.load_state_dict(equal_weight_fusion(local_SS))

        # Global validation
        total_val_loss, total_val_ious = 0.0, [0.0] * 5
        for i in range(NUM_CLIENTS):
            task = task_info[i]
            task_name = task["name"]
            num_classes = task["num_classes"]
            ds_class = task["dataset"]
            path = task["path"]

            val_tf = get_transforms(task_name)
            val_loader = get_loader(
                os.path.join(path, f"client{i + 1}/val_imgs"),
                os.path.join(path, f"client{i + 1}/val_masks"),
                ds_class, val_tf,
                with_names=True, shuffle=False, fraction=1.0
            )
            loss_fn = ComboLoss(num_classes)

            val_loss, val_acc, val_iou_wb, val_iou_nb = evaluate(
                val_loader,
                client_FEs_round[i], global_SS, client_BEs_round[i],
                loss_fn, num_classes
            )

            total_val_loss += val_loss
            for j in range(num_classes):
                total_val_ious[j] += val_iou_wb[j]

        avg_val_loss = total_val_loss / NUM_CLIENTS
        avg_ious = [v / NUM_CLIENTS for v in total_val_ious]
        print(f"[Global Validation] Loss: {avg_val_loss:.4f} | IoU w/bg: {sum(avg_ious) / 5:.4f}")

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            #torch.save(global_SS.state_dict(), os.path.join(BEST_DIR, "MTS5_best_DACAonly_SS.pth"))
            save_path = os.path.join(BEST_DIR, "MTS5_best_DACAonly_SS.pth")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(global_SS.state_dict(), save_path)
            print(f"[Best Global SS Model Saved] in {save_path}")

        # Test and detailed metrics
        for i in range(NUM_CLIENTS):
            task = task_info[i]
            task_name = task["name"]
            num_classes = task["num_classes"]
            ds_class = task["dataset"]
            path = task["path"]

            test_tf = get_transforms(task_name)
            test_loader = get_loader(
                os.path.join(path, f"client{i + 1}/test_imgs"),
                os.path.join(path, f"client{i + 1}/test_masks"),
                ds_class, test_tf,
                with_names=True, shuffle=False, fraction=1.0
            )
            loss_fn = ComboLoss(num_classes)

            test_loss, test_acc, test_iou_wb, test_iou_nb = evaluate(
                test_loader,
                client_FEs_round[i], global_SS, client_BEs_round[i],
                loss_fn, num_classes
            )

            print(f"[Client {i + 1} Testing] Loss: {test_loss:.4f} | IoU wb: {sum(test_iou_wb) / num_classes:.4f}")
            test_iou_wbg_all[i].append(sum(test_iou_wb) / num_classes)
            test_iou_nbg_all[i].append(sum(test_iou_nb) / (num_classes - 1))

            save_test_images(
                test_loader,
                client_FEs_round[i],
                global_SS,
                client_BEs_round[i],
                out_dir_root="Outputs/unet_mucald_test_preds_DACAonly",
                cid=i + 1
            )

            print(f"[Client {i + 1}] Computing detailed metrics...")
            metrics = compute_full_metrics(
                test_loader,
                client_FEs_round[i], global_SS, client_BEs_round[i],
                num_classes=num_classes, device=DEVICE
            )

            seg = metrics["seg"]
            print(f"--- Client {i + 1} ({task_name}) Segmentation summary ---")
            for c in range(num_classes):
                print(
                    f"Class {c}: IoU={seg['iou'][c]:.4f} | Dice={seg['dice'][c]:.4f} | "
                    f"P={seg['precision'][c]:.4f} R={seg['recall'][c]:.4f} F1={seg['f1'][c]:.4f} | "
                    f"HD95={seg['hd95'][c]:.2f} px | ASSD={seg['assd'][c]:.3f} px"
                )

            mean_iou_wbg = float(np.mean(seg["iou"]))
            mean_iou_nbg = float(np.mean(seg["iou"][1:])) if num_classes > 1 else mean_iou_wbg
            mean_dice = float(np.mean(seg["dice"]))
            mean_P = float(np.mean(seg["precision"]))
            mean_R = float(np.mean(seg["recall"]))
            mean_f1 = float(np.mean(seg["f1"]))
            mean_HD95 = float(np.mean(seg["hd95"]))
            mean_ASSD = float(np.mean(seg["assd"]))

            print(
                f"Summary: mean IoU (w/bg)={mean_iou_wbg:.4f}, mean IoU (no/bg)={mean_iou_nbg:.4f}, "
                f"mean Dice={mean_dice:.4f}, mean P={mean_P:.4f}, mean R={mean_R:.4f}, "
                f"mean F1={mean_f1:.4f}, mean hd95={mean_HD95:.4f}, mean ASSD={mean_ASSD:.4f}"
            )

        plot_curves(r + 1)


if __name__ == "__main__":
    main()