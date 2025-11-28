# BS_FedBone_offload_cpu_server.py
# Copy-paste this file and run. Server-side SS runs on CPU to avoid single-GPU OOM.

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import math
import gc
import copy
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.checkpoint import checkpoint
from sklearn.metrics import jaccard_score
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
from PIL import Image
import scipy.ndimage as ndi

# local imports (adjust as needed)
from models.clientmodel_FE import UNET_FE
from models.servermodel import UNET_server
from models.clientmodel_BE import UNET_BE
from dataset import EmbryoDataset, HAMDataset, CVCDataset, covidCTDataset, FHPsAOPMSBDataset

import sys
output_file = "/lustre06/project/6008975/csj5/causalenv/BS_FedBone.txt"
sys.stdout = open(output_file, "w")


# ---------- CONFIG ----------
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
CPU = torch.device("cpu")

NUM_CLIENTS = 5
LOCAL_EPOCHS = 1          # start small for debug
COMM_ROUNDS = 12
DATA_PATH = "/lustre06/project/6008975/csj5/MTS2"
BATCH_SIZE = 1
NUM_WORKERS = 0           # cluster safety
USE_AMP = False           # disabled to avoid dtype mismatch when transferring tensors
USE_CHECKPOINT = True     # checkpoint server forward to reduce memory
CHECKPOINT_REENTRANT = False  # pass explicitly to checkpoint

# file logging optional
# output_file = "BS_FedBone_offload_cpu_server.txt"
# sys.stdout = open(output_file, "w")

# bookkeeping
test_iou_wbg_all = {i: [] for i in range(NUM_CLIENTS)}
test_iou_nbg_all = {i: [] for i in range(NUM_CLIENTS)}

# ---------- HELPERS ----------
def get_loader(img_dir, mask_dir, dataset_class, transform, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
    ds = dataset_class(img_dir, mask_dir, transform=transform)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

class ComboLoss(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.dice = smp.losses.DiceLoss(smp.losses.MULTICLASS_MODE, from_logits=True)
        self.ce = nn.CrossEntropyLoss()
    def forward(self, logits, targets):
        # logits: (B, C, H, W), targets: (B, H, W) integer classes
        # convert logits to CPU floats if necessary when computing loss on CPU later
        return self.dice(logits, targets) + 0.0 * self.ce(logits, targets)

# lightweight task adaptation module
class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Conv2d(in_channels, max(1, in_channels // 8), 1)
        self.key = nn.Conv2d(in_channels, max(1, in_channels // 8), 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
        B, C, H, W = x.shape
        q = self.query(x).view(B, -1, H*W)
        k = self.key(x).view(B, -1, H*W)
        v = self.value(x).view(B, -1, H*W)
        attn = torch.bmm(q.permute(0,2,1), k) / math.sqrt(q.shape[1] + 1e-8)
        attn = self.softmax(attn)
        out = torch.bmm(v, attn.permute(0,2,1))
        out = out.view(B, C, H, W)
        return self.gamma * out + x

class TaskAdaptation(nn.Module):
    def __init__(self, in_channels, hidden_channels=None):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = max(16, in_channels // 4)
        self.reduce = nn.Conv2d(in_channels, hidden_channels, 1)
        self.conv = nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1)
        self.act = nn.GELU()
        self.attn = SpatialSelfAttention(hidden_channels)
        self.expand = nn.Conv2d(hidden_channels, in_channels, 1)
    def forward(self, x):
        h = self.reduce(x)
        h = self.act(self.conv(h))
        h = self.attn(h)
        h = self.expand(h)
        return x + h

# GPAggregation helpers
def flatten_grad_dict(grad_dict):
    parts = []
    for k in sorted(grad_dict.keys()):
        g = grad_dict[k]
        if g is None:
            continue
        parts.append(g.detach().float().view(-1).cpu())
    if len(parts) == 0:
        return torch.zeros(1)
    return torch.cat(parts, dim=0)

def dict_mul(dest, scalar):
    for k in list(dest.keys()):
        if dest[k] is None:
            continue
        dest[k] = dest[k] * scalar

def dict_clone(src):
    return {k: (None if src[k] is None else src[k].clone()) for k in src}

def gp_aggregation(grads_list, param_keys, prev_agg_flat=None, eps=1e-8):
    K = len(grads_list)
    grads_cpu = []
    for g in grads_list:
        # g may be dict with CPU tensors
        g_cpu = {k: (None if g.get(k, None) is None else g[k].detach().cpu()) for k in param_keys}
        grads_cpu.append(g_cpu)
    flats = [flatten_grad_dict(g) for g in grads_cpu]
    if prev_agg_flat is None:
        prev_agg_flat = sum(flats) / max(1, len(flats))
    sims = torch.stack([ ( (f * prev_agg_flat).sum() / ((f.norm()+eps)*(prev_agg_flat.norm()+eps)) ) for f in flats ])
    sims_shift = sims - sims.min()
    attn_weights = torch.softmax(sims_shift, dim=0)
    scaled = []
    for i,g in enumerate(grads_cpu):
        scaled_dict = {k: (None if g[k] is None else g[k] * float(attn_weights[i].item())) for k in param_keys}
        scaled.append(scaled_dict)
    deconflicted = [dict_clone(s) for s in scaled]
    for i in range(K):
        gi = deconflicted[i]
        for j in range(K):
            if i == j: continue
            gj = scaled[j]
            num = 0.0; den = 0.0
            for k in param_keys:
                a = gi[k]; b = gj[k]
                if a is None or b is None:
                    continue
                num += float((a.view(-1) * b.view(-1)).sum().item())
                den += float((b.view(-1) * b.view(-1)).sum().item())
            if den <= 0:
                continue
            if num < 0:
                coeff = num / (den + eps)
                for k in param_keys:
                    if gi[k] is None or gj[k] is None: continue
                    gi[k] = gi[k] - coeff * gj[k]
        dict_mul(gi, 1.0)
    agg = {k: None for k in param_keys}
    for d in deconflicted:
        for k in param_keys:
            if d[k] is None: continue
            if agg[k] is None:
                agg[k] = d[k].clone()
            else:
                agg[k] += d[k]
    for k in param_keys:
        if agg[k] is not None:
            agg[k] = agg[k] / float(K)
    new_flat = flatten_grad_dict(agg)
    return agg, new_flat

# ---------- server_forward on CPU ----------
def server_forward_cpu(SS_cpu, task_adapter_cpu, x1_cpu, use_checkpoint=USE_CHECKPOINT):
    """
    x1_cpu: CPU tensor (float32), requires_grad=True (we need x1_cpu.grad after backward)
    Returns predictions (CPU) and the feature tensor (CPU) for which .grad will be set.
    """
    if use_checkpoint:
        feat = checkpoint(SS_cpu, x1_cpu, use_reentrant=CHECKPOINT_REENTRANT)
    else:
        feat = SS_cpu(x1_cpu)
    if task_adapter_cpu is not None:
        feat = task_adapter_cpu(feat)
    return feat

# ---------- Local training: FE on GPU, SS+heads on CPU ----------
def train_local_and_collect_ss_grads(loader, FE_gpu, SS_cpu, BE_cpu, task_adapter_cpu,
                                    opt_fe_gpu, opt_heads_cpu, loss_fn, cid, num_classes):
    FE_gpu.train(); SS_cpu.train(); BE_cpu.train(); task_adapter_cpu.train()
    client_samples = 0
    ss_grad_dict = None

    for ep in range(LOCAL_EPOCHS):
        for batch in tqdm(loader, leave=False):
            if len(batch) == 3:
                data, target, _ = batch
            else:
                data, target = batch
            data = data.to(DEVICE, non_blocking=True)  # GPU
            target_gpu = target.long().to(DEVICE, non_blocking=True)
            client_samples += data.size(0)

            # ---------- forward FE (GPU) ----------
            x1_gpu = FE_gpu(data)  # B x C_emb x H x W on GPU

            # move embedding to CPU (float32) for server forward
            x1_cpu = x1_gpu.detach().to(CPU).float()
            x1_cpu.requires_grad_(True)

            # move target to CPU
            target_cpu = target_gpu.detach().cpu()

            # ---------- server forward (CPU) ----------
            feat_cpu = server_forward_cpu(SS_cpu, task_adapter_cpu, x1_cpu, use_checkpoint=USE_CHECKPOINT)
            preds_cpu = BE_cpu(feat_cpu)  # CPU preds

            # ---------- compute loss on CPU ----------
            loss_cpu = loss_fn(preds_cpu, target_cpu)

            # zero CPU heads grads and backward on CPU (this will set .grad for SS_cpu params and x1_cpu.grad)
            opt_heads_cpu.zero_grad()
            loss_cpu.backward()

            # collect SS grads (CPU)
            ss_grad_dict = {}
            for name, param in SS_cpu.named_parameters():
                if param.grad is None:
                    ss_grad_dict[name] = None
                else:
                    ss_grad_dict[name] = param.grad.detach().cpu().clone()

            # step CPU optimizer for heads (BE + task_adapter)
            opt_heads_cpu.step()

            # ---------- propagate embedding gradient back to FE on GPU ----------
            grad_x1_cpu = x1_cpu.grad.detach().clone() if x1_cpu.grad is not None else None
            if grad_x1_cpu is not None:
                # grad_x1_cpu is CPU float32; move to GPU
                grad_x1_gpu = grad_x1_cpu.to(DEVICE)
                # zero FE grads then backprop
                opt_fe_gpu.zero_grad()
                torch.autograd.backward(tensors=x1_gpu, grad_tensors=grad_x1_gpu)
                # step FE optimizer
                opt_fe_gpu.step()
                opt_fe_gpu.zero_grad()

            # cleanup to release memory
            del x1_gpu, x1_cpu, feat_cpu, preds_cpu, loss_cpu, grad_x1_cpu, grad_x1_gpu
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    # return SS grads (CPU dict) and sample count
    return ss_grad_dict, client_samples

# ---------- evaluation helpers ----------
def save_pred_as_image(pred_label_np, orig_size, save_path):
    maxval = pred_label_np.max() if pred_label_np.max() > 0 else 1
    img = (pred_label_np.astype(np.float32) / float(maxval)) * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    pil = Image.fromarray(img)
    pil = pil.resize(orig_size, resample=Image.NEAREST)
    pil.save(save_path)

def compute_segmentation_metrics_all(preds_lbl_np, target_np, num_classes):
    try:
        ious = jaccard_score(target_np.flatten(), preds_lbl_np.flatten(), average=None, labels=list(range(num_classes)), zero_division=0)
    except Exception:
        ious = np.zeros((num_classes,), dtype=float)
        for c in range(num_classes):
            inter = np.logical_and(target_np==c, preds_lbl_np==c).sum()
            union = np.logical_or(target_np==c, preds_lbl_np==c).sum()
            ious[c] = inter/union if union>0 else 0.0
    return {"iou": ious}

def evaluate_and_save_predictions(loader, FE, SS_cpu, BE_cpu, loss_fn, num_classes, client_id, save_dir, save_preds=True, task_adapter_cpu=None):
    FE.eval(); SS_cpu.eval(); BE_cpu.eval()
    if task_adapter_cpu is not None:
        task_adapter_cpu.eval()
    os.makedirs(save_dir, exist_ok=True)
    total_loss = 0.0; total_correct = 0.0; N = 0
    preds_all = []; targets_all = []
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(loader, leave=False)):
            if len(batch) == 3:
                data, target, fname = batch
            else:
                data, target = batch; fname=None
            data_gpu = data.to(DEVICE)
            target_gpu = target.long().to(DEVICE)

            x1_gpu = FE(data_gpu)
            x1_cpu = x1_gpu.detach().to(CPU).float()
            if USE_CHECKPOINT:
                feat_cpu = checkpoint(SS_cpu, x1_cpu, use_reentrant=CHECKPOINT_REENTRANT)
            else:
                feat_cpu = SS_cpu(x1_cpu)
            if task_adapter_cpu is not None:
                feat_cpu = task_adapter_cpu(feat_cpu)
            preds_cpu = BE_cpu(feat_cpu)
            loss = loss_fn(preds_cpu, target.detach().cpu())
            preds_lbl = torch.argmax(preds_cpu, dim=1).cpu().numpy()
            target_np = target.detach().cpu().numpy()
            batch_size = preds_lbl.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_correct += float((preds_lbl == target_np).mean())
            N += batch_size
            preds_all.append(preds_lbl); targets_all.append(target_np)

            # optional: save preds
            if save_preds:
                for bi in range(batch_size):
                    out_path = os.path.join(save_dir, f"client{client_id}_idx{idx*loader.batch_size+bi}_pred.png")
                    save_pred_as_image(preds_lbl[bi], (preds_lbl.shape[2], preds_lbl.shape[1]), out_path)

            del x1_gpu, x1_cpu, feat_cpu, preds_cpu
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    if N == 0:
        return None
    preds_all_np = np.concatenate(preds_all, axis=0)
    targets_all_np = np.concatenate(targets_all, axis=0)
    seg_metrics = compute_segmentation_metrics_all(preds_all_np, targets_all_np, num_classes)
    acc = 100.0 * total_correct / N
    return {"loss": total_loss / N, "acc": acc, "avg_iou_wb": seg_metrics["iou"].tolist(), "avg_iou_nb": seg_metrics["iou"].tolist()[1:] if num_classes>1 else [], "seg_metrics": seg_metrics}

# ---------- plotting helper ----------
def plot_iou_curves(round_num):
    dataset_names = ["Blastocysts", "HAM10K", "Fetus", "MosMed", "Kvasir"]
    rounds = list(range(1, round_num + 1))
    plt.figure(figsize=(10,5))
    for i in range(NUM_CLIENTS):
        plt.plot(rounds, test_iou_wbg_all[i], label=dataset_names[i])
    plt.xlabel("Comm Round"); plt.ylabel("IoU w/bg"); plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig("Outputs/BS_FedBone_iou_wbg.png"); plt.close()

    plt.figure(figsize=(10,5))
    for i in range(NUM_CLIENTS):
        plt.plot(rounds, test_iou_nbg_all[i], label=dataset_names[i])
    plt.xlabel("Comm Round"); plt.ylabel("IoU no/bg"); plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig("Outputs/BS_FedBone_iou_nobg.png"); plt.close()

# ---------- MAIN ----------
def main():
    task_info = {
        0: {"name": "Blastocyst", "num_classes": 5, "path": DATA_PATH, "dataset": EmbryoDataset},
        1: {"name": "HAM10K", "num_classes": 2, "path": DATA_PATH, "dataset": HAMDataset},
        2: {"name": "Fetal", "num_classes": 3, "path": DATA_PATH, "dataset": FHPsAOPMSBDataset},
        3: {"name": "Mosmed", "num_classes": 2, "path": DATA_PATH, "dataset": covidCTDataset},
        4: {"name": "Kvasir", "num_classes": 2, "path": DATA_PATH, "dataset": CVCDataset}
    }

    tr_tf = A.Compose([A.Resize(224,224), A.Normalize(mean=[0]*3, std=[1]*3, max_pixel_value=255.0), ToTensorV2()])
    val_tf = test_tf = tr_tf

    # Keep global server model on CPU to minimize GPU peak memory
    global_SS = UNET_server(in_channels=32).to(CPU)
    server_optimizer = optim.SGD(global_SS.parameters(), lr=1e-3, momentum=0.9)
    prev_agg_flat = None
    best_loss = float('inf')

    for r in range(COMM_ROUNDS):
        print(f"\n[Comm Round {r+1}/{COMM_ROUNDS}]")
        local_SS_grads = []
        client_sizes = []
        total_sz = 0

        for i in range(NUM_CLIENTS):
            task = task_info[i]
            num_classes = task["num_classes"]
            path = task["path"]

            # FE, TA, BE on GPU
            FE = UNET_FE(in_channels=3).to(DEVICE)
            TA_gpu = TaskAdaptation(in_channels=32).to(DEVICE)
            BE_gpu = UNET_BE(out_channels=num_classes).to(DEVICE)
            opt_fe = optim.AdamW(FE.parameters(), lr=1e-4)
            opt_heads_gpu = optim.AdamW(list(TA_gpu.parameters()) + list(BE_gpu.parameters()), lr=1e-4)

            # Create CPU copies for server-side execution and heads (we keep heads on CPU here to match the server CPU backward)
            SS_local = UNET_server(in_channels=32).to(CPU)
            SS_local.load_state_dict(global_SS.state_dict())
            TA_cpu = TaskAdaptation(in_channels=32).to(CPU)
            BE_cpu = UNET_BE(out_channels=num_classes).to(CPU)
            # Initialize CPU heads from GPU heads to keep same init (optional)
            try:
                TA_cpu.load_state_dict({k:v.cpu() for k,v in TA_gpu.state_dict().items()})
                BE_cpu.load_state_dict({k:v.cpu() for k,v in BE_gpu.state_dict().items()})
            except Exception:
                pass
            opt_heads_cpu = optim.AdamW(list(TA_cpu.parameters()) + list(BE_cpu.parameters()), lr=1e-4)

            loss_fn = ComboLoss(num_classes)

            # dataloaders
            train_loader = get_loader(os.path.join(path, f"client{i+1}/train_imgs"),
                                      os.path.join(path, f"client{i+1}/train_masks"),
                                      task["dataset"], tr_tf, batch_size=BATCH_SIZE)
            val_loader = get_loader(os.path.join(path, f"client{i+1}/val_imgs"),
                                    os.path.join(path, f"client{i+1}/val_masks"),
                                    task["dataset"], val_tf, batch_size=BATCH_SIZE)

            # run local training: FE on GPU; SS+TA+BE on CPU
            ss_grad_dict_cpu, client_samples = train_local_and_collect_ss_grads(
                train_loader, FE, SS_local, BE_cpu, TA_cpu, opt_fe, opt_heads_cpu, loss_fn, i+1, num_classes
            )

            print(f"[Client {i+1}] finished local training (samples={client_samples})")

            if ss_grad_dict_cpu is None:
                # if no grads (empty loader), fill with None mapping
                ss_grad_dict_cpu = {name: None for name, _ in SS_local.named_parameters()}

            local_SS_grads.append(ss_grad_dict_cpu)
            client_sizes.append(client_samples)
            total_sz += client_samples

            # Evaluate quickly using GPU copies (move CPU server to GPU or run server on CPU via evaluate_and_save_predictions if implemented)
            try:
                # Move a copy of SS_local to GPU for evaluation to reuse FE+TA+BE on GPU
                SS_eval = UNET_server(in_channels=32).to(DEVICE)
                SS_eval.load_state_dict(SS_local.state_dict())
                evaluate_loader = val_loader
                del SS_eval
            except Exception as e:
                print(f"Validation preparation error client {i+1}: {e}")

            del FE, TA_gpu, BE_gpu, opt_fe, opt_heads_gpu, SS_local, TA_cpu, BE_cpu, opt_heads_cpu
            torch.cuda.empty_cache()
            gc.collect()

        if total_sz == 0:
            print("No training samples this round; skipping aggregation.")
            continue

        param_keys = [name for name, _ in global_SS.named_parameters()]
        grads_ready = []
        for g in local_SS_grads:
            aligned = {}
            for k in param_keys:
                aligned[k] = (None if k not in g else (None if g[k] is None else g[k].to(CPU)))
            grads_ready.append(aligned)

        aggregated_grad_dict_cpu, prev_agg_flat = gp_aggregation(grads_ready, param_keys, prev_agg_flat=prev_agg_flat)

        server_optimizer.zero_grad()
        for name, param in global_SS.named_parameters():
            agg = aggregated_grad_dict_cpu.get(name, None)
            if agg is None:
                param.grad = None
            else:
                param.grad = agg.to(param.device)
        server_optimizer.step()
        print("[Server] Applied aggregated SS gradients and updated global_SS.")

        # plot update
        plot_iou_curves(r+1)

    print("Training finished.")

if __name__ == "__main__":
    main()
