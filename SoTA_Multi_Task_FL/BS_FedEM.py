
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import copy
import math
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
from PIL import Image
import scipy.ndimage as ndi
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

import sys
output_file = "XXXXX/causalenv/BS_FedEM.txt"
sys.stdout = open(output_file, "w")

NUM_CLIENTS = 5
LOCAL_EPOCHS = 5
COMM_ROUNDS = 24
DATA_PATH = LINK_TO_DATA_PATH

M_COMPONENTS = 3
LR_COMPONENT = 1e-4
BATCH_SIZE = 1

test_iou_wbg_all = {i: [] for i in range(NUM_CLIENTS)}
test_iou_nbg_all = {i: [] for i in range(NUM_CLIENTS)}


def get_loader(img_dir, mask_dir, dataset_class, transform, batch_size=BATCH_SIZE, num_workers=0):
    ds = dataset_class(img_dir, mask_dir, transform=transform)
    return DataLoader(ds, batch_size=batch_size, num_workers=num_workers, shuffle=True, drop_last=False)

class ComboLoss(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.dice = smp.losses.DiceLoss(smp.losses.MULTICLASS_MODE, from_logits=True)
        self.ce = nn.CrossEntropyLoss(reduction='none')
    def forward(self, logits, targets):
        dice_loss = self.dice(logits, targets)
        return dice_loss, self.ce(logits, targets)

def init_components(m_count, out_channels_template):
    comps = []
    for m in range(m_count):
        be = UNET_BE(out_channels=out_channels_template).to(DEVICE)
        comps.append(be)
    return comps

def broadcast_components_to_client(global_components):
    local_copies = []
    for comp in global_components:
        model_copy = copy.deepcopy(comp)
        model_copy.to(DEVICE)
        local_copies.append(model_copy)
    return local_copies

def average_components_weighted(client_component_list, weights):
    M = len(client_component_list[0])
    agg = []
    for m in range(M):
        sd0 = copy.deepcopy(client_component_list[0][m].state_dict())
        for k in sd0.keys():
            sd0[k] = sum(weights[c] * client_component_list[c][m].state_dict()[k] for c in range(len(client_component_list)))
        model_template = copy.deepcopy(client_component_list[0][m])
        model_template.load_state_dict(sd0)
        agg.append(model_template)
    return agg

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

def ssim_per_sample(a, b):
    # a, b are CHW float arrays in [0,1]
    ch_ssims = []
    for ch in range(a.shape[0]):
        try:
            ch_ssim = sk_ssim(a[ch], b[ch], data_range=(a[ch].max() - a[ch].min()) if (a[ch].max() - a[ch].min()) != 0 else 1.0)
        except Exception:
            ch_ssim = 0.0
        ch_ssims.append(ch_ssim)
    return float(np.mean(ch_ssims))

def evaluate_client_loader_with_mixture(loader, FE, SS, client_components, pi_t, loss_ce_fn, num_classes, save_preds=False, save_dir=None, client_id=0):
    FE.eval(); SS.eval()
    for comp in client_components:
        comp.eval()
    if save_preds and save_dir:
        os.makedirs(save_dir, exist_ok=True)

    total_loss = 0.0
    total_correct = 0.0
    N = 0

    inter_sum = np.zeros((num_classes,), dtype=float)
    union_sum = np.zeros((num_classes,), dtype=float)
    tp = np.zeros((num_classes,), dtype=float)
    fp = np.zeros((num_classes,), dtype=float)
    fn = np.zeros((num_classes,), dtype=float)
    hd95_sum = np.zeros((num_classes,), dtype=float)
    assd_sum = np.zeros((num_classes,), dtype=float)

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
            bs = data.shape[0]

            data_cpu_for_recon = data.clone() if isinstance(data, torch.Tensor) else np.array(data)
            data_cuda = data.to(DEVICE)
            target_cuda = target.long().to(DEVICE)

            x1 = FE(data_cuda)
            x2 = SS(x1)

            logits_tensors = []
            logits_list_cpu = []
            for m in range(len(client_components)):
                logits_m = client_components[m](x2)  # (B, C, H, W) tensor on DEVICE
                logits_tensors.append(logits_m)
                logits_list_cpu.append(logits_m.cpu().numpy())

            logits_stack = np.stack(logits_list_cpu, axis=0)  # (M, B, C, H, W)
            weighted_logits = np.tensordot(pi_t, logits_stack, axes=([0],[0]))  # (B, C, H, W)
            preds_lbl = np.argmax(weighted_logits, axis=1)  # (B, H, W) numpy array

            stacked = torch.stack(logits_tensors, dim=0)  # (M, B, C, H, W)
            pi_tensor = torch.tensor(pi_t, dtype=torch.float32, device=DEVICE).view(len(pi_t),1,1,1,1)
            weighted_logits_gpu = (pi_tensor * stacked).sum(dim=0)  # (B,C,H,W)


            ce_loss = loss_ce_fn(weighted_logits_gpu, target_cuda).mean()
            total_loss += float(ce_loss.item())

            preds_lbl_gpu = torch.argmax(weighted_logits_gpu, dim=1)
            total_correct += float(((preds_lbl_gpu == target_cuda).float().mean()).item())

            preds_soft_cpu = torch.softmax(weighted_logits_gpu, dim=1).cpu().numpy()  # (B, C, H, W)

            tgt_np = target.cpu().numpy()

            for bi in range(bs):
                N += 1
                pred_lbl = preds_lbl[bi]
                gt = tgt_np[bi]
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

                try:
                    dat_np = data_cpu_for_recon[bi].cpu().numpy() if isinstance(data_cpu_for_recon, torch.Tensor) else np.array(data_cpu_for_recon[bi])
                except Exception:
                    dat_np = data_cpu_for_recon[bi] if not isinstance(data_cpu_for_recon, torch.Tensor) else data_cpu_for_recon[bi].cpu().numpy()
                cmin, cmax = dat_np.min(), dat_np.max()
                if cmax > 2.0:
                    dat_np = dat_np / 255.0
                else:
                    dat_np = (dat_np - cmin) / (cmax - cmin + 1e-8)
                pil_clean = Image.fromarray((np.transpose(dat_np, (1,2,0)) * 255.0).astype(np.uint8))
                basename, fullpath = try_get_filename_from_dataset(dataset, idx * loader.batch_size + bi)
                orig_size = None
                if fullpath is not None and os.path.exists(fullpath):
                    try:
                        with Image.open(fullpath) as im:
                            orig_size = im.size
                    except Exception:
                        orig_size = None
                if orig_size is None:
                    H_r, W_r = preds_lbl[bi].shape
                    orig_size = (W_r, H_r)
                pil_clean = pil_clean.resize(orig_size, resample=Image.BILINEAR)
                clean_arr = np.array(pil_clean).astype(np.float32) / 255.0
                if clean_arr.ndim == 2:
                    clean_arr = np.stack([clean_arr]*3, axis=-1)
                if clean_arr.shape[2] == 4:
                    clean_arr = clean_arr[..., :3]
                clean_chw = np.transpose(clean_arr, (2,0,1))

                pred_lbl_resized = Image.fromarray(pred_lbl.astype(np.uint8)).resize(orig_size, resample=Image.NEAREST)
                pred_lbl_arr = np.array(pred_lbl_resized).astype(np.float32)
                denom = float(max(1, pred_lbl_arr.max()))
                pred_rgb1 = (pred_lbl_arr / denom).astype(np.float32)
                pred_rgb1_3 = np.stack([pred_rgb1]*3, axis=-1)
                pred1_chw = np.transpose(pred_rgb1_3, (2,0,1))

                probs = preds_soft_cpu[bi]  # (C, H, W)
                classes = np.arange(probs.shape[0]).reshape((-1,1,1))
                expected = (probs * classes).sum(axis=0)  # (H, W)
                pil_expected = Image.fromarray(((expected / max(1.0, expected.max())) * 255.0).astype(np.uint8))
                pil_expected = pil_expected.resize(orig_size, resample=Image.BILINEAR)
                expected_arr = np.array(pil_expected).astype(np.float32) / 255.0
                expected_rgb3 = np.stack([expected_arr]*3, axis=-1)
                pred2_chw = np.transpose(expected_rgb3, (2,0,1))

                mse1 = float(np.mean((clean_chw - pred1_chw)**2))
                try:
                    psnr1 = sk_psnr(clean_chw, pred1_chw, data_range=float(clean_chw.max() - clean_chw.min()) if clean_chw.max() - clean_chw.min() != 0 else 1.0)
                except Exception:
                    psnr1 = 0.0
                ssim1 = ssim_per_sample(clean_chw, pred1_chw)
                mse_sum1 += mse1
                psnr_sum1 += psnr1
                ssim_sum1 += ssim1

                mse2 = float(np.mean((clean_chw - pred2_chw)**2))
                try:
                    psnr2 = sk_psnr(clean_chw, pred2_chw, data_range=float(clean_chw.max() - clean_chw.min()) if clean_chw.max() - clean_chw.min() != 0 else 1.0)
                except Exception:
                    psnr2 = 0.0
                ssim2 = ssim_per_sample(clean_chw, pred2_chw)
                mse_sum2 += mse2
                psnr_sum2 += psnr2
                ssim_sum2 += ssim2

                if save_preds and save_dir:
                    if basename is None:
                        basename = f"client{client_id}_idx{idx*loader.batch_size + bi}.png"
                    name_root, ext = os.path.splitext(basename)
                    out_name = f"{name_root}_pred.png"
                    out_path = os.path.join(save_dir, out_name)
                    pred_label_np = preds_lbl[bi].astype(np.uint8)
                    save_pred_as_image(pred_label_np, orig_size, out_path)

            del x1, x2, stacked, weighted_logits_gpu
            for t in logits_tensors:
                try:
                    del t
                except Exception:
                    pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if N == 0:
        return None

    num_classes_local = num_classes
    iou_per_class = np.zeros((num_classes_local,), dtype=float)
    dice_per_class = np.zeros((num_classes_local,), dtype=float)
    precision_per_class = np.zeros((num_classes_local,), dtype=float)
    recall_per_class = np.zeros((num_classes_local,), dtype=float)
    f1_per_class = np.zeros((num_classes_local,), dtype=float)
    hd95_per_class = np.zeros((num_classes_local,), dtype=float)
    assd_per_class = np.zeros((num_classes_local,), dtype=float)

    for c in range(num_classes_local):
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

    acc = 100.0 * total_correct / max(1, N)
    avg_iou_wb = iou_per_class.tolist()
    avg_iou_nb = avg_iou_wb[1:] if num_classes > 1 else []

    return {
        "loss": total_loss / max(1, N),
        "acc": acc,
        "avg_iou_wb": avg_iou_wb,
        "avg_iou_nb": avg_iou_nb,
        "seg_metrics": seg_metrics,
        "recon_split1": recon_split1,
        "recon_split2": recon_split2
    }

def client_local_em_updates(train_loader, FE, SS, client_components, pi_t, local_steps=1, device=DEVICE):
    FE.train(); SS.train()
    for comp in client_components:
        comp.train()

    ce_loss_fn = nn.CrossEntropyLoss(reduction='none')
    M = len(client_components)

    pi_numerators = np.zeros((M,), dtype=float)
    sample_count = 0
    cache = []

    with torch.no_grad():
        for batch in tqdm(train_loader, leave=False):
            if len(batch) == 3:
                data, target, _ = batch
            else:
                data, target = batch
            data = data.to(device)
            target = target.long().to(device)
            bs = data.shape[0]
            sample_count += bs
            x1 = FE(data)
            x2 = SS(x1)

            losses_per_comp = torch.zeros((M, bs), device=device)
            for m in range(M):
                logits_m = client_components[m](x2)
                ce_per_pixel = ce_loss_fn(logits_m, target)
                ce_per_sample = ce_per_pixel.view(bs, -1).mean(dim=1)
                losses_per_comp[m] = ce_per_sample

            log_pi = np.log(np.maximum(pi_t, 1e-12))
            log_pi_t = torch.tensor(log_pi, device=device).view(M, 1)
            log_weighted = log_pi_t - losses_per_comp
            max_log = torch.max(log_weighted, dim=0, keepdim=True)[0]
            stabilized = log_weighted - max_log
            expd = torch.exp(stabilized)
            sumexp = torch.sum(expd, dim=0, keepdim=True)
            q_batch = (expd / (sumexp + 1e-12)).cpu().numpy()

            pi_numerators += np.sum(q_batch, axis=1)
            cache.append((x2.detach().cpu(), target.detach().cpu(), q_batch))

    if sample_count == 0:
        return client_components, pi_t
    pi_new = pi_numerators / float(sample_count)
    pi_new = np.maximum(pi_new, 1e-12)
    pi_new = pi_new / pi_new.sum()

    optimizers = [optim.AdamW(comp.parameters(), lr=LR_COMPONENT) for comp in client_components]
    ce_loss_pixel = nn.CrossEntropyLoss(reduction='none')

    for ep in range(LOCAL_EPOCHS):
        for (x2_cpu, targ_cpu, q_batch) in cache:
            x2 = x2_cpu.to(device)
            targ = targ_cpu.to(device)
            bs = x2.shape[0]
            for m in range(M):
                optimizers[m].zero_grad()
                logits_m = client_components[m](x2)
                ce_per_pixel = ce_loss_pixel(logits_m, targ)
                ce_per_sample = ce_per_pixel.view(bs, -1).mean(dim=1)
                q_weights = torch.tensor(q_batch[m], dtype=torch.float32, device=device)
                loss_m = (q_weights * ce_per_sample).sum() / (q_weights.sum() + 1e-12)
                loss_m.backward()
                optimizers[m].step()

    return client_components, pi_new

def plot_iou_curves(round_num):
    dataset_names = ["Blastocysts", "HAM10K", "Fetus", "MosMed", "Kvasir"]
    rounds = list(range(1, round_num + 1))
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    for i in range(NUM_CLIENTS):
        plt.plot(rounds, test_iou_wbg_all[i], label=dataset_names[i])
    plt.xlabel("Communication Round")
    plt.ylabel("IoU w/b")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Outputs/FedEM_iou_wbg.png")
    plt.close()

    plt.figure(figsize=(10, 5))
    for i in range(NUM_CLIENTS):
        plt.plot(rounds, test_iou_nbg_all[i], label=dataset_names[i])
    plt.xlabel("Communication Round")
    plt.ylabel("IoU n/b")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Outputs/FedEM_iou_nobg.png")
    plt.close()

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

    max_classes = max([task_info[i]["num_classes"] for i in range(NUM_CLIENTS)])
    server_components = init_components(M_COMPONENTS, out_channels_template=max_classes)

    client_pis = [np.ones((M_COMPONENTS,), dtype=float) / float(M_COMPONENTS) for _ in range(NUM_CLIENTS)]

    best_loss = float('inf')

    for r in range(COMM_ROUNDS):
        print(f"\n[Comm Round {r+1}/{COMM_ROUNDS}]")
        clients_local_components = [broadcast_components_to_client(server_components) for _ in range(NUM_CLIENTS)]

        client_FEs = []
        client_SSs = []
        client_train_loaders = []
        client_train_sizes = []
        client_ds = []

        for i in range(NUM_CLIENTS):
            task = task_info[i]
            num_classes = task["num_classes"]
            path = task["path"]
            ds_class = task["dataset"]

            FE = UNET_FE(in_channels=3).to(DEVICE)
            SS = UNET_server(in_channels=32).to(DEVICE)
            client_FEs.append(FE)
            client_SSs.append(SS)

            train_loader = get_loader(os.path.join(path, f"client{i + 1}/train_imgs"),
                                      os.path.join(path, f"client{i + 1}/train_masks"),
                                      ds_class, tr_tf, batch_size=BATCH_SIZE)
            client_train_loaders.append(train_loader)
            client_train_sizes.append(len(train_loader.dataset))
            client_ds.append(ds_class)

        total_sz = sum(client_train_sizes) if sum(client_train_sizes) > 0 else 1
        agg_weights = [sz / float(total_sz) for sz in client_train_sizes]

        updated_client_components = []
        updated_client_pis = []
        for i in range(NUM_CLIENTS):
            print(f" Client {i+1}: running local EM updates (E-step + M-step)")
            FE = client_FEs[i]
            SS = client_SSs[i]
            local_components = clients_local_components[i]
            pi_t = client_pis[i]
            train_loader = client_train_loaders[i]
            local_components_upd, new_pi = client_local_em_updates(train_loader, FE, SS, local_components, pi_t, local_steps=LOCAL_EPOCHS)
            updated_client_components.append(local_components_upd)
            updated_client_pis.append(new_pi)
            client_pis[i] = new_pi

        print(" Server: aggregating component updates from clients")
        server_components = average_components_weighted(updated_client_components, agg_weights)

        avg_val_loss = 0.0
        for i in range(NUM_CLIENTS):
            task = task_info[i]
            num_classes = task["num_classes"]
            path = task["path"]
            ds_class = task["dataset"]

            val_loader = get_loader(os.path.join(path, f"client{i + 1}/val_imgs"),
                                    os.path.join(path, f"client{i + 1}/val_masks"),
                                    ds_class, val_tf, batch_size=BATCH_SIZE)
            FE = client_FEs[i]
            SS = client_SSs[i]
            client_components_latest = broadcast_components_to_client(server_components)
            pi_t = client_pis[i]

            loss_ce_fn = nn.CrossEntropyLoss(reduction='none')
            results = evaluate_client_loader_with_mixture(val_loader, FE, SS, client_components_latest, pi_t, loss_ce_fn, num_classes, save_preds=False, save_dir=None, client_id=i+1)
            if results is None:
                continue
            avg_val_loss += results['loss']
            print(f"[Client {i+1} Validation] Loss: {results['loss']:.4f} | Acc: {results['acc']:.2f}% | IoU wb: {float(np.mean(results['avg_iou_wb'])):.4f}")
            # print recon metrics optionally
            print(f"[Client {i+1} Validation] Recon split1: {results['recon_split1']} | Recon split2: {results['recon_split2']}")

        avg_val_loss = avg_val_loss / NUM_CLIENTS
        print(f"[Global Validation] Avg Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            os.makedirs("BestModels", exist_ok=True)
            for m_idx, comp in enumerate(server_components):
                torch.save(comp.state_dict(), f"BestModels/best_component_{m_idx}.pth")
            print("[Saved best shared components based on validation loss]")

        for i in range(NUM_CLIENTS):
            task = task_info[i]
            num_classes = task["num_classes"]
            path = task["path"]
            ds_class = task["dataset"]

            test_loader = get_loader(os.path.join(path, f"client{i + 1}/test_imgs"),
                                     os.path.join(path, f"client{i + 1}/test_masks"),
                                     ds_class, test_tf, batch_size=BATCH_SIZE)
            client_components_latest = broadcast_components_to_client(server_components)
            pi_t = client_pis[i]
            FE = client_FEs[i]
            SS = client_SSs[i]
            loss_ce_fn = nn.CrossEntropyLoss(reduction='none')

            save_dir = os.path.join("Outputs/test_preds_FedEM", f"client{i+1}_preds")
            results = evaluate_client_loader_with_mixture(test_loader, FE, SS, client_components_latest, pi_t, loss_ce_fn, num_classes, save_preds=True, save_dir=save_dir, client_id=i+1)
            if results is None:
                print(f"[Client {i+1} Testing] No test predictions (empty loader).")
                continue

            test_loss = results["loss"]
            test_acc = results["acc"]
            test_iou_wb = results["avg_iou_wb"]
            test_iou_nb = results["avg_iou_nb"]
            print(f"[Client {i + 1} Testing] Loss: {test_loss:.4f} | Acc: {test_acc:.2f}% | IoU wb: {float(np.mean(test_iou_wb)):.4f}")

            test_iou_wbg_all[i].append(float(np.mean(test_iou_wb)))
            if len(test_iou_nb) > 0:
                test_iou_nbg_all[i].append(float(np.mean(test_iou_nb)))
            else:
                test_iou_nbg_all[i].append(0.0)

            seg = results["seg_metrics"]
            print(f"[Client {i+1}] Segmentation per-class IoU: {seg['iou']}")
            print(f"[Client {i+1}] Segmentation per-class Dice: {seg['dice']}")
            print(f"[Client {i+1}] Recon split1: {results['recon_split1']}, Recon split2: {results['recon_split2']}")
            mean_iou_wbg = float(np.mean(seg['iou']))
            mean_iou_nbg = float(np.mean(seg['iou'][1:])) if num_classes > 1 else mean_iou_wbg
            print(f"Summary: mean IoU (w/bg)={mean_iou_wbg:.4f}, mean IoU (no/bg)={mean_iou_nbg:.4f}")

        plot_iou_curves(r + 1)

if __name__ == "__main__":
    main()
