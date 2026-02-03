# centralized_split_mosmed_fixed.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import copy
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
from dataset import covidCTDataset
from PIL import Image
import scipy.ndimage as ndi
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim

import sys
output_file = "XXXXX/causalenv/mosmed_CEN_LOCAL.txt"
sys.stdout = open(output_file, "w")

# -------- CONFIG --------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_PATH = LINK_TO_DATA_PATH
mosmed_SUBDIR = "client4"
NUM_CLASSES = 2
BATCH_SIZE = 1
NUM_WORKERS = 1
NUM_EPOCHS = 120
LR = 1e-4
BEST_MODELS_DIR = "BestModels"
OUTPUT_DIR = "Outputs/test_preds_mosmed"
os.makedirs(BEST_MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------- Helpers --------
def get_loader(img_dir, mask_dir, dataset_class, transform, batch_size=1, num_workers=1, shuffle=True):
    ds = dataset_class(img_dir, mask_dir, transform=transform)
    return DataLoader(ds, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)


class ComboLoss(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.dice = smp.losses.DiceLoss(mode='multiclass', from_logits=True)
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        """
        logits: (B, C, H, W)
        targets: (B, H, W) integer labels OR (B, C, H, W) one-hot
        """
        device = logits.device
        try:
            if targets.dim() == 4 and targets.shape == logits.shape:
                t = targets.type_as(logits)
                return self.dice(logits, t) + 0.0 * self.ce(logits, torch.argmax(t, dim=1))
            if targets.dim() == 3:
                t_idx = targets.long().to(device)
                return self.dice(logits, t_idx) + 0.0 * self.ce(logits, t_idx)
            # fallback
            return self.dice(logits, targets)
        except (AssertionError, RuntimeError, ValueError):
            if targets.dim() == 3:
                B, H, W = targets.shape
                C = logits.shape[1]
                t_idx = targets.long().to(device)
                target_onehot = torch.nn.functional.one_hot(t_idx, num_classes=C)  # (B,H,W,C)
                target_onehot = target_onehot.permute(0, 3, 1, 2).contiguous().type_as(logits)
                return self.dice(logits, target_onehot) + 0.0 * self.ce(logits, t_idx)
            else:
                raise

# ------------------------------------------------------------------------------

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

def save_pred_as_image(pred_label_np, orig_size, save_path):
    maxval = pred_label_np.max() if pred_label_np.max() > 0 else 1
    img = (pred_label_np.astype(np.float32) / float(maxval)) * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    pil = Image.fromarray(img)
    pil = pil.resize(orig_size, resample=Image.NEAREST)
    pil.save(save_path)

def evaluate_and_save_predictions(loader, FE, SS, BE, loss_fn, num_classes, save_dir, save_preds=True):
    # make sure models are in eval mode
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
            del x1, x2, preds, data_cuda, target_cuda
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
                    basename = f"mosmed_idx{idx*loader.batch_size + bi}.png"
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

                # reconstruction metrics
                try:
                    dat_item = batch[0][bi] if isinstance(batch[0], torch.Tensor) else np.array(batch[0][bi])
                except Exception:
                    dat_item = data.cpu()[bi].numpy()
                dat_np = dat_item if not isinstance(dat_item, torch.Tensor) else dat_item.cpu().numpy()
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

                pred_lbl_resized = Image.fromarray(pred_lbl.astype(np.uint8)).resize(orig_size, resample=Image.NEAREST)
                pred_lbl_arr = np.array(pred_lbl_resized).astype(np.float32)
                denom = float(max(1, pred_lbl_arr.max()))
                pred_rgb1 = (pred_lbl_arr / denom).astype(np.float32)
                pred_rgb1_3 = np.stack([pred_rgb1]*3, axis=-1)
                pred1_chw = np.transpose(pred_rgb1_3, (2,0,1))

                probs = preds_soft_cpu[bi]
                classes = np.arange(probs.shape[0]).reshape((-1,1,1))
                expected = (probs * classes).sum(axis=0)
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
                mse_sum1 += mse1; psnr_sum1 += psnr1; ssim_sum1 += ssim1

                mse2 = float(np.mean((clean_chw - pred2_chw)**2))
                try:
                    psnr2 = sk_psnr(clean_chw, pred2_chw, data_range=float(clean_chw.max() - clean_chw.min()) if clean_chw.max() - clean_chw.min() != 0 else 1.0)
                except Exception:
                    psnr2 = 0.0
                ssim2 = ssim_per_sample(clean_chw, pred2_chw)
                mse_sum2 += mse2; psnr_sum2 += psnr2; ssim_sum2 += ssim2

    if N == 0:
        return None

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

    recon_split1 = {"mse": float(mse_sum1 / N), "psnr": float(psnr_sum1 / N), "ssim": float(ssim_sum1 / N)}
    recon_split2 = {"mse": float(mse_sum2 / N), "psnr": float(psnr_sum2 / N), "ssim": float(ssim_sum2 / N)}

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

def train_one_epoch(train_loader, FE, SS, BE, opt, loss_fn):
    FE.train(); SS.train(); BE.train()
    running_loss = 0.0
    for data, target in tqdm(train_loader, leave=False):
        data, target = data.to(DEVICE), target.long().to(DEVICE)
        x1 = FE(data)
        x2 = SS(x1)
        preds = BE(x2)
        loss = loss_fn(preds, target)
        opt.zero_grad(); loss.backward(); opt.step()
        running_loss += float(loss.item())
    return running_loss / max(1, len(train_loader))

def evaluate_simple(val_loader, FE, SS, BE, loss_fn, num_classes):
    FE.eval(); SS.eval(); BE.eval()
    total_loss = 0.0
    total_correct = 0.0
    iou_accum = None
    batch_count = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(DEVICE), target.long().to(DEVICE)
            x1 = FE(data)
            x2 = SS(x1)
            preds = BE(x2)
            loss = loss_fn(preds, target)
            preds_lbl = torch.argmax(preds, dim=1)
            total_loss += float(loss.item())
            total_correct += float(((preds_lbl == target).float().mean().item()))
            try:
                ious = jaccard_score(target.cpu().flatten(), preds_lbl.cpu().flatten(), average=None, labels=list(range(num_classes)))
            except Exception:
                t = target.cpu().numpy().flatten()
                p = preds_lbl.cpu().numpy().flatten()
                ious = []
                for c in range(num_classes):
                    inter = np.logical_and(t == c, p == c).sum()
                    union = np.logical_or(t == c, p == c).sum()
                    ious.append(inter/union if union>0 else 0.0)
                ious = np.array(ious)
            if iou_accum is None:
                iou_accum = ious
            else:
                iou_accum = iou_accum + ious
            batch_count += 1
    if batch_count == 0:
        return None, 0.0, [0.0]*num_classes
    avg_iou = (iou_accum / batch_count).tolist()
    N = len(val_loader.dataset)
    acc = 100.0 * total_correct / max(1, N)
    return total_loss / max(1, len(val_loader)), acc, avg_iou

def main():
    tr_tf = A.Compose([A.Resize(224, 224), A.Normalize(mean=[0]*3, std=[1]*3, max_pixel_value=255.0), ToTensorV2()])
    val_tf = tr_tf
    test_tf = tr_tf

    base = os.path.join(DATA_PATH, mosmed_SUBDIR)
    train_img_dir = os.path.join(base, "train_imgs")
    train_mask_dir = os.path.join(base, "train_masks")
    val_img_dir = os.path.join(base, "val_imgs")
    val_mask_dir = os.path.join(base, "val_masks")
    test_img_dir = os.path.join(base, "test_imgs")
    test_mask_dir = os.path.join(base, "test_masks")

    train_loader = get_loader(train_img_dir, train_mask_dir, covidCTDataset, tr_tf, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)
    val_loader = get_loader(val_img_dir, val_mask_dir, covidCTDataset, val_tf, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False)
    test_loader = get_loader(test_img_dir, test_mask_dir, covidCTDataset, test_tf, batch_size=1, num_workers=NUM_WORKERS, shuffle=False)

    FE = UNET_FE(in_channels=3).to(DEVICE)
    SS = UNET_server(in_channels=32).to(DEVICE)
    BE = UNET_BE(out_channels=NUM_CLASSES).to(DEVICE)

    opt = optim.AdamW(list(FE.parameters()) + list(SS.parameters()) + list(BE.parameters()), lr=LR)
    loss_fn = ComboLoss(NUM_CLASSES)

    best_val_loss = float('inf')

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
        train_loss = train_one_epoch(train_loader, FE, SS, BE, opt, loss_fn)
        print(f"Train Loss: {train_loss:.4f}")

        val_res = evaluate_simple(val_loader, FE, SS, BE, loss_fn, NUM_CLASSES)
        if val_res is None:
            print("Empty validation set.")
            continue
        val_loss, val_acc, val_avg_iou = val_res
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | Val IoU w/bg: {float(np.mean(val_avg_iou)):.4f} | Val IoU no/bg: {float(np.mean(val_avg_iou[1:])):.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(FE.state_dict(), os.path.join(BEST_MODELS_DIR, "best_FE_mosmed.pth"))
            torch.save(SS.state_dict(), os.path.join(BEST_MODELS_DIR, "best_SS_mosmed.pth"))
            torch.save(BE.state_dict(), os.path.join(BEST_MODELS_DIR, "best_BE_mosmed.pth"))
            print("Saved best mosmed models.")

    # FINAL TESTING
    results = evaluate_and_save_predictions(test_loader, FE, SS, BE, loss_fn, NUM_CLASSES, save_dir=OUTPUT_DIR, save_preds=True)
    if results is None:
        print("No test results (empty test set).")
        return

    seg = results["seg_metrics"]
    recon1 = results["recon_split1"]
    recon2 = results["recon_split2"]

    mean_iou_wbg = float(np.mean(seg['iou']))
    mean_iou_nbg = float(np.mean(seg['iou'][1:])) if NUM_CLASSES > 1 else mean_iou_wbg
    mean_dice = float(np.mean(seg['dice']))
    mean_P = float(np.mean(seg['precision']))
    mean_R = float(np.mean(seg['recall']))
    mean_f1 = float(np.mean(seg['f1']))
    mean_HD95 = float(np.mean(seg['hd95']))
    mean_ASSD = float(np.mean(seg['assd']))

    print("\n===== mosmed TEST SUMMARY =====")
    print(f"Mean IoU (w/bg): {mean_iou_wbg:.4f}")
    print(f"Mean IoU (no/bg): {mean_iou_nbg:.4f}")
    print(f"Mean Dice: {mean_dice:.4f}")
    print(f"Mean Precision: {mean_P:.4f}")
    print(f"Mean Recall: {mean_R:.4f}")
    print(f"Mean F1-score: {mean_f1:.4f}")
    print(f"Mean HD95: {mean_HD95:.4f}")
    print(f"Mean ASSD: {mean_ASSD:.4f}")
    print("\nPer-class IoU:", seg['iou'])
    print("Per-class Dice:", seg['dice'])
    print("Per-class Precision:", seg['precision'])
    print("Per-class Recall:", seg['recall'])
    print("Per-class F1:", seg['f1'])
    print("Per-class HD95:", seg['hd95'])
    print("Per-class ASSD:", seg['assd'])

    print("\nReconstruction metrics (split1: argmax -> grayscale):", recon1)
    print("Reconstruction metrics (split2: softmax expectation):", recon2)
    print(f"\nPredictions saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
