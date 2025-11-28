import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
CUDA_LAUNCH_BLOCKING = "1"

import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.optim as optim
from sklearn.metrics import jaccard_score
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from models.UNet_3Plus_FE import UNet_3Plus_FE
from models.UNet_3Plus_SS import UNet_3Plus_SS
from models.UNet_3Plus_BE import UNet_3Plus_BE
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from dataset import EmbryoDataset
import copy
import torch.nn as nn

DEVICE = "cuda"
NUM_CLASSES = 5
NUM_CLIENTS = 5
LOCAL_EPOCHS = 12
COMM_ROUNDS = 10

def get_loader(img_dir, mask_dir, dataset_class, transform, batch_size=1, num_workers=1):
    ds = dataset_class(img_dir, mask_dir, transform=transform)
    return DataLoader(ds, batch_size=batch_size, num_workers=num_workers)

class ComboLoss(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.dice = smp.losses.DiceLoss(smp.losses.MULTICLASS_MODE, from_logits=True)
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        return 1.0 * self.dice(logits, targets) + 0.0 * self.ce(logits, targets)

def train_local(train_loader, val_loader, FE, SS, BE, optimizer,scheduler, loss_fn, client_id):
    FE.train()
    SS.train()
    BE.train()
    for epoch in range(LOCAL_EPOCHS):
        total_loss, total_correct = 0.0, 0.0
        iou_classes = [0.0] * NUM_CLASSES
        for data, targets in tqdm(train_loader, leave=False):
            data, targets = data.to(DEVICE), targets.long().to(DEVICE)
            x1 = FE(data)
            x1.retain_grad()
            x2 = SS(x1)
            x2.retain_grad()
            preds = BE(x2)
            loss = loss_fn(preds, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds_label = torch.argmax(preds, dim=1)
            total_correct += (preds_label == targets).float().mean().item()
            total_loss += loss.item()

            ious = jaccard_score(targets.cpu().flatten(), preds_label.cpu().flatten(), average=None, labels=list(range(NUM_CLASSES)))
            for i in range(NUM_CLASSES):
                iou_classes[i] += ious[i]

        #scheduler.step()
        n = len(train_loader.dataset)
        acc = 100. * total_correct / n
        avg_iou = [v / n for v in iou_classes]
        print(f"Client {client_id} | Local Epoch {epoch+1} | Train Loss: {total_loss/n:.4f} | Acc: {acc:.2f}% | IoU w/bg: {sum(avg_iou)/NUM_CLASSES:.4f} | IoU no/bg: {sum(avg_iou[1:])/(NUM_CLASSES-1):.4f}")
        print("  Train Per-Class IoU:", ' | '.join([f"C{i}:{avg_iou[i]:.4f}" for i in range(NUM_CLASSES)]))

        val_loss, val_acc, val_ious = evaluate(val_loader, FE, SS, BE, loss_fn)
        print(f"Client {client_id} | Local Epoch {epoch+1} | Val Loss: {val_loss:.4f} | Acc: {val_acc:.2f}% | IoU w/bg: {sum(val_ious)/NUM_CLASSES:.4f} | IoU no/bg: {sum(val_ious[1:])/(NUM_CLASSES-1):.4f}")
        print("  Val Per-Class IoU:", ' | '.join([f"C{i}:{val_ious[i]:.4f}" for i in range(NUM_CLASSES)]))

def evaluate(loader, FE, SS, BE, loss_fn):
    #FE.eval()
    #SS.eval()
    #BE.eval()
    total_loss, total_correct = 0.0, 0.0
    iou_classes = [0.0] * NUM_CLASSES
    with torch.no_grad():
        for data, targets in loader:
            data, targets = data.to(DEVICE), targets.long().to(DEVICE)
            x1 = FE(data)
            x2 = SS(x1)
            preds = BE(x2)
            loss = loss_fn(preds, targets)
            preds_label = torch.argmax(preds, dim=1)
            total_correct += (preds_label == targets).float().mean().item()
            total_loss += loss.item()
            ious = jaccard_score(targets.cpu().flatten(), preds_label.cpu().flatten(), average=None, labels=list(range(NUM_CLASSES)))
            for i in range(NUM_CLASSES):
                iou_classes[i] += ious[i]
    N = len(loader.dataset)
    return total_loss / N, 100. * total_correct / N, [v / N for v in iou_classes]

def average_models_weighted(models, weights):
    avg_state_dict = copy.deepcopy(models[0].state_dict())
    for key in avg_state_dict.keys():
        avg_state_dict[key] = sum(weights[i] * models[i].state_dict()[key] for i in range(len(models)))
    return avg_state_dict

def main():
    val_transform = A.Compose([
            A.Resize(height=224, width=224),
            A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
            ToTensorV2(),
    ])
    train_transform = A.Compose([
            A.Resize(height=224, width=224),
            A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
            ToTensorV2(),
    ])

    '''
    train_transform = A.Compose([
        A.Resize(height=224, width=224),
        A.HorizontalFlip(p=0.2),
        A.VerticalFlip(p=0.2),
        A.RandomRotate90(p=0.3),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.GaussNoise(p=0.5),
        A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
        ToTensorV2(),
    ])
    '''
    dataDir = "C:/Users/csj5/Projects/Data/BlastocystDATA/datafed/data/shuffled/federated/"
    client_dirs = [f"client{i + 1}" for i in range(NUM_CLIENTS)]
    client_train_loaders, client_val_loaders = [], []

    for cname in client_dirs:
        train_img = dataDir + f"./{cname}/train_imgs/"
        train_mask = dataDir + f"./{cname}/train_masks/"
        val_img = dataDir + f"./{cname}/val_imgs/"
        val_mask = dataDir + f"./{cname}/val_masks/"

        client_train_loaders.append(get_loader(train_img, train_mask, EmbryoDataset, train_transform))
        client_val_loaders.append(get_loader(val_img, val_mask, EmbryoDataset, val_transform))

    # Load mutual test set for global testing
    test_img = dataDir + "./test_imgs_new/"
    test_mask = dataDir + "./test_masks_new/"
    test_loader = get_loader(test_img, test_mask, EmbryoDataset, val_transform)

    global_FE = UNet_3Plus_FE(in_channels=3).to(DEVICE)
    global_SS  = UNet_3Plus_SS(in_channels=64).to(DEVICE)
    global_BE  = UNet_3Plus_BE(n_classes=NUM_CLASSES).to(DEVICE)

    best_loss = float('inf')
    for round in range(COMM_ROUNDS):
        print(f"\n[Communication Round {round+1}/{COMM_ROUNDS}]")
        local_FE, local_SS, local_BE = [], [], []

        for i in range(NUM_CLIENTS):
            FE = UNet_3Plus_FE(in_channels=3).to(DEVICE)
            SS = UNet_3Plus_SS(in_channels=32).to(DEVICE)
            BE = UNet_3Plus_BE(n_classes=NUM_CLASSES).to(DEVICE)

            FE.load_state_dict(copy.deepcopy(global_FE.state_dict()))
            SS.load_state_dict(copy.deepcopy(global_SS.state_dict()))
            BE.load_state_dict(copy.deepcopy(global_BE.state_dict()))

            optimizer = optim.AdamW(list(FE.parameters()) + list(SS.parameters()) + list(BE.parameters()), lr=1e-4)
            #scheduler = CosineAnnealingLR(optimizer, T_max=LOCAL_EPOCHS)
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=1,eta_min=1e-6)
            loss_fn = ComboLoss(NUM_CLASSES)
            train_local(client_train_loaders[i], client_val_loaders[i], FE, SS, BE, optimizer, scheduler,loss_fn, i+1)
            local_FE.append(FE)
            local_SS.append(SS)
            local_BE.append(BE)

        client_data_sizes = [len(loader.dataset) for loader in client_train_loaders]
        total_size = sum(client_data_sizes)
        weights = [size / total_size for size in client_data_sizes]

        global_FE.load_state_dict(average_models_weighted(local_FE, weights))
        global_SS.load_state_dict(average_models_weighted(local_SS, weights))
        global_BE.load_state_dict(average_models_weighted(local_BE, weights))

        loss_fn = smp.losses.DiceLoss(smp.losses.MULTICLASS_MODE, from_logits=True)
        total_val_loss, total_val_ious = 0.0, [0.0] * NUM_CLASSES
        for i in range(NUM_CLIENTS):
            val_loss, val_acc, val_ious = evaluate(client_val_loaders[i], global_FE, global_SS, global_BE, loss_fn)
            total_val_loss += val_loss
            for j in range(NUM_CLASSES):
                total_val_ious[j] += val_ious[j]

        avg_val_loss = total_val_loss / NUM_CLIENTS
        avg_ious = [v / NUM_CLIENTS for v in total_val_ious]
        print(f"[Global Validation] Loss: {avg_val_loss:.4f} | IoU w/bg: {sum(avg_ious)/NUM_CLASSES:.4f} | IoU no/bg: {sum(avg_ious[1:])/(NUM_CLASSES-1):.4f}")
        print("  Global Per-Class IoU:", ' | '.join([f"C{i}:{avg_ious[i]:.4f}" for i in range(NUM_CLASSES)]))

        test_loss, test_acc, test_ious = evaluate(test_loader, global_FE, global_SS, global_BE, loss_fn)
        print(f"[Global Testing] Loss: {test_loss:.4f} | Acc: {test_acc:.2f}% | IoU w/bg: {sum(test_ious)/NUM_CLASSES:.4f} | IoU no/bg: {sum(test_ious[1:])/(NUM_CLASSES-1):.4f}")
        print("  Test Per-Class IoU:", ' | '.join([f"C{i}:{test_ious[i]:.4f}" for i in range(NUM_CLASSES)]))

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(global_FE.state_dict(), "BestModels/best_FE_splitfed_blasto.pth")
            torch.save(global_SS.state_dict(), "BestModels/best_SS_splitfed_blasto.pth")
            torch.save(global_BE.state_dict(), "BestModels/best_BE_splitfed_blasto.pth")
            print("[Global best model saved]")
if __name__ == "__main__":
    main()
