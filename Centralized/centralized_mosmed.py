import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
CUDA_LAUNCH_BLOCKING=1
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.optim as optim
from sklearn.metrics import jaccard_score
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from models.vision_transformer import SwinUnet
from dataset import covidCTDataset
import torch.nn as nn
DEVICE = "cuda"
import os

from config import Config
config = Config()
img_size = config.DATA_IMG_SIZE
patch_size = config.MODEL_SWIN_PATCH_SIZE


# Loaders
def get_loaders_train(dataset_class, train_img, train_mask, batch_size, transform, num_workers):
    train_ds = dataset_class(train_img, train_mask, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,num_workers=num_workers)
    return train_loader

def get_loaders_val(dataset_class, val_img, val_mask, batch_size, transform, num_workers):
    val_ds = dataset_class(val_img, val_mask, transform=transform)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers)
    return val_loader

def get_loader_test(dataset_class, test_img, test_mask, transform):
    test_ds = dataset_class(test_img, test_mask, transform=transform)
    test_loader = DataLoader(test_ds)
    return test_loader

class ComboLoss(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.dice = smp.losses.DiceLoss(smp.losses.MULTICLASS_MODE, from_logits=True)
        self.ce = nn.CrossEntropyLoss()
    def forward(self, logits, targets):
        return 1.0 * self.dice(logits, targets) + 0.0 * self.ce(logits, targets)

# Evaluation function
def eval_performance(loader, model, loss_fn, num_classes):
    val_running_loss = 0.0
    valid_running_correct = 0.0
    valid_iou_score_class = [0.0] * num_classes

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.type(torch.LongTensor).to(DEVICE)
            predictions = model(x)
            loss = loss_fn(predictions, y)
            preds = torch.argmax(predictions, dim=1)
            equals = preds == y
            valid_running_correct += torch.mean(equals.type(torch.FloatTensor)).item()
            val_running_loss += loss.item()
            iou_sklearn = jaccard_score(y.cpu().flatten(), preds.cpu().flatten(), average=None, labels=list(range(num_classes)))
            for i in range(num_classes):
                valid_iou_score_class[i] += iou_sklearn[i]

    dataset_size = len(loader.dataset)
    epoch_loss = val_running_loss / dataset_size
    epoch_acc = 100. * (valid_running_correct / dataset_size)
    epoch_iou_class = [v / dataset_size for v in valid_iou_score_class]
    epoch_iou_withbackground = sum(epoch_iou_class) / num_classes
    epoch_iou_nobackground = sum(epoch_iou_class[1:]) / (num_classes - 1)
    return epoch_loss, epoch_acc, epoch_iou_withbackground, epoch_iou_nobackground, epoch_iou_class

# Test function
def test(loader, model, loss_fn, num_classes):
    test_running_loss = 0.0
    test_running_correct = 0.0
    test_iou_score_class = [0.0] * num_classes

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.type(torch.LongTensor).to(DEVICE)
            predictions = model(x)
            loss = loss_fn(predictions, y)
            preds = torch.argmax(predictions, dim=1)
            equals = preds == y
            test_running_correct += torch.mean(equals.type(torch.FloatTensor)).item()
            test_running_loss += loss.item()
            iou_sklearn = jaccard_score(y.cpu().flatten(), preds.cpu().flatten(), average=None, labels=list(range(num_classes)))
            for i in range(num_classes):
                test_iou_score_class[i] += iou_sklearn[i]

    dataset_size = len(loader.dataset)
    epoch_loss = test_running_loss / dataset_size
    epoch_acc = 100. * (test_running_correct / dataset_size)
    epoch_iou_class = [v / dataset_size for v in test_iou_score_class]
    epoch_iou_withbackground = sum(epoch_iou_class) / num_classes
    epoch_iou_nobackground = sum(epoch_iou_class[1:]) / (num_classes - 1)
    return epoch_loss, epoch_acc, epoch_iou_withbackground, epoch_iou_nobackground, epoch_iou_class

# Training function
def train(train_loader, model, optimizer, scheduler, loss_fn, num_classes):
    loop = tqdm(train_loader)
    total_loss, total_correct = 0.0, 0.0
    iou_classes = [0.0] * num_classes
    for data, targets in loop:
        data, targets = data.to(DEVICE), targets.long().to(DEVICE)
        preds = model(data)
        loss = loss_fn(preds, targets)

        preds_label = torch.argmax(preds, dim=1)
        total_correct += (preds_label == targets).float().mean().item()
        total_loss += loss.item()

        ious = jaccard_score(targets.cpu().flatten(), preds_label.cpu().flatten(),
                             average=None, labels=list(range(num_classes)))
        for i in range(num_classes):
            iou_classes[i] += ious[i]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loop.set_postfix(loss=loss.item())

    scheduler.step()

    N = len(train_loader.dataset)
    avg_loss = total_loss / N
    avg_acc = 100. * total_correct / N
    avg_ious = [v / N for v in iou_classes]
    return avg_loss, avg_acc, sum(avg_ious)/num_classes, sum(avg_ious[1:])/(num_classes-1), avg_ious


# Main training loop
def main():
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 1
    NUM_EPOCHS = 120
    NUM_WORKERS = 1
    IMAGE_HEIGHT = 224
    IMAGE_WIDTH = 224

    dataset_class = covidCTDataset
    NUM_CLASSES = 2

    TRAIN_IMG_DIR = LINK_TO_DATA_PATH
    TRAIN_MASK_DIR = LINK_TO_DATA_PATH
    VAL_IMG_DIR = LINK_TO_DATA_PATH
    VAL_MASK_DIR = LINK_TO_DATA_PATH
    TEST_IMG_DIR = LINK_TO_DATA_PATH
    TEST_MASK_DIR = LINK_TO_DATA_PATH

    val_transform = A.Compose([
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
            ToTensorV2(),
    ])
    train_transform = A.Compose([
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.5),
            A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
            ToTensorV2(),
    ])

    train_loader = get_loaders_train(dataset_class, TRAIN_IMG_DIR, TRAIN_MASK_DIR, BATCH_SIZE, train_transform, NUM_WORKERS)
    val_loader = get_loaders_val(dataset_class, VAL_IMG_DIR, VAL_MASK_DIR, BATCH_SIZE, val_transform, NUM_WORKERS)
    test_loader = get_loader_test(dataset_class, TEST_IMG_DIR, TEST_MASK_DIR, val_transform)

    model = SwinUnet(config, img_size=224, num_classes=NUM_CLASSES).cuda()

    for param in model.parameters():
        param.requires_grad = True

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO]: {total_params:,} total parameters.")
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO]: {total_trainable_params:,} trainable parameters.")

    loss_fn = ComboLoss(NUM_CLASSES)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE,weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=1,eta_min=1e-6)
    #scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    best_iou = 0
    for epoch in range(NUM_EPOCHS):
        print(f"[INFO]: Epoch {epoch + 1} of {NUM_EPOCHS}")

        train_results = train(train_loader, model, optimizer, scheduler, loss_fn, NUM_CLASSES)
        val_results = eval_performance(val_loader, model, loss_fn, NUM_CLASSES)
        train_loss, train_acc, train_iou_with_bg, train_iou_no_bg, train_iou_classes = train_results
        val_loss, val_acc, val_iou_with_bg, val_iou_no_bg, val_iou_classes = val_results

        def print_results(mode, loss, acc, iou_bg, iou_no_bg, iou_classes):
            print(f"\n[{mode}] Loss: {loss:.4f} | Acc: {acc:.2f}% | IoU(w/ bg): {iou_bg:.4f} | IoU(no bg): {iou_no_bg:.4f}")
            print("Per-Class IoU:", " | ".join([f"C{i}: {iou_classes[i]:.4f}" for i in range(len(iou_classes))]))

        print_results("TRAIN", train_loss, train_acc, train_iou_with_bg, train_iou_no_bg, train_iou_classes)
        print_results("VAL", val_loss, val_acc, val_iou_with_bg, val_iou_no_bg, val_iou_classes)

        os.makedirs("BestModels", exist_ok=True)
        if val_iou_no_bg > best_iou:
            best_iou = val_iou_no_bg
            torch.save(model.state_dict(),'BestModels/best_model_centralized_swinUnet_mosmed.pth')
            print("Model saved!")

    print("[INFO]: Testing the best model...")
    model.load_state_dict(torch.load('BestModels/best_model_centralized_swinUnet_mosmed.pth'))
    test_results = test(test_loader, model, loss_fn, NUM_CLASSES)
    test_loss, test_acc, test_iou_with_bg, test_iou_no_bg, test_iou_classes = test_results
    print_results("Test", test_loss, test_acc, test_iou_with_bg, test_iou_no_bg, test_iou_classes)
if __name__ == "__main__":
    main()
