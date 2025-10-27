# train_TUSIMPLE.py - Training U-Net su TuSimple (train + test)
# Encoder: MobileNetV2

import os
import warnings
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.amp import autocast, GradScaler
import segmentation_models_pytorch as smp
from tqdm import tqdm
import numpy as np

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message='.*UnsupportedFieldAttributeWarning.*')

from dataset import LaneSegmentationDataset
from augmentation import (
    get_training_augmentation,
    get_validation_augmentation,
)
from metrics import LaneMetrics


# ==================== LOSS FUNCTIONS ====================

class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, gamma=0.75, smooth=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, predictions, targets):
        probs = torch.sigmoid(predictions)
        TP = (probs * targets).sum()
        FP = (probs * (1 - targets)).sum()
        FN = ((1 - probs) * targets).sum()

        tversky_index = (TP + self.smooth) / (
                TP + self.alpha * FP + self.beta * FN + self.smooth
        )

        focal_tversky = torch.pow(1 - tversky_index, self.gamma)
        return focal_tversky


class CombinedLossOptimized(nn.Module):
    def __init__(self, focal_weight=0.6, dice_weight=0.4):
        super().__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.focal_tversky = FocalTverskyLoss(alpha=0.3, beta=0.7, gamma=0.75)

    def forward(self, predictions, targets):
        focal = self.focal_tversky(predictions, targets)

        probs = torch.sigmoid(predictions)
        intersection = (probs * targets).sum()
        dice = (2.0 * intersection + 1.0) / (probs.sum() + targets.sum() + 1.0)
        dice_loss = 1 - dice

        return self.focal_weight * focal + self.dice_weight * dice_loss


class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, max_epochs, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        self.base_lr = optimizer.defaults['lr']

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            lr = self.min_lr + (self.base_lr - self.min_lr) * \
                 0.5 * (1 + np.cos(np.pi * (epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        return lr


# ==================== CONFIGURAZIONE TUSIMPLE ====================

class Config:
    # 🎯 TUSIMPLE PATHS
    DATA_DIR = '/kaggle/input/tusimple-preprocessed/tusimple_preprocessed'

    # Training set
    TRAIN_IMAGES_DIR = os.path.join(DATA_DIR, 'training', 'frames')
    TRAIN_MASKS_DIR = os.path.join(DATA_DIR, 'training', 'lane-masks')

    # Test set
    TEST_IMAGES_DIR = os.path.join(DATA_DIR, 'test', 'frames')
    TEST_MASKS_DIR = os.path.join(DATA_DIR, 'test', 'lane-masks')

    # Parametri training
    TRAIN_RATIO = 0.8
    VAL_RATIO = 0.2

    # ⭐ ENCODER MOBILENETV2
    ENCODER = 'mobilenet_v2'
    ENCODER_WEIGHTS = 'imagenet'
    CLASSES = 1
    ACTIVATION = None

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    EPOCHS = 50
    BATCH_SIZE = 16
    LEARNING_RATE = 5e-4
    WEIGHT_DECAY = 1e-4
    DROPOUT = 0.2
    NUM_WORKERS = 4

    USE_AMP = True
    WARMUP_EPOCHS = 3

    MODEL_DIR = '/kaggle/working/models'
    BEST_MODEL_PATH = os.path.join(MODEL_DIR, 'best_unet_mobilenet.pth')
    LAST_MODEL_PATH = os.path.join(MODEL_DIR, 'last_unet_mobilenet.pth')


# ==================== DATASET WRAPPERS ====================

class TrainAugmentedDataset:
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, mask = self.dataset[idx]
        if self.transform:
            sample = self.transform(image=image, mask=mask)
            image = sample['image']
            mask = sample['mask']
        return image, mask


class ValAugmentedDataset:
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, mask = self.dataset[idx]
        if self.transform:
            sample = self.transform(image=image, mask=mask)
            image = sample['image']
            mask = sample['mask']
        return image, mask


# ==================== TRAINING FUNCTIONS ====================

def create_model(config):
    """Crea il modello U-Net con MobileNetV2"""
    return smp.Unet(
        encoder_name=config.ENCODER,
        encoder_weights=config.ENCODER_WEIGHTS,
        classes=config.CLASSES,
        activation=config.ACTIVATION,
        decoder_dropout=config.DROPOUT,
    )


def create_dataloaders_tusimple(config):
    """Crea train/val dataloaders da TuSimple (train + test)"""

    print("📂 Caricamento dataset TuSimple (train + test)...")

    # ✅ Carica TRAIN + TEST usando il dataset.py
    full_dataset = LaneSegmentationDataset(
        images_dir=config.TRAIN_IMAGES_DIR,
        masks_dir=config.TRAIN_MASKS_DIR,
        test_images_dir=config.TEST_IMAGES_DIR,  # ← AGGIUNGI TEST SET!
        test_masks_dir=config.TEST_MASKS_DIR,  # ← AGGIUNGI TEST SET!
        transform=None,
        preprocessing=None,
    )

    total_size = len(full_dataset)
    print(f"\n✅ Dataset TuSimple caricato: {total_size} immagini totali")

    train_size = int(config.TRAIN_RATIO * total_size)
    val_size = total_size - train_size

    print(f"\n📊 Split ratio: {config.TRAIN_RATIO * 100:.1f}% training / {config.VAL_RATIO * 100:.1f}% validation")
    print(f"   Train samples: {train_size}")
    print(f"   Val samples: {val_size}")

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_dataset_augmented = TrainAugmentedDataset(
        dataset=train_dataset,
        transform=get_training_augmentation(),
    )

    val_dataset_augmented = ValAugmentedDataset(
        dataset=val_dataset,
        transform=get_validation_augmentation(),
    )

    train_loader = DataLoader(
        train_dataset_augmented,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        prefetch_factor=2,
    )

    val_loader = DataLoader(
        val_dataset_augmented,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        prefetch_factor=2,
    )

    print(f"✅ DataLoaders creati!")
    return train_loader, val_loader


def train_epoch(model, train_loader, optimizer, loss_fn, device, scaler=None, use_amp=False):
    model.train()
    total_loss = 0

    pbar = tqdm(train_loader, desc="Training")

    for images, masks in pbar:
        images = images.to(device)
        masks = masks.unsqueeze(1).to(device)

        if use_amp and scaler:
            with autocast('cuda'):
                outputs = model(images)
                loss = loss_fn(outputs, masks)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

        optimizer.zero_grad()
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.6f}'})

    return total_loss / len(train_loader)


def validate_epoch(model, val_loader, loss_fn, device, threshold=0.5):
    model.eval()
    total_loss = 0
    all_metrics = {'iou': [], 'dice': [], 'f1': []}

    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc="Validating"):
            images = images.to(device)
            masks = masks.unsqueeze(1).to(device)

            outputs = model(images)
            loss = loss_fn(outputs, masks)
            total_loss += loss.item()

            preds = (torch.sigmoid(outputs) > threshold).float()

            # Calcola le metriche
            for i in range(preds.shape[0]):
                pred_single = preds[i].squeeze()
                mask_single = masks[i].squeeze()

                all_metrics['iou'].append(LaneMetrics.iou(pred_single, mask_single))
                all_metrics['dice'].append(LaneMetrics.dice_coefficient(pred_single, mask_single))
                all_metrics['f1'].append(LaneMetrics.f1_score(pred_single, mask_single))

    avg_loss = total_loss / len(val_loader)

    metrics_result = {
        'iou': np.mean(all_metrics['iou']) if all_metrics['iou'] else 0.0,
        'dice': np.mean(all_metrics['dice']) if all_metrics['dice'] else 0.0,
        'f1': np.mean(all_metrics['f1']) if all_metrics['f1'] else 0.0,
    }

    return avg_loss, metrics_result


# ==================== MAIN ====================

def main():
    config = Config()

    # Verifica paths
    if not os.path.exists(config.TRAIN_IMAGES_DIR):
        raise FileNotFoundError(f"❌ Training images not found: {config.TRAIN_IMAGES_DIR}")

    if not os.path.exists(config.TRAIN_MASKS_DIR):
        raise FileNotFoundError(f"❌ Training masks not found: {config.TRAIN_MASKS_DIR}")

    if not os.path.exists(config.TEST_IMAGES_DIR):
        raise FileNotFoundError(f"❌ Test images not found: {config.TEST_IMAGES_DIR}")

    if not os.path.exists(config.TEST_MASKS_DIR):
        raise FileNotFoundError(f"❌ Test masks not found: {config.TEST_MASKS_DIR}")

    os.makedirs(config.MODEL_DIR, exist_ok=True)

    print("=" * 80)
    print("🚀 TRAINING U-NET SU TUSIMPLE (Train + Test)")
    print("🎯 Dataset: TuSimple train + test combined")
    print("🎯 Encoder: MobileNetV2")
    print("🎯 Loss: Focal Tversky (gamma=0.75) + Dice")
    print("⚡ Mixed Precision: Enabled")
    print("=" * 80)
    print(f"Encoder: {config.ENCODER}")
    print(f"Device: {config.DEVICE}")
    print(f"Learning rate: {config.LEARNING_RATE} (with warmup)")
    print(f"Epochs: {config.EPOCHS}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print("=" * 80)

    model = create_model(config)
    model.to(config.DEVICE)

    train_loader, val_loader = create_dataloaders_tusimple(config)

    loss_fn = CombinedLossOptimized(focal_weight=0.6, dice_weight=0.4)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )

    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=config.WARMUP_EPOCHS,
        max_epochs=config.EPOCHS,
        min_lr=1e-6
    )

    scaler = GradScaler() if config.USE_AMP else None

    best_iou = 0.0
    best_metrics = {}
    no_improve_count = 0
    patience = 10
    best_threshold = 0.5

    print("\n🏋️ Inizio training su TuSimple...\n")

    for epoch in range(config.EPOCHS):
        print(f"\n{'=' * 80}")
        print(f"Epoch {epoch + 1}/{config.EPOCHS}")
        print(f"{'=' * 80}")

        current_lr = scheduler.step(epoch)

        train_loss = train_epoch(
            model, train_loader, optimizer, loss_fn,
            config.DEVICE, scaler, config.USE_AMP
        )

        val_loss, metrics = validate_epoch(
            model, val_loader, loss_fn, config.DEVICE, threshold=best_threshold
        )

        print(f"\n📈 RISULTATI VALIDAZIONE:")
        print(f" Train Loss: {train_loss:.4f}")
        print(f" Val Loss: {val_loss:.4f}")
        print(f" ───────────────────────")
        print(f" 🎯 IoU: {metrics['iou']:.4f}")
        print(f" F1: {metrics['f1']:.4f}")
        print(f" Dice: {metrics['dice']:.4f}")
        print(f" LR: {current_lr:.6f}")

        torch.save(model.state_dict(), config.LAST_MODEL_PATH)

        if metrics['iou'] > best_iou:
            best_iou = metrics['iou']
            best_metrics = metrics
            no_improve_count = 0
            torch.save(model.state_dict(), config.BEST_MODEL_PATH)
            print(f"\n ✅ NUOVO MIGLIOR MODELLO!")
            print(f" IoU: {best_iou:.4f}")
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                print(f"\n⚠️ Early stopping: nessun miglioramento per {patience} epoch")
                break

    print("\n" + "=" * 80)
    print("✅ TRAINING SU TUSIMPLE COMPLETATO!")
    print("=" * 80)
    print(f"Best IoU: {best_iou:.4f} 🎯")
    print(f"Best F1: {best_metrics['f1']:.4f}")
    print(f"Best Dice: {best_metrics['dice']:.4f}")
    print(f"Modello salvato: {config.BEST_MODEL_PATH}")
    print("=" * 80)


if __name__ == '__main__':
    main()
