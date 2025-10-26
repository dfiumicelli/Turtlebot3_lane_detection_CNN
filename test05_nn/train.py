# train_IMPROVED_FINAL.py - VERSIONE AGGIORNATA CON NUOVA API PYTORCH
# Correzioni:
# - torch.amp.autocast() invece di torch.cuda.amp.autocast()
# - torch.amp.GradScaler() invece di torch.cuda.amp.GradScaler()
# - Tutti i parametri ottimizzati per lane detection

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
    get_training_augmentation_improved,
    get_validation_augmentation_improved,
)
from metrics import LaneMetrics

# ==================== CONFIGURAZIONE OTTIMIZZATA ====================
class Config:
    DATA_DIR = '/kaggle/input/tusimple-preprocessed/tusimple_preprocessed'
    IMAGES_DIR = os.path.join(DATA_DIR, 'training/frames')
    MASKS_DIR = os.path.join(DATA_DIR, 'training/lane-masks')
    TEST_IMAGES_DIR = os.path.join(DATA_DIR, 'test/frames')
    TEST_MASKS_DIR = os.path.join(DATA_DIR, 'test/lane-masks')
    
    TRAIN_RATIO = 0.8
    VAL_RATIO = 0.2
    
    ENCODER = 'efficientnet-b4'
    ENCODER_WEIGHTS = 'imagenet'
    CLASSES = 1
    ACTIVATION = None  # Usiamo sigmoid nella loss
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    EPOCHS = 50
    BATCH_SIZE = 16
    LEARNING_RATE = 5e-4
    WEIGHT_DECAY = 1e-4
    DROPOUT = 0.2
    NUM_WORKERS = 4
    
    # ✅ Mixed Precision Training
    USE_AMP = True
    
    # ✅ Warmup epochs
    WARMUP_EPOCHS = 5
    
    MODEL_DIR = '/kaggle/working/models'
    BEST_MODEL_PATH = os.path.join(MODEL_DIR, 'best_unet_improved.pth')
    LAST_MODEL_PATH = os.path.join(MODEL_DIR, 'last_unet_improved.pth')


def create_model(config):
    """Crea modello U-Net con encoder specificato"""
    model = smp.Unet(
        encoder_name=config.ENCODER,
        encoder_weights=config.ENCODER_WEIGHTS,
        classes=config.CLASSES,
        activation=config.ACTIVATION,
        decoder_dropout=config.DROPOUT,
    )
    return model


def create_dataloaders_with_split(config):
    """Crea train/val dataloaders"""
    print("📂 Caricamento dataset completo (training + test)...")
    
    full_dataset = LaneSegmentationDataset(
        images_dir=config.IMAGES_DIR,
        masks_dir=config.MASKS_DIR,
        test_images_dir=config.TEST_IMAGES_DIR,
        test_masks_dir=config.TEST_MASKS_DIR,
        transform=None,
        preprocessing=None,
    )
    
    total_size = len(full_dataset)
    print(f"✅ Dataset caricato: {total_size} immagini totali")
    
    train_size = int(config.TRAIN_RATIO * total_size)
    val_size = total_size - train_size
    
    print(f"\n📊 Split ratio: {config.TRAIN_RATIO*100:.1f}% training / {config.VAL_RATIO*100:.1f}% validation")
    print(f"   Train samples: {train_size}")
    print(f"   Val samples: {val_size}")
    
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_dataset_augmented = TrainAugmentedDataset(
        dataset=train_dataset,
        transform=get_training_augmentation_improved(),
    )
    
    val_dataset_augmented = ValAugmentedDataset(
        dataset=val_dataset,
        transform=get_validation_augmentation_improved(),
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


class TrainAugmentedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
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


class ValAugmentedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
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


# ✅ Warmup Learning Rate Scheduler
class WarmupCosineScheduler:
    """Scheduler con warmup lineare + cosine decay"""
    def __init__(self, optimizer, warmup_epochs, max_epochs, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        
    def step(self, epoch):
        if epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            # Cosine decay
            progress = (epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr


def train_epoch(model, loader, optimizer, loss_fn, device, scaler=None, use_amp=False):
    """✅ Training con Mixed Precision (API moderna)"""
    model.train()
    total_loss = 0
    loop = tqdm(loader, desc='Training')
    
    for images, masks in loop:
        images = images.to(device).float()
        masks = masks.unsqueeze(1).to(device).float()
        
        optimizer.zero_grad()
        
        # ✅ NUOVO: torch.amp.autocast (API moderna)
        if use_amp and scaler:
            with autocast(device_type='cuda', dtype=torch.float16):
                predictions = model(images)
                loss = loss_fn(predictions, masks)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            predictions = model(images)
            loss = loss_fn(predictions, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
        
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())
    
    return total_loss / len(loader)


def validate_epoch(model, loader, loss_fn, device, threshold=0.5):
    """Validation con monitoring IoU"""
    model.eval()
    total_loss = 0
    all_metrics = {
        'iou': [],
        'dice': [],
        'sensitivity': [],
        'specificity': [],
        'f1': [],
        'mcc': [],
        'accuracy': []
    }
    
    with torch.no_grad():
        loop = tqdm(loader, desc='Validation')
        for images, masks in loop:
            images = images.to(device).float()
            masks = masks.unsqueeze(1).to(device).float()
            
            predictions = model(images)
            loss = loss_fn(predictions, masks)
            
            pred_prob = torch.sigmoid(predictions)
            pred_binary = (pred_prob > threshold).float()
            
            total_loss += loss.item()
            
            for batch_idx in range(pred_prob.shape[0]):
                pred_single = pred_prob[batch_idx].squeeze()
                pred_bin_single = pred_binary[batch_idx].squeeze()
                mask_single = masks[batch_idx].squeeze()
                
                # IoU
                intersection = (pred_bin_single * mask_single).sum()
                union = pred_bin_single.sum() + mask_single.sum() - intersection
                iou = (intersection + 1e-6) / (union + 1e-6)
                all_metrics['iou'].append(iou.item())
                
                # Altre metriche
                all_metrics['dice'].append(LaneMetrics.dice_coefficient(pred_single, mask_single))
                all_metrics['sensitivity'].append(LaneMetrics.sensitivity(pred_single, mask_single))
                all_metrics['specificity'].append(LaneMetrics.specificity(pred_single, mask_single))
                all_metrics['f1'].append(LaneMetrics.f1_score(pred_single, mask_single))
                all_metrics['mcc'].append(LaneMetrics.mcc(pred_single, mask_single))
                all_metrics['accuracy'].append(LaneMetrics.pixel_accuracy(pred_single, mask_single))
            
            batch_iou = sum(all_metrics['iou'][-len(masks):]) / len(masks)
            batch_f1 = sum(all_metrics['f1'][-len(masks):]) / len(masks)
            loop.set_postfix(
                loss=loss.item(),
                iou=f"{batch_iou:.3f}",
                f1=f"{batch_f1:.3f}"
            )
    
    avg_loss = total_loss / len(loader)
    metrics_final = {}
    for k, v in all_metrics.items():
        metrics_final[k] = sum(v) / len(v) if v else 0.0
    
    return avg_loss, metrics_final


# ✅ Focal Tversky Loss
class FocalTverskyLoss(nn.Module):
    """Focal Tversky Loss - ottimizzata per class imbalance estremo"""
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
    """Focal Tversky + Dice"""
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


def main():
    config = Config()
    
    if not os.path.exists(config.IMAGES_DIR):
        raise FileNotFoundError(f"❌ Cartella non trovata: {config.IMAGES_DIR}")
    if not os.path.exists(config.MASKS_DIR):
        raise FileNotFoundError(f"❌ Cartella non trovata: {config.MASKS_DIR}")
    
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    
    print("=" * 80)
    print("🚀 TRAINING U-NET - VERSIONE OTTIMIZZATA FINALE")
    print("🎯 Loss: Focal Tversky (gamma=0.75) + Dice")
    print("⚡ Mixed Precision: Enabled (torch.amp)")
    print("=" * 80)
    print(f"Encoder: {config.ENCODER}")
    print(f"Device: {config.DEVICE}")
    print(f"Learning rate: {config.LEARNING_RATE} (with warmup)")
    print(f"Epochs: {config.EPOCHS}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print("=" * 80)
    
    model = create_model(config)
    model.to(config.DEVICE)
    
    train_loader, val_loader = create_dataloaders_with_split(config)
    
    loss_fn = CombinedLossOptimized(focal_weight=0.6, dice_weight=0.4)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # ✅ Warmup + Cosine Scheduler
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=config.WARMUP_EPOCHS,
        max_epochs=config.EPOCHS,
        min_lr=1e-6
    )
    
    # ✅ NUOVO: torch.amp.GradScaler (API moderna)
    scaler = GradScaler() if config.USE_AMP else None
    
    best_iou = 0.0
    best_metrics = {}
    no_improve_count = 0
    patience = 15
    best_threshold = 0.5
    
    print("\n🏋️ Inizio training...\n")
    
    for epoch in range(config.EPOCHS):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch + 1}/{config.EPOCHS}")
        print(f"{'='*80}")
        
        current_lr = scheduler.step(epoch)
        
        train_loss = train_epoch(
            model, train_loader, optimizer, loss_fn, 
            config.DEVICE, scaler, config.USE_AMP
        )
        
        val_loss, metrics = validate_epoch(
            model, val_loader, loss_fn, config.DEVICE, threshold=best_threshold
        )
        
        print(f"\n📈 RISULTATI VALIDAZIONE:")
        print(f"   Train Loss: {train_loss:.4f}")
        print(f"   Val Loss: {val_loss:.4f}")
        print(f"   ───────────────────────")
        print(f"   🎯 IoU: {metrics['iou']:.4f} ← METRICA PRINCIPALE")
        print(f"   F1: {metrics['f1']:.4f}")
        print(f"   Dice: {metrics['dice']:.4f}")
        print(f"   Sensitivity: {metrics['sensitivity']:.4f}")
        print(f"   Specificity: {metrics['specificity']:.4f}")
        print(f"   LR: {current_lr:.6f}")
        
        torch.save(model.state_dict(), config.LAST_MODEL_PATH)
        
        if metrics['iou'] > best_iou:
            best_iou = metrics['iou']
            best_metrics = metrics
            no_improve_count = 0
            
            torch.save(model.state_dict(), config.BEST_MODEL_PATH)
            
            print(f"\n   ✅ NUOVO MIGLIOR MODELLO!")
            print(f"   IoU: {best_iou:.4f}")
            print(f"   F1: {best_metrics['f1']:.4f}")
            print(f"   Dice: {best_metrics['dice']:.4f}")
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                print(f"\n⚠️ Early stopping: nessun miglioramento per {patience} epoch")
                break
    
    print("\n" + "=" * 80)
    print("✅ TRAINING COMPLETATO!")
    print("=" * 80)
    print(f"IoU: {best_iou:.4f} 🎯")
    print(f"F1: {best_metrics['f1']:.4f}")
    print(f"Dice: {best_metrics['dice']:.4f}")
    print(f"Sensitivity: {best_metrics['sensitivity']:.4f}")
    print(f"Specificity: {best_metrics['specificity']:.4f}")
    print(f"MCC: {best_metrics['mcc']:.4f}")
    print(f"Best Threshold: {best_threshold:.2f}")
    print(f"Modello: {config.BEST_MODEL_PATH}")
    print("=" * 80)


if __name__ == '__main__':
    main()

