# train_KAGGLE_NO_WARNINGS.py - VERSIONE SENZA PYDANTIC WARNINGS

import os
import warnings
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import segmentation_models_pytorch as smp
from tqdm import tqdm

# ✅ SILENZIALIZZA I WARNING DI PYDANTIC
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message='.*UnsupportedFieldAttributeWarning.*')

from dataset import LaneSegmentationDataset
from augmentation import (
    get_training_augmentation,
    get_validation_augmentation,
)

# ==================== CONFIGURAZIONE KAGGLE ====================
class Config:
    DATA_DIR = '/kaggle/input/tusimple-preprocessed/tusimple_preprocessed'
    IMAGES_DIR = os.path.join(DATA_DIR, 'training/frames')
    MASKS_DIR = os.path.join(DATA_DIR, 'training/lane-masks')
    
    TRAIN_RATIO = 0.8
    VAL_RATIO = 0.2
    
    ENCODER = 'mobilenet_v2'
    ENCODER_WEIGHTS = 'imagenet'
    CLASSES = 1
    ACTIVATION = 'sigmoid'
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    EPOCHS = 50
    BATCH_SIZE = 16
    
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-5
    DROPOUT = 0.1
    
    NUM_WORKERS = 4
    
    MODEL_DIR = '/kaggle/working/models'
    BEST_MODEL_PATH = os.path.join(MODEL_DIR, 'best_unet.pth')
    LAST_MODEL_PATH = os.path.join(MODEL_DIR, 'last_unet.pth')

def create_model(config):
    """Crea il modello U-Net"""
    model = smp.Unet(
        encoder_name=config.ENCODER,
        encoder_weights=config.ENCODER_WEIGHTS,
        classes=config.CLASSES,
        activation=config.ACTIVATION,
        decoder_dropout=config.DROPOUT,
    )
    return model

def create_dataloaders_with_split(config):
    """Crea i DataLoader"""
    
    print("📂 Caricamento dataset completo...")
    full_dataset = LaneSegmentationDataset(
        images_dir=config.IMAGES_DIR,
        masks_dir=config.MASKS_DIR,
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
    
    print(f"✅ DataLoaders creati con {config.NUM_WORKERS} workers!")
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


def train_epoch(model, loader, optimizer, loss_fn, device):
    """Training per una singola epoch"""
    model.train()
    total_loss = 0
    
    loop = tqdm(loader, desc='Training')
    for images, masks in loop:
        images = images.to(device).float()
        masks = masks.unsqueeze(1).to(device).float()
        
        predictions = model(images)
        loss = loss_fn(predictions, masks)
        
        optimizer.zero_grad()
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())
    
    return total_loss / len(loader)


def validate_epoch(model, loader, loss_fn, device):
    """Validation per una singola epoch"""
    model.eval()
    total_loss = 0
    total_intersection = 0
    total_union = 0
    
    with torch.no_grad():
        loop = tqdm(loader, desc='Validation')
        for images, masks in loop:
            images = images.to(device).float()
            masks = masks.unsqueeze(1).to(device).float()
            
            predictions = model(images)
            loss = loss_fn(predictions, masks)
            
            pred_binary = (predictions > 0.5).float()
            
            intersection = (pred_binary * masks).sum()
            union = pred_binary.sum() + masks.sum() - intersection
            
            total_intersection += intersection.item()
            total_union += union.item()
            total_loss += loss.item()
            
            temp_iou = (intersection + 1e-6) / (union + 1e-6)
            loop.set_postfix(loss=loss.item(), iou=temp_iou.item())
    
    avg_loss = total_loss / len(loader)
    final_iou = (total_intersection + 1e-6) / (total_union + 1e-6)
    
    return avg_loss, final_iou


class CorrectedCombinedLoss(nn.Module):
    """Dice Loss + BCE Loss"""
    
    def __init__(self, alpha=0.5, smooth=1.0):
        super().__init__()
        self.alpha = alpha
        self.smooth = smooth
        
    def forward(self, predictions, targets):
        probs = torch.sigmoid(predictions)
        
        intersection = (probs * targets).sum()
        dice_loss = 1.0 - (2.0 * intersection + self.smooth) / (
            probs.sum() + targets.sum() + self.smooth
        )
        
        bce_loss = nn.functional.binary_cross_entropy(probs, targets, reduction='mean')
        
        combined_loss = self.alpha * dice_loss + (1 - self.alpha) * bce_loss
        
        return combined_loss


def main():
    """Funzione principale"""
    config = Config()
    
    if not os.path.exists(config.IMAGES_DIR):
        raise FileNotFoundError(f"❌ Cartella non trovata: {config.IMAGES_DIR}")
    if not os.path.exists(config.MASKS_DIR):
        raise FileNotFoundError(f"❌ Cartella non trovata: {config.MASKS_DIR}")
    
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    
    print("=" * 60)
    print("🚀 TRAINING U-NET - KAGGLE OPTIMIZED (NO WARNINGS)")
    print("Loss Function: Dice (50%) + BCE (50%)")
    print("=" * 60)
    print(f"Device: {config.DEVICE}")
    print(f"Learning rate: {config.LEARNING_RATE}")
    print(f"Num workers: {config.NUM_WORKERS}")
    print(f"Epochs: {config.EPOCHS}")
    print("=" * 60)
    
    model = create_model(config)
    model.to(config.DEVICE)
    
    train_loader, val_loader = create_dataloaders_with_split(config)
    
    loss_fn = CorrectedCombinedLoss(alpha=0.5, smooth=1.0)
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=5,
        T_mult=2,
        eta_min=1e-6
    )
    
    best_iou = 0.0
    best_loss = float('inf')
    no_improve_count = 0
    patience = 15
    
    print("\n🏋️ Inizio training...\n")
    
    for epoch in range(config.EPOCHS):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{config.EPOCHS}")
        print(f"{'='*60}")
        
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, config.DEVICE)
        val_loss, val_iou = validate_epoch(model, val_loader, loss_fn, config.DEVICE)
        
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        
        print(f"\n📈 Results:")
        print(f"   Train Loss: {train_loss:.4f}")
        print(f"   Val Loss: {val_loss:.4f}")
        print(f"   Val IoU: {val_iou:.4f}")
        print(f"   Learning rate: {current_lr:.6f}")
        
        torch.save(model.state_dict(), config.LAST_MODEL_PATH)
        
        if val_iou > best_iou:
            best_iou = val_iou
            best_loss = val_loss
            no_improve_count = 0
            torch.save(model.state_dict(), config.BEST_MODEL_PATH)
            print(f"\n   ✅ Nuovo miglior modello! IoU: {best_iou:.4f}, Loss: {best_loss:.4f}")
        else:
            no_improve_count += 1
        
        if no_improve_count >= patience:
            print(f"\n⚠️ Early stopping: nessun miglioramento per {patience} epoch")
            break
    
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETATO!")
    print(f"   Miglior IoU: {best_iou:.4f}")
    print(f"   Loss al miglior IoU: {best_loss:.4f}")
    print(f"   Modello salvato in: {config.BEST_MODEL_PATH}")
    print("=" * 60)


if __name__ == '__main__':
    main()
