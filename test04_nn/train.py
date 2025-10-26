# train_METRICS_FIXED.py - FIX CRÍTICO DELLE METRICHE

import os
import warnings
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import segmentation_models_pytorch as smp
from tqdm import tqdm

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message='.*UnsupportedFieldAttributeWarning.*')

from dataset import LaneSegmentationDataset
from augmentation import (
    get_training_augmentation,
    get_validation_augmentation,
)
from metrics import LaneMetrics

# ==================== CONFIGURAZIONE KAGGLE ====================
class Config:
    DATA_DIR = '/kaggle/input/tusimple-preprocessed/tusimple_preprocessed'
    
    IMAGES_DIR = os.path.join(DATA_DIR, 'training/frames')
    MASKS_DIR = os.path.join(DATA_DIR, 'training/lane-masks')
    
    TEST_IMAGES_DIR = os.path.join(DATA_DIR, 'test/frames')
    TEST_MASKS_DIR = os.path.join(DATA_DIR, 'test/lane-masks')
    
    TRAIN_RATIO = 0.8
    VAL_RATIO = 0.2
    
    ENCODER = 'resnet50'
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
    BEST_MODEL_PATH = os.path.join(MODEL_DIR, 'best_unet_fixed.pth')
    LAST_MODEL_PATH = os.path.join(MODEL_DIR, 'last_unet_fixed.pth')

def create_model(config):
    model = smp.Unet(
        encoder_name=config.ENCODER,
        encoder_weights=config.ENCODER_WEIGHTS,
        classes=config.CLASSES,
        activation=config.ACTIVATION,
        decoder_dropout=config.DROPOUT,
    )
    return model

def create_dataloaders_with_split(config):
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
    """
    ✅ VERSIONE CORRETTA - Metriche calcolate per OGNI IMMAGINE
    """
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
            
            # ✅ CRITICO: Conversione a probabilità e binarizzazione
            pred_prob = torch.sigmoid(predictions)
            pred_binary = (pred_prob > 0.5).float()
            
            total_loss += loss.item()
            
            # ✅ CORRETTO: Per OGNI immagine nel batch
            for batch_idx in range(pred_prob.shape[0]):
                # Estrai singola immagine e maschera dal batch
                pred_single = pred_prob[batch_idx].squeeze()      # [H, W]
                pred_bin_single = pred_binary[batch_idx].squeeze() # [H, W]
                mask_single = masks[batch_idx].squeeze()           # [H, W]
                
                # ✅ Calcola IoU CORRETTAMENTE
                intersection = (pred_bin_single * mask_single).sum()
                union = pred_bin_single.sum() + mask_single.sum() - intersection
                iou = (intersection + 1e-6) / (union + 1e-6)
                all_metrics['iou'].append(iou.item())
                
                # ✅ Calcola altre metriche
                all_metrics['dice'].append(LaneMetrics.dice_coefficient(pred_single, mask_single))
                all_metrics['sensitivity'].append(LaneMetrics.sensitivity(pred_single, mask_single))
                all_metrics['specificity'].append(LaneMetrics.specificity(pred_single, mask_single))
                all_metrics['f1'].append(LaneMetrics.f1_score(pred_single, mask_single))
                all_metrics['mcc'].append(LaneMetrics.mcc(pred_single, mask_single))
                all_metrics['accuracy'].append(LaneMetrics.pixel_accuracy(pred_single, mask_single))
            
            # ✅ Mostra metriche batch corrente
            batch_sens = sum(all_metrics['sensitivity'][-len(masks):]) / len(masks)
            batch_spec = sum(all_metrics['specificity'][-len(masks):]) / len(masks)
            loop.set_postfix(
                loss=loss.item(),
                sens=f"{batch_sens:.3f}",
                spec=f"{batch_spec:.3f}"
            )
    
    # ✅ Media finale di TUTTE le metriche
    avg_loss = total_loss / len(loader)
    
    metrics_final = {}
    for k, v in all_metrics.items():
        metrics_final[k] = sum(v) / len(v) if v else 0.0
    
    return avg_loss, metrics_final


class BinaryCrossEntropyWithPosWeight(nn.Module):
    """✅ BCE Loss con class weight dinamico"""
    
    def __init__(self, pos_weight=10.0):
        super().__init__()
        self.pos_weight = pos_weight
    
    def forward(self, predictions, targets):
        loss = nn.functional.binary_cross_entropy_with_logits(
            predictions,
            targets,
            pos_weight=torch.tensor(self.pos_weight)
        )
        return loss


class WeightedTverskyLoss(nn.Module):
    """✅ Tversky Loss AGGRESSIVA per class imbalance"""
    
    def __init__(self, alpha=0.1, beta=0.9, smooth=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
    
    def forward(self, predictions, targets):
        probs = torch.sigmoid(predictions)
        
        TP = (probs * targets).sum()
        FP = (probs * (1 - targets)).sum()
        FN = ((1 - probs) * targets).sum()
        
        tversky_index = (TP + self.smooth) / (
            TP + self.alpha * FP + self.beta * FN + self.smooth
        )
        
        return 1 - tversky_index


class CombinedLossAggressivo(nn.Module):
    """✅ Combinazione aggressiva per class imbalance"""
    
    def __init__(self, bce_weight=0.3, tversky_weight=0.7):
        super().__init__()
        self.bce_weight = bce_weight
        self.tversky_weight = tversky_weight
        
        self.bce_loss = BinaryCrossEntropyWithPosWeight(pos_weight=10.0)
        self.tversky_loss = WeightedTverskyLoss(alpha=0.1, beta=0.9)
    
    def forward(self, predictions, targets):
        bce = self.bce_loss(predictions, targets)
        tversky = self.tversky_loss(predictions, targets)
        
        return self.bce_weight * bce + self.tversky_weight * tversky


def main():
    config = Config()
    
    if not os.path.exists(config.IMAGES_DIR):
        raise FileNotFoundError(f"❌ Cartella non trovata: {config.IMAGES_DIR}")
    if not os.path.exists(config.MASKS_DIR):
        raise FileNotFoundError(f"❌ Cartella non trovata: {config.MASKS_DIR}")
    
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    
    print("=" * 70)
    print("🚀 TRAINING U-NET - METRICHE CORRETTE")
    print("🎯 Loss: BCE (pos_weight=10) + Weighted Tversky (alpha=0.1)")
    print("=" * 70)
    print(f"Encoder: {config.ENCODER}")
    print(f"Device: {config.DEVICE}")
    print(f"Learning rate: {config.LEARNING_RATE}")
    print(f"Epochs: {config.EPOCHS}")
    print("=" * 70)
    
    model = create_model(config)
    model.to(config.DEVICE)
    
    train_loader, val_loader = create_dataloaders_with_split(config)
    
    loss_fn = CombinedLossAggressivo(bce_weight=0.3, tversky_weight=0.7)
    
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
    
    best_specificity = 0.0
    best_metrics = {}
    no_improve_count = 0
    patience = 15
    
    print("\n🏋️ Inizio training...\n")
    
    for epoch in range(config.EPOCHS):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch + 1}/{config.EPOCHS}")
        print(f"{'='*70}")
        
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, config.DEVICE)
        val_loss, metrics = validate_epoch(model, val_loader, loss_fn, config.DEVICE)
        
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        
        print(f"\n📈 RISULTATI VALIDAZIONE:")
        print(f"   Train Loss:     {train_loss:.4f}")
        print(f"   Val Loss:       {val_loss:.4f}")
        print(f"   ───────────────────────")
        print(f"   Sensitivity:    {metrics['sensitivity']:.4f} (corsie trovate)")
        print(f"   🎯 Specificity: {metrics['specificity']:.4f} ← METRICA PRINCIPALE")
        print(f"   Dice:           {metrics['dice']:.4f}")
        print(f"   F1:             {metrics['f1']:.4f}")
        print(f"   IoU:            {metrics['iou']:.4f}")
        print(f"   LR:             {current_lr:.6f}")
        
        torch.save(model.state_dict(), config.LAST_MODEL_PATH)
        
        if metrics['specificity'] > best_specificity:
            best_specificity = metrics['specificity']
            best_metrics = metrics
            no_improve_count = 0
            torch.save(model.state_dict(), config.BEST_MODEL_PATH)
            print(f"\n   ✅ NUOVO MIGLIOR MODELLO!")
            print(f"      Specificity: {best_specificity:.4f}")
            print(f"      Sensitivity: {best_metrics['sensitivity']:.4f}")
            print(f"      F1: {best_metrics['f1']:.4f}")
        else:
            no_improve_count += 1
        
        if no_improve_count >= patience:
            print(f"\n⚠️ Early stopping: nessun miglioramento per {patience} epoch")
            break
    
    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETATO!")
    print("=" * 70)
    print(f"Specificity:  {best_specificity:.4f} (pochi FP) 🎯")
    print(f"Sensitivity:  {best_metrics['sensitivity']:.4f} (corsie trovate)")
    print(f"F1:           {best_metrics['f1']:.4f}")
    print(f"Dice:         {best_metrics['dice']:.4f}")
    print(f"IoU:          {best_metrics['iou']:.4f}")
    print(f"MCC:          {best_metrics['mcc']:.4f}")
    print(f"Modello:      {config.BEST_MODEL_PATH}")
    print("=" * 70)


if __name__ == '__main__':
    main()
