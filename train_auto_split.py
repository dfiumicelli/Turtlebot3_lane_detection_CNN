# train.py - VERSIONE CON SPLIT AUTOMATICO DA DATASET UNICO

import os
import torch
from torch.utils.data import DataLoader, random_split
import segmentation_models_pytorch as smp
from tqdm import tqdm
from dataset import LaneSegmentationDataset
from augmentation import (
    get_training_augmentation,
    get_validation_augmentation,
    get_preprocessing
)

# ==================== CONFIGURAZIONE ====================
class Config:
    # ✅ Dataset in una sola cartella (niente train/val split separato)
    DATA_DIR = '/kaggle/input/tusimple-preprocessed/tusimple_preprocessed/'
    IMAGES_DIR = os.path.join(DATA_DIR, 'training/frames')
    MASKS_DIR = os.path.join(DATA_DIR, 'training/lane-masks')
    
    # Rapporto di split
    TRAIN_RATIO = 0.9  # 90% training
    VAL_RATIO = 0.1    # 10% validation (calcolato automaticamente)
    
    # Modello
    ENCODER = 'mobilenet_v2'
    ENCODER_WEIGHTS = 'imagenet'
    CLASSES = 1
    ACTIVATION = 'sigmoid'
    
    # Training
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    EPOCHS = 50
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4
    
    # Output
    MODEL_DIR = '/kaggle/working/'
    BEST_MODEL_PATH = os.path.join(MODEL_DIR, 'best_unet.pth')
    LAST_MODEL_PATH = os.path.join(MODEL_DIR, 'last_unet.pth')

def create_model(config):
    """Crea il modello U-Net"""
    model = smp.Unet(
        encoder_name=config.ENCODER,
        encoder_weights=config.ENCODER_WEIGHTS,
        classes=config.CLASSES,
        activation=config.ACTIVATION,
    )
    return model

def create_dataloaders_with_split(config):
    """
    Crea i DataLoader con split automatico da un dataset unico.
    
    Approccio:
    1. Carica tutto il dataset SENZA augmentation
    2. Divide in train/val usando random_split
    3. Applica augmentation differenziata a train e val
    """
    
    # Ottieni preprocessing function specifica per l'encoder
    preprocessing_fn = smp.encoders.get_preprocessing_fn(
        config.ENCODER,
        config.ENCODER_WEIGHTS
    )
    
    # ===== STEP 1: Carica il dataset COMPLETO senza augmentation =====
    print("📂 Caricamento dataset completo...")
    full_dataset = LaneSegmentationDataset(
        images_dir=config.IMAGES_DIR,
        masks_dir=config.MASKS_DIR,
        transform=None,  # ⚠️ NO augmentation per ora
        preprocessing=None,  # ⚠️ NO preprocessing per ora
    )
    
    total_size = len(full_dataset)
    print(f"✅ Dataset caricato: {total_size} immagini totali")
    
    # ===== STEP 2: Calcola le dimensioni dei subset =====
    train_size = int(config.TRAIN_RATIO * total_size)
    val_size = total_size - train_size
    
    print(f"\n📊 Split ratio: {config.TRAIN_RATIO*100:.1f}% training / {config.VAL_RATIO*100:.1f}% validation")
    print(f"   Train samples: {train_size}")
    print(f"   Val samples: {val_size}")
    
    # ===== STEP 3: Split il dataset =====
    # Usa un seed fisso per reproducibilità
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # ✅ Seed fisso per reproducibilità
    )
    
    # ===== STEP 4: Applica augmentation DIVERSA per train e val =====
    # Per il training set: applica augmentation
    train_dataset_augmented = TrainAugmentedDataset(
        dataset=train_dataset,
        transform=get_training_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
    )
    
    # Per il validation set: solo light augmentation (resize) e preprocessing
    val_dataset_augmented = ValAugmentedDataset(
        dataset=val_dataset,
        transform=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
    )
    
    # ===== STEP 5: Crea i DataLoader =====
    train_loader = DataLoader(
        train_dataset_augmented,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset_augmented,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    
    print(f"✅ DataLoaders creati!")
    return train_loader, val_loader


class TrainAugmentedDataset(torch.utils.data.Dataset):
    """
    Wrapper che applica augmentation al training set
    """
    def __init__(self, dataset, transform=None, preprocessing=None):
        self.dataset = dataset
        self.transform = transform
        self.preprocessing = preprocessing
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, mask = self.dataset[idx]
        
        # Applica augmentation
        if self.transform:
            sample = self.transform(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # Applica preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        return image, mask


class ValAugmentedDataset(torch.utils.data.Dataset):
    """
    Wrapper che applica SOLO preprocessing al validation set (no augmentation geometriche)
    """
    def __init__(self, dataset, transform=None, preprocessing=None):
        self.dataset = dataset
        self.transform = transform
        self.preprocessing = preprocessing
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, mask = self.dataset[idx]
        
        # Per validation, applica solo light transforms (es. resize, normalize)
        if self.transform:
            sample = self.transform(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # Applica preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        return image, mask


def train_epoch(model, loader, optimizer, loss_fn, device):
    """Training per una singola epoch"""
    model.train()
    total_loss = 0
    
    loop = tqdm(loader, desc='Training')
    for images, masks in loop:
        images = images.to(device)
        masks = masks.unsqueeze(1).to(device)
        
        predictions = model(images)
        loss = loss_fn(predictions, masks)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())
    
    return total_loss / len(loader)


def validate_epoch(model, loader, loss_fn, device):
    """Validation per una singola epoch"""
    model.eval()
    total_loss = 0
    total_iou = 0
    
    with torch.no_grad():
        loop = tqdm(loader, desc='Validation')
        for images, masks in loop:
            images = images.to(device)
            masks = masks.unsqueeze(1).to(device)
            
            predictions = model(images)
            loss = loss_fn(predictions, masks)
            
            pred_binary = (predictions > 0.5).float()
            intersection = (pred_binary * masks).sum()
            union = pred_binary.sum() + masks.sum() - intersection
            iou = (intersection + 1e-6) / (union + 1e-6)
            
            total_loss += loss.item()
            total_iou += iou.item()
            
            loop.set_postfix(loss=loss.item(), iou=iou.item())
    
    avg_loss = total_loss / len(loader)
    avg_iou = total_iou / len(loader)
    
    return avg_loss, avg_iou


def main():
    """Funzione principale"""
    config = Config()
    
    # Verifica che i percorsi esistono
    if not os.path.exists(config.IMAGES_DIR):
        raise FileNotFoundError(f"❌ Cartella non trovata: {config.IMAGES_DIR}")
    if not os.path.exists(config.MASKS_DIR):
        raise FileNotFoundError(f"❌ Cartella non trovata: {config.MASKS_DIR}")
    
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    
    print("=" * 60)
    print("🚀 TRAINING U-NET PER LANE DETECTION")
    print("Dataset: TuSimple Preprocessed (con split automatico)")
    print("=" * 60)
    print(f"Device: {config.DEVICE}")
    print(f"Encoder: {config.ENCODER}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Learning rate: {config.LEARNING_RATE}")
    print(f"Epochs: {config.EPOCHS}")
    print("=" * 60)
    
    model = create_model(config)
    model.to(config.DEVICE)
    
    train_loader, val_loader = create_dataloaders_with_split(config)
    
    loss_fn = smp.losses.DiceLoss(mode='binary')
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=5,
        factor=0.5,
        verbose=True
    )
    
    best_iou = 0.0
    
    print("\n🏋️ Inizio training...\n")
    
    for epoch in range(config.EPOCHS):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{config.EPOCHS}")
        print(f"{'='*60}")
        
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, config.DEVICE)
        val_loss, val_iou = validate_epoch(model, val_loader, loss_fn, config.DEVICE)
        
        scheduler.step(val_loss)
        
        print(f"\n📈 Results:")
        print(f"   Train Loss: {train_loss:.4f}")
        print(f"   Val Loss: {val_loss:.4f}")
        print(f"   Val IoU: {val_iou:.4f}")
        
        torch.save(model.state_dict(), config.LAST_MODEL_PATH)
        
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), config.BEST_MODEL_PATH)
            print(f"\n   ✅ Nuovo miglior modello! IoU: {best_iou:.4f}")
    
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETATO!")
    print(f"   Miglior IoU: {best_iou:.4f}")
    print(f"   Modello salvato in: {config.BEST_MODEL_PATH}")
    print("=" * 60)


if __name__ == '__main__':
    main()
