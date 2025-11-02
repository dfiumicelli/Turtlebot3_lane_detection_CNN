import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.amp import autocast, GradScaler
import segmentation_models_pytorch as smp
from tqdm import tqdm
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

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


# ==================== DATASET CON MAPPING 1:1 ====================

class FineTuneDataset:
    """Dataset per fine-tuning"""

    def __init__(self, images_dir, masks_dir, img_size=(320, 240)):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.img_size = img_size

        self.image_files = sorted([
            f for f in os.listdir(images_dir)
            if f.lower().endswith(('.jpg', '.png', '.jpeg'))
        ])

        self.mask_files = sorted([
            f for f in os.listdir(masks_dir)
            if f.lower().endswith(('.jpg', '.png', '.jpeg'))
        ])

        print(f"\n Dataset caricato (MAPPING 1:1):")
        print(f"   Immagini: {len(self.image_files)}")
        print(f"   Maschere: {len(self.mask_files)}")

        self.num_pairs = min(len(self.image_files), len(self.mask_files))
        print(f"   Coppie valide: {self.num_pairs}\n")

        assert self.num_pairs > 0, "Nessun file trovato!"

    def __len__(self):
        return self.num_pairs

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        mask_name = self.mask_files[idx]

        img_path = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(self.masks_dir, mask_name)

        image = cv2.imread(img_path)
        if image is None:
            print(f"Errore lettura immagine: {img_path}")
            return self.__getitem__(0)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Errore lettura maschera: {mask_path}")
            return self.__getitem__(0)

        image = cv2.resize(image, self.img_size, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)

        mask = (mask > 127).astype(np.float32)

        transform = A.Compose([
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ])

        sample = transform(image=image, mask=mask)
        image = sample['image']
        mask = sample['mask']

        return image, mask


# ==================== CONFIGURAZIONE ====================

class Config:
    DATASET_DIR = '/kaggle/input/turtlebot-dataset/turtlebot_dataset'
    IMAGES_DIR = os.path.join(DATASET_DIR, 'test_images')
    MASKS_DIR = os.path.join(DATASET_DIR, 'mask_images')

    PRETRAINED_MODEL_PATH = '/kaggle/input/model-not-tuned/pytorch/default/1/best_unet_mobilenet1.pth'

    TRAIN_RATIO = 0.9
    VAL_RATIO = 0.1
    IMG_SIZE = (320, 240)

    ENCODER = 'mobilenet_v2'
    CLASSES = 1
    ACTIVATION = None

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    #EARLY STOPPING
    EPOCHS = 100  # Max epochs (ma con early stop fermerà prima)
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-5
    WEIGHT_DECAY = 1e-4
    DROPOUT = 0.2
    NUM_WORKERS = 2

    USE_AMP = True
    WARMUP_EPOCHS = 2

    #EARLY STOPPING PARAMS
    PATIENCE = 2
    MIN_DELTA = 0.001  # Minimo improvement richiesto
    VALIDATION_INTERVAL = 1  # Valida ogni epoch

    MODEL_DIR = '/kaggle/working/models'
    BEST_MODEL_PATH = os.path.join(MODEL_DIR, 'best_unet_finetuned.pth')
    LAST_MODEL_PATH = os.path.join(MODEL_DIR, 'last_unet_finetuned.pth')
    CHECKPOINT_DIR = os.path.join(MODEL_DIR, 'checkpoints')


# ==================== TRAINING ====================

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
    iou_scores = []

    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc="Validating"):
            images = images.to(device)
            masks = masks.unsqueeze(1).to(device)

            outputs = model(images)
            loss = loss_fn(outputs, masks)
            total_loss += loss.item()

            preds = (torch.sigmoid(outputs) > threshold).float()

            for i in range(preds.shape[0]):
                pred_flat = preds[i].flatten()
                mask_flat = masks[i].flatten()

                intersection = (pred_flat * mask_flat).sum()
                union = pred_flat.sum() + mask_flat.sum() - intersection
                iou = intersection / (union + 1e-6)
                iou_scores.append(iou.item())

    avg_loss = total_loss / len(val_loader)
    avg_iou = np.mean(iou_scores) if iou_scores else 0.0

    return avg_loss, avg_iou


# ==================== MAIN ====================

def main():
    config = Config()

    print("\n" + "=" * 80)
    print("FINE-TUNING (EARLY STOPPING AGGRESSIVO)")
    print("=" * 80 + "\n")

    if not os.path.exists(config.IMAGES_DIR):
        raise FileNotFoundError(f"{config.IMAGES_DIR}")
    if not os.path.exists(config.MASKS_DIR):
        raise FileNotFoundError(f"{config.MASKS_DIR}")
    if not os.path.exists(config.PRETRAINED_MODEL_PATH):
        raise FileNotFoundError(f"{config.PRETRAINED_MODEL_PATH}")

    os.makedirs(config.MODEL_DIR, exist_ok=True)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    print("Caricamento dataset...\n")
    full_dataset = FineTuneDataset(
        images_dir=config.IMAGES_DIR,
        masks_dir=config.MASKS_DIR,
        img_size=config.IMG_SIZE
    )

    total_size = len(full_dataset)
    train_size = int(config.TRAIN_RATIO * total_size)
    val_size = total_size - train_size

    print(f"Split: {train_size} train / {val_size} val\n")
    print(f"EARLY STOPPING:")
    print(f"   Patience: {config.PATIENCE} epoch")
    print(f"   Min delta: {config.MIN_DELTA}\n")

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS,
                            pin_memory=True)

    print("Caricamento modello pre-addestrato...\n")
    model = smp.Unet(
        encoder_name=config.ENCODER,
        encoder_weights=None,
        classes=config.CLASSES,
        activation=config.ACTIVATION,
    )

    checkpoint = torch.load(config.PRETRAINED_MODEL_PATH, map_location=config.DEVICE)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.to(config.DEVICE)
    print("Modello caricato\n")

    loss_fn = CombinedLossOptimized(focal_weight=0.6, dice_weight=0.4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=config.WARMUP_EPOCHS, max_epochs=config.EPOCHS,
                                      min_lr=1e-7)
    scaler = GradScaler() if config.USE_AMP else None

    best_iou = 0.0
    no_improve_count = 0
    epoch_history = []

    print("Inizio fine-tuning...\n")

    for epoch in range(config.EPOCHS):
        print(f"\nEpoch {epoch + 1}/{config.EPOCHS}")
        print("=" * 60)

        current_lr = scheduler.step(epoch)

        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, config.DEVICE, scaler, config.USE_AMP)
        val_loss, val_iou = validate_epoch(model, val_loader, loss_fn, config.DEVICE, threshold=0.5)

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | IoU: {val_iou:.4f} | LR: {current_lr:.6f}")

        epoch_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_iou': val_iou,
            'lr': current_lr
        })

        # Salva tutti i checkpoint per analisi
        ckpt_path = os.path.join(config.CHECKPOINT_DIR, f'epoch_{epoch + 1:03d}_iou_{val_iou:.4f}.pth')
        torch.save(model.state_dict(), ckpt_path)

        torch.save(model.state_dict(), config.LAST_MODEL_PATH)

        # EARLY STOPPING CON MIN_DELTA
        if val_iou > best_iou + config.MIN_DELTA:
            best_iou = val_iou
            no_improve_count = 0
            torch.save(model.state_dict(), config.BEST_MODEL_PATH)
            print(f"NUOVO BEST! IoU: {best_iou:.4f}")
        else:
            no_improve_count += 1
            print(f"No improvement ({no_improve_count}/{config.PATIENCE})")

            if no_improve_count >= config.PATIENCE:
                print(f"\nEARLY STOPPING! Nessun miglioramento per {config.PATIENCE} epoch")
                print(f"Epoch migliore: {best_iou:.4f}")
                break

    print("\n" + "=" * 80)
    print("FINE-TUNING COMPLETATO!")
    print("=" * 80)
    print(f"Best IoU: {best_iou:.4f}")
    print(f"Modello: {config.BEST_MODEL_PATH}")
    print(f"Checkpoints: {config.CHECKPOINT_DIR}")
    print("\nANALIZZA I CHECKPOINTS:")
    print("   best_unet_finetuned.pth ← Scegli questo")
    print("   O testa i checkpoints singoli per trovare il migliore")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()