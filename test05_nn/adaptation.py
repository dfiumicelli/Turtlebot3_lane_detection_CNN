# ============================================
# FINE-TUNING CON MOLANE LABELS - VERSIONE CORRETTA
# ============================================

import torch
import torch.nn as nn
import torch.optim as optim
import os
import cv2
import numpy as np
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import segmentation_models_pytorch as smp
from tqdm import tqdm
import gc

# ============================================
# LOSS FUNCTIONS
# ============================================

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


# ============================================
# UTILITIES
# ============================================

def cleanup_memory():
    gc.collect()
    torch.cuda.empty_cache()


# ============================================
# DATASET
# ============================================

class MoLaneLabeledDataset(Dataset):
    def __init__(self, images_base_dir, masks_base_dir, img_size=(512, 512)):
        self.img_size = img_size
        self.images_base_dir = images_base_dir
        self.masks_base_dir = masks_base_dir
        self.image_files = []
        
        print(f"🔍 Searching for labeled samples...")
        
        for root, dirs, files in os.walk(images_base_dir):
            for file in files:
                if file.endswith(('.jpg', '.png')):
                    rel_path = os.path.relpath(os.path.join(root, file), images_base_dir)
                    self.image_files.append(rel_path)
        
        self.image_files = sorted(self.image_files)
        print(f"✅ Found {len(self.image_files)} labeled samples\n")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_rel_path = self.image_files[idx]
        
        img_path = os.path.join(self.images_base_dir, img_rel_path)
        img = cv2.imread(img_path)
        
        if img is None:
            return self.__getitem__(0)
        
        # 🔧 CORREZIONE: Sostituisci "image" con "label" E ".jpg" con ".png"
        mask_rel_path = img_rel_path.replace('image', 'label').replace('.jpg', '.png')
        mask_path = os.path.join(self.masks_base_dir, mask_rel_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if mask is None:
            print(f"⚠️ Could not load mask: {mask_path}")
            return self.__getitem__(0)
        
        img = cv2.resize(img, self.img_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        
        mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)
        mask = mask.astype(np.float32) / 255.0
        
        return torch.from_numpy(img).float(), torch.from_numpy(mask).float()



# ============================================
# TRAINING
# ============================================

def train_epoch(model, train_loader, loss_fn, optimizer, device, epoch, total_epochs):
    model.train()
    total_loss = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs}")
    
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.unsqueeze(1).to(device)
        
        outputs = model(images)
        loss = loss_fn(outputs, masks)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.6f}'})  # ← CORRETTO!
    
    avg_loss = total_loss / len(train_loader)
    return avg_loss


def validate(model, val_loader, loss_fn, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc="Validating"):
            images = images.to(device)
            masks = masks.unsqueeze(1).to(device)
            
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            total_loss += loss.item()
    
    avg_loss = total_loss / len(val_loader)
    return avg_loss


# ============================================
# MAIN
# ============================================

def main():
    print("\n" + "="*80)
    print("🚀 FINE-TUNING ON MOLANE LABELED DATA")
    print("="*80 + "\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    cleanup_memory()
    
    # ============ STEP 1: Carica Dataset ============
    print("STEP 1: Loading labeled datasets...")
    print("-" * 60)
    
    val_images = '/kaggle/input/carlane-benchmark/CARLANE/MoLane/data/val/target'
    val_masks = '/kaggle/working/molane_val_masks'
    
    try:
        val_dataset = MoLaneLabeledDataset(val_images, val_masks)
        val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False, num_workers=1)
    except Exception as e:
        print(f"❌ Error loading validation set: {e}")
        return
    
    test_images = '/kaggle/input/carlane-benchmark/CARLANE/MoLane/data/test/target'
    test_masks = '/kaggle/working/molane_test_masks'
    
    try:
        test_dataset = MoLaneLabeledDataset(test_images, test_masks)
        test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False, num_workers=1)
    except Exception as e:
        print(f"❌ Error loading test set: {e}")
        test_dataset = None
    
    if test_dataset:
        combined_dataset = ConcatDataset([val_dataset, test_dataset])
        print(f"✅ Validation set: {len(val_dataset)} samples")
        print(f"✅ Test set: {len(test_dataset)} samples")
    else:
        combined_dataset = val_dataset
        print(f"✅ Validation set: {len(val_dataset)} samples")
    
    train_loader = DataLoader(combined_dataset, batch_size=10, shuffle=True, num_workers=1)
    print(f"✅ Combined: {len(combined_dataset)} samples\n")
    
    # ============ STEP 2: Carica Modello ============
    print("STEP 2: Loading pre-trained model...")
    print("-" * 60)
    
    print("📂 Loading model...")
    
    model = smp.Unet(
        encoder_name='efficientnet-b4',
        encoder_weights='imagenet',
        classes=1,
        activation=None,
        decoder_dropout=0.2,
    )
    
    model_path = '/kaggle/input/best-unet/pytorch/default/1/best_unet_improved.pth'
    checkpoint = torch.load(model_path, map_location=device)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    print("✅ Model loaded\n")
    
    # ============ STEP 3: Setup Training ============
    print("STEP 3: Setting up training...")
    print("-" * 60)
    
    loss_fn = CombinedLossOptimized(focal_weight=0.6, dice_weight=0.4)
    optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    
    print(f"Loss: Focal Tversky (0.6) + Dice (0.4)")
    print(f"Optimizer: AdamW (lr=1e-5)")
    print(f"Scheduler: CosineAnnealing\n")
    
    # ============ STEP 4: Training Loop ============
    print("STEP 4: Training...")
    print("-" * 60 + "\n")
    
    num_epochs = 10
    best_loss = float('inf')
    patience = 3
    patience_counter = 0
    
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, loss_fn, optimizer, 
                                device, epoch, num_epochs)
        val_loss = validate(model, val_loader, loss_fn, device)
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss: {val_loss:.6f}")
        
        scheduler.step()
        
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            print(f"  ✅ Best validation loss!")
        else:
            patience_counter += 1
            print(f"  ⚠️ No improvement ({patience_counter}/{patience})")
        
        if patience_counter >= patience:
            print(f"\n⛔ Early stopping after {epoch+1} epochs")
            break
        
        print()
    
    # ============ STEP 5: Salva Modello ============
    print("STEP 5: Saving model...")
    print("-" * 60)
    
    output_dir = '/kaggle/working'
    os.makedirs(output_dir, exist_ok=True)
    
    weights_path = os.path.join(output_dir, 'unet_finetuned_molane_weights.pth')
    torch.save(model.state_dict(), weights_path)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'adaptation_method': 'Fine-tuning on MoLane labeled data',
        'loss_function': 'Focal Tversky (0.6) + Dice (0.4)',
        'timestamp': str(__import__('datetime').datetime.now())
    }
    
    checkpoint_path = os.path.join(output_dir, 'unet_finetuned_molane_checkpoint.pth')
    torch.save(checkpoint, checkpoint_path)
    
    print(f"✅ Weights saved to: {weights_path}")
    print(f"✅ Checkpoint saved to: {checkpoint_path}\n")
    
    # ============ FINALE ============
    print("="*60)
    print("✨ FINE-TUNING COMPLETED!")
    print("="*60)
    print(f"\n📊 Summary:")
    print(f"   • Training samples: {len(combined_dataset)}")
    print(f"   • Best validation loss: {best_loss:.6f}")
    print(f"   • Total epochs: {epoch+1}")
    print(f"   • Model: smp.Unet(efficientnet-b4)")
    print(f"   • Loss: Focal Tversky + Dice")
    print(f"\n🎉 Model ready for deployment on your robot!")


if __name__ == "__main__":
    main()
