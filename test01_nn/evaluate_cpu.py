# evaluate_cpu.py - VALUTAZIONE SU CPU (per PC locale)

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from tqdm import tqdm
from dataset_combined import LaneSegmentationDataset
from augmentation import get_validation_augmentation

# ==================== CONFIGURAZIONE ====================
class Config:
    # ✅ PATH LOCALI (modifica con i tuoi path)
    DATA_DIR = './data'
    TEST_IMAGES_DIR = os.path.join(DATA_DIR, 'test/frames')
    TEST_MASKS_DIR = os.path.join(DATA_DIR, 'test/lane-masks')
    
    # ✅ Modello addestrato (scaricato da Kaggle)
    MODEL_PATH = './models/best_unet.pth'
    
    ENCODER = 'mobilenet_v2'
    ENCODER_WEIGHTS = 'imagenet'
    CLASSES = 1
    ACTIVATION = 'sigmoid'
    
    # ✅ FORZA CPU (anche se hai GPU)
    DEVICE = 'cpu'  # ← SEMPRE CPU
    
    BATCH_SIZE = 8  # Ridotto per CPU
    NUM_WORKERS = 0  # 0 per CPU (evita multiprocessing issues)

def create_model(config):
    """Crea il modello U-Net"""
    model = smp.Unet(
        encoder_name=config.ENCODER,
        encoder_weights=config.ENCODER_WEIGHTS,
        classes=config.CLASSES,
        activation=config.ACTIVATION,
    )
    return model

def create_test_dataloader(config):
    """Crea il DataLoader per il test set"""
    
    print(f"📂 Caricamento test set da: {config.TEST_IMAGES_DIR}")
    
    test_dataset = LaneSegmentationDataset(
        images_dir=config.TEST_IMAGES_DIR,
        masks_dir=config.TEST_MASKS_DIR,
        transform=get_validation_augmentation(),  # No augmentation aggressiva
        preprocessing=None,
    )
    
    print(f"✅ Test set: {len(test_dataset)} immagini")
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=False,  # False per CPU
    )
    
    return test_loader

def evaluate(model, loader, device):
    """Valuta il modello su un dataset"""
    model.eval()
    
    total_intersection = 0
    total_union = 0
    total_samples = 0
    
    dice_loss_fn = DiceLoss()
    total_loss = 0
    
    print("\n🔬 Valutazione in corso...")
    
    with torch.no_grad():
        loop = tqdm(loader, desc='Evaluating')
        for images, masks in loop:
            images = images.to(device).float()
            masks = masks.unsqueeze(1).to(device).float()
            
            # Inferenza
            predictions = model(images)
            
            # Calcola loss
            loss = dice_loss_fn(predictions, masks)
            total_loss += loss.item()
            
            # Binarizza predizioni
            pred_binary = (predictions > 0.5).float()
            
            # Calcola IoU
            intersection = (pred_binary * masks).sum()
            union = pred_binary.sum() + masks.sum() - intersection
            
            total_intersection += intersection.item()
            total_union += union.item()
            total_samples += images.size(0)
            
            # Mostra progresso
            temp_iou = (intersection + 1e-6) / (union + 1e-6)
            loop.set_postfix(iou=temp_iou.item(), loss=loss.item())
    
    # Calcola metriche finali
    avg_loss = total_loss / len(loader)
    final_iou = (total_intersection + 1e-6) / (total_union + 1e-6)
    
    return avg_loss, final_iou, total_samples


class DiceLoss(nn.Module):
    """Dice Loss"""
    
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
        
    def forward(self, predictions, targets):
        probs = torch.sigmoid(predictions)
        
        intersection = (probs * targets).sum()
        union = probs.sum() + targets.sum()
        dice_loss = 1.0 - (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        return dice_loss


def main():
    """Funzione principale"""
    config = Config()
    
    if not os.path.exists(config.TEST_IMAGES_DIR):
        raise FileNotFoundError(f"❌ Test set non trovato: {config.TEST_IMAGES_DIR}")
    
    if not os.path.exists(config.MODEL_PATH):
        raise FileNotFoundError(f"❌ Modello non trovato: {config.MODEL_PATH}")
    
    print("=" * 60)
    print("🔬 VALUTAZIONE MODELLO U-NET (CPU)")
    print("=" * 60)
    print(f"Device: {config.DEVICE}")
    print(f"Modello: {config.MODEL_PATH}")
    print(f"Test set: {config.TEST_IMAGES_DIR}")
    print("=" * 60)
    
    # Crea modello
    print("\n📦 Caricamento modello...")
    model = create_model(config)
    
    # Carica pesi (su CPU anche se addestrato su GPU)
    state_dict = torch.load(config.MODEL_PATH, map_location=config.DEVICE)
    model.load_state_dict(state_dict)
    model.to(config.DEVICE)
    
    print("✅ Modello caricato!")
    
    # Crea dataloader
    test_loader = create_test_dataloader(config)
    
    # Valuta
    import time
    start_time = time.time()
    
    test_loss, test_iou, num_samples = evaluate(model, test_loader, config.DEVICE)
    
    elapsed_time = time.time() - start_time
    
    # Stampa risultati
    print("\n" + "=" * 60)
    print("📊 RISULTATI VALUTAZIONE TEST SET")
    print("=" * 60)
    print(f"   Numero immagini valutate: {num_samples}")
    print(f"   Test Loss: {test_loss:.4f}")
    print(f"   Test IoU: {test_iou:.4f}")
    print(f"   Tempo impiegato: {elapsed_time:.1f} secondi")
    print(f"   Velocità: {elapsed_time/num_samples:.2f} sec/immagine")
    print("=" * 60)
    
    # Interpretazione IoU
    print("\n📈 Interpretazione IoU:")
    if test_iou >= 0.85:
        print("   🎉 ECCELLENTE! (IoU >= 0.85)")
    elif test_iou >= 0.75:
        print("   ✅ MOLTO BUONO (IoU >= 0.75)")
    elif test_iou >= 0.65:
        print("   👍 BUONO (IoU >= 0.65)")
    elif test_iou >= 0.50:
        print("   ⚠️ ACCETTABILE (IoU >= 0.50)")
    else:
        print("   ❌ INSUFFICIENTE (IoU < 0.50)")


if __name__ == '__main__':
    main()
