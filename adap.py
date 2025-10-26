import torch
import torch.nn as nn
import os
import glob
import cv2
import numpy as np
from torch.utils.data import DataLoader, Dataset


# ============================================
# STEP 1: DATASET RICORSIVO (tutte le sottocartelle)
# ============================================

class MoLaneRecursiveDataset(Dataset):
    """Carica ricorsivamente TUTTE le immagini da tutte le sottocartelle"""

    def __init__(self, base_dir, max_images=None, img_size=(512, 512)):
        self.img_size = img_size

        # Ricerca ricorsiva: ** = tutte le sottocartelle, *.jpg = tutti i file jpg
        print(f"🔍 Searching for images in {base_dir}...")
        self.img_paths = sorted(glob.glob(
            os.path.join(base_dir, '**', '*.jpg'),
            recursive=True
        ))

        if not self.img_paths:
            print(f"❌ No images found in {base_dir}")
            self.img_paths = []
        else:
            if max_images:
                self.img_paths = self.img_paths[:max_images]

            print(f"✅ Found {len(self.img_paths)} images from all subdirectories")
            print(f"\nExample paths:")
            for path in self.img_paths[:3]:
                print(f"   {path}")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]

        # Carica immagine
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ Warning: Could not load {img_path}")
            return self.__getitem__(0)

        # Resize e normalizza
        img = cv2.resize(img, self.img_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0

        # Transpose HWC -> CHW per PyTorch
        img = np.transpose(img, (2, 0, 1))

        return torch.from_numpy(img).float()


# ============================================
# STEP 2: ADAPTIVE BATCH NORMALIZATION
# ============================================

def apply_AdaBN(model, target_unlabeled_loader, device='cuda'):
    """
    Applica Adaptive Batch Normalization al modello.
    Ricalibra i parametri BN sui dati target unlabeled.

    Args:
        model: U-Net model
        target_unlabeled_loader: DataLoader con immagini unlabeled MoLane
        device: 'cuda' o 'cpu'

    Returns:
        model: modello adattato
    """

    print("\n" + "=" * 60)
    print("⚙️  APPLYING ADAPTIVE BATCH NORMALIZATION (AdaBN)")
    print("=" * 60)

    model.eval()  # Modalità evaluation
    total_batches = len(target_unlabeled_loader)

    for batch_idx, images in enumerate(target_unlabeled_loader):
        images = images.to(device)

        # Forward pass SENZA calcolo di gradiente
        # Questo aggiorna SOLO le statistiche BN
        with torch.no_grad():
            _ = model(images)

        # Progress bar
        progress = (batch_idx + 1) / total_batches * 100
        bar_length = 40
        filled = int(bar_length * (batch_idx + 1) / total_batches)
        bar = '█' * filled + '░' * (bar_length - filled)

        if (batch_idx + 1) % max(1, total_batches // 10) == 0:
            print(f"  [{bar}] {progress:.1f}% [{batch_idx + 1}/{total_batches}] batches")

    print("✅ AdaBN completed! Model adapted to target domain (gommato)")
    print("=" * 60 + "\n")

    return model


# ============================================
# STEP 3: SALVA IL MODELLO ADATTATO
# ============================================

def save_adapted_model(model, output_path, model_name="unet_adapted_adabn"):
    """
    Salva il modello adattato con AdaBN.

    Args:
        model: modello adattato
        output_path: cartella di output
        model_name: nome del file (senza estensione)
    """

    print("💾 Saving adapted model...")

    # Crea cartella se non esiste
    os.makedirs(output_path, exist_ok=True)

    # Salva solo i pesi del modello
    weights_path = os.path.join(output_path, f"{model_name}_weights.pth")
    torch.save(model.state_dict(), weights_path)
    print(f"✅ Model weights saved to: {weights_path}")

    # Salva checkpoint completo (con metadati)
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'device': str(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'),
        'adaptation_method': 'AdaBN',
        'target_domain': 'MoLane target (gommato)',
        'timestamp': str(__import__('datetime').datetime.now())
    }

    checkpoint_path = os.path.join(output_path, f"{model_name}_checkpoint.pth")
    torch.save(checkpoint, checkpoint_path)
    print(f"✅ Checkpoint saved to: {checkpoint_path}")

    print("\n📥 Download these files from Kaggle output section:")
    print(f"   - {model_name}_weights.pth")
    print(f"   - {model_name}_checkpoint.pth")

    return weights_path, checkpoint_path


# ============================================
# MAIN: ESEGUI TUTTO
# ============================================

def main():
    """Esegui l'intero pipeline: carica dati -> AdaBN -> salva modello"""

    print("\n" + "=" * 60)
    print("🚀 MOLANE + AdaBN PIPELINE")
    print("=" * 60 + "\n")

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📍 Device: {device}")
    print(f"📍 GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU'}\n")

    # ✅ STEP 1: Carica dataset MoLane target (tutte le sottocartelle!)
    print("STEP 1: Loading MoLane target dataset (recursive)...")
    print("-" * 60)

    molane_target_train = '/kaggle/input/carlane-benchmark/CARLANE/MoLane/data/train/target'

    unlabeled_dataset = MoLaneRecursiveDataset(
        molane_target_train,
        max_images=None,  # Carica TUTTE le immagini (43.843)
        img_size=(512, 512)
    )

    if len(unlabeled_dataset) == 0:
        print("❌ ERROR: No images found! Check the path:")
        print(f"   {molane_target_train}")
        return

    unlabeled_loader = DataLoader(
        unlabeled_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=2
    )

    print(f"✅ Dataset ready: {len(unlabeled_dataset)} images")
    print(f"✅ DataLoader ready: {len(unlabeled_loader)} batches\n")

    # ✅ STEP 2: Carica modello U-Net pre-addestrato
    print("STEP 2: Loading pre-trained U-Net model...")
    print("-" * 60)

    try:
        # Carica il modello da Kaggle input
        model_path = '/kaggle/input/best-unet/pytorch/default/1/best_unet_improved.pth'
        model = torch.load(model_path)
        model = model.to(device)
        model.eval()
        print(f"✅ Model loaded from: {model_path}")
        print(f"✅ Model moved to: {device}\n")
    except FileNotFoundError:
        print(f"❌ ERROR: Model not found at {model_path}")
        print("   Make sure 'best-unet' dataset is added to Kaggle notebook")
        return
    except Exception as e:
        print(f"❌ ERROR loading model: {e}")
        return

    # ✅ STEP 3: Applica AdaBN
    print("STEP 3: Applying Adaptive Batch Normalization...")
    print("-" * 60)

    model = apply_AdaBN(model, unlabeled_loader, device=device)

    # ✅ STEP 4: Salva il modello adattato
    print("STEP 4: Saving adapted model...")
    print("-" * 60)

    output_dir = '/kaggle/working'
    weights_path, checkpoint_path = save_adapted_model(
        model,
        output_dir,
        model_name="unet_adapted_adabn_molane"
    )

    # ✅ FINALE: Verifica
    print("\n" + "=" * 60)
    print("✨ PIPELINE COMPLETED SUCCESSFULLY! ✨")
    print("=" * 60)
    print("\n📊 SUMMARY:")
    print(f"   • Input: {len(unlabeled_dataset)} unlabeled images from MoLane target")
    print(f"   • Method: Adaptive Batch Normalization (AdaBN)")
    print(f"   • Output: Model adapted to gommato domain")
    print(f"   • Files saved to: {output_dir}")
    print(f"\n🎉 Your model is now ready to use on your robot!")


# ============================================
# ESEGUI IL PROGRAMMA
# ============================================

if __name__ == "__main__":
    main()

