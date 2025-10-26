# predict_FINAL_CORRECTED.py - FIX COMPATIBILITÀ MODELLO

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from matplotlib import pyplot as plt
from tqdm import tqdm

class LaneSegmentationPredictor:
    """Predittore per lane detection - VERSIONE CORRETTA"""
    
    def __init__(self, model_path, encoder='resnet50', device='cpu'):
        """
        ✅ CRITICO: Encoder deve CORRISPONDERE a quello usato nel training!
        
        Args:
            model_path: Percorso al modello (.pth)
            encoder: DEVE essere identico al training!
                      Opzioni: 'resnet50', 'resnet34', 'mobilenet_v2', etc.
            device: 'cpu' o 'cuda'
        """
        self.device = device
        self.encoder = encoder
        
        print(f"📦 Caricamento modello con encoder: {encoder}")
        print(f"   Device: {device}")
        
        try:
            self.model = self._create_model(encoder)
            self.model.to(device)
            
            # ✅ Carica pesi
            state_dict = torch.load(model_path, map_location=device)
            self.model.load_state_dict(state_dict)
            
            self.model.eval()
            
            # ✅ Forza float32
            self.model = self.model.float()
            
            print(f"✅ Modello caricato con successo!")
            print(f"   Parametri: {sum(p.numel() for p in self.model.parameters()):,}")
            
        except Exception as e:
            print(f"❌ ERRORE nel caricamento: {e}")
            print(f"\n⚠️ SUGGERIMENTI:")
            print(f"   1. Verifica che 'encoder' sia corretto: {encoder}")
            print(f"   2. Verifica che il file .pth sia corrotto")
            print(f"   3. Prova a ricaricare il modello da Kaggle")
            raise
    
    @staticmethod
    def _create_model(encoder='resnet50'):
        """Crea modello U-Net con encoder specificato"""
        model = smp.Unet(
            encoder_name=encoder,
            encoder_weights='imagenet',
            classes=1,
            activation='sigmoid',
        )
        return model
    
    def _preprocess_image(self, image_path):
        """Preprocessa immagine per inferenza"""
        
        # Leggi immagine
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"❌ Immagine non trovata: {image_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_shape = image.shape
        
        # Resize a 256x256
        image = cv2.resize(image, (256, 256))
        
        # ✅ Converti a float32
        image = image.astype(np.float32)
        
        # ✅ Normalizzazione ImageNet
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        image = (image / 255.0 - mean) / std
        
        # ✅ Converti a tensor: HxWxC → CxHxW
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)  # BxCxHxW
        
        # ✅ CRITICO: Forza float32
        image = image.float()
        
        return image, original_shape
    
    def predict(self, image_path, threshold=0.5):
        """
        ✅ Predice la maschera per un'immagine
        """
        
        try:
            # Preprocessa
            image_tensor, original_shape = self._preprocess_image(image_path)
            image_tensor = image_tensor.to(self.device)
            
            # ✅ Assicurati che sia float32
            image_tensor = image_tensor.float()
            
            # Predici
            with torch.no_grad():
                prediction = self.model(image_tensor)
            
            # Estrai e converti a numpy
            prob_mask = torch.sigmoid(prediction).squeeze().cpu().numpy()
            binary_mask = (prob_mask > threshold).astype(np.uint8) * 255
            
            # Leggi immagine originale
            original_image = cv2.imread(image_path)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            
            return prob_mask, binary_mask, original_image
        
        except Exception as e:
            print(f"❌ Errore nella predizione: {e}")
            raise
    
    def visualize(self, image_path, threshold=0.5, save_path=None):
        """Visualizza predizione"""
        
        try:
            prob_mask, binary_mask, original_image = self.predict(image_path, threshold)
            
            # Resize mask a dimensioni originali
            original_h, original_w = original_image.shape[:2]
            binary_mask_resized = cv2.resize(binary_mask, (original_w, original_h))
            
            # Crea overlay
            overlay = original_image.copy()
            lane_region = binary_mask_resized > 127
            
            # ✅ Overlay corsie (verde)
            # python
            # Correzione: crea immagine verde, mischia e applica solo la maschera
            green = np.zeros_like(original_image, dtype=np.uint8)
            green[:] = (0, 255, 0)

            # Miscelazione su tutta l'immagine (alpha e beta scalari)
            blended = cv2.addWeighted(original_image, 0.7, green, 0.3, 0)

            # Applica l'overlay solo sulle regioni delle corsie
            overlay = original_image.copy()
            overlay[lane_region] = blended[lane_region]
            
            # Visualizza
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            axes[0].imshow(original_image)
            axes[0].set_title('Immagine Originale')
            axes[0].axis('off')
            
            axes[1].imshow(prob_mask, cmap='gray')
            axes[1].set_title(f'Probabilità Corsie (max={prob_mask.max():.3f})')
            axes[1].axis('off')
            
            axes[2].imshow(overlay)
            axes[2].set_title('Overlay Corsie Rilevate')
            axes[2].axis('off')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"✅ Visualizzazione salvata: {save_path}")
            
            plt.show()
            
        except Exception as e:
            print(f"❌ Errore nella visualizzazione: {e}")
            raise


# ==================== CONFIGURAZIONE ====================

# ✅ CRITICO: Deve CORRISPONDERE al training!
MODEL_PATH = 'C:\\Users\\mfiumicelli\\downloads\\test04_nn\\best_unet_fixed.pth'
ENCODER = 'resnet50'  # ← CAMBIA qui se hai usato un encoder diverso!
TEST_IMAGE = 'C:\\Users\\mfiumicelli\\Downloads\\strada4.jpg'
TEST_IMAGES_DIR = 'C:\\Users\\mfiumicelli\\Downloads\\dataset_TuSimple\\tusimple_preprocessed\\test\\frames'
OUTPUT_DIR = './'
DEVICE = 'cpu'
THRESHOLD = 0.5


def main():
    """Funzione principale"""
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"❌ Modello non trovato: {MODEL_PATH}")
    
    print("=" * 60)
    print("🎯 PREDIZIONE LANE DETECTION - U-NET")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Encoder: {ENCODER} ← IMPORTANTE: Deve corrispondere al training!")
    print(f"Modello: {MODEL_PATH}")
    print(f"Immagine: {TEST_IMAGE}")
    print("=" * 60)
    
    try:
        # Carica predittore
        predictor = LaneSegmentationPredictor(MODEL_PATH, encoder=ENCODER, device=DEVICE)
        
        if os.path.isfile(TEST_IMAGE):
            # Singola immagine
            print(f"\n🖼️ Elaborazione: {TEST_IMAGE}")
            output_path = os.path.join(OUTPUT_DIR, 'prediction.png')
            predictor.visualize(TEST_IMAGE, threshold=THRESHOLD, save_path=output_path)
        
        elif os.path.isdir(TEST_IMAGE):
            # Cartella di immagini
            print(f"\n📁 Elaborazione cartella: {TEST_IMAGE}")
            image_files = [f for f in os.listdir(TEST_IMAGE) 
                          if f.endswith(('.jpg', '.png', '.jpeg'))]
            
            for image_file in tqdm(image_files[:5], desc="Elaborando"):
                image_path = os.path.join(TEST_IMAGE, image_file)
                output_path = os.path.join(OUTPUT_DIR, f'prediction_{image_file}')
                
                try:
                    predictor.visualize(image_path, threshold=THRESHOLD, save_path=output_path)
                except Exception as e:
                    print(f"⚠️ Errore con {image_file}: {e}")
    
    except Exception as e:
        print(f"\n❌ ERRORE CRITICO: {e}")
        print(f"\n⚠️ TROUBLESHOOTING:")
        print(f"   1. Verifica che MODEL_PATH sia corretto")
        print(f"   2. Verifica che ENCODER sia '{ENCODER}' (nel training)")
        print(f"   3. Prova a reimportare il modello da Kaggle")
        print(f"   4. Se persiste, riaddestra il modello con train_METRICS_FIXED.py [209]")


if __name__ == '__main__':
    main()
