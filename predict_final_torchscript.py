# python

import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2

class LanePredictor:
    """Classe per fare predizioni con U-Net addestrato - ROBOT 640x480 (supporta TorchScript)"""

    def __init__(self, model_path, device=None):
        """
        Carica il modello TorchScript (.pt) o checkpoint tradizionale (.pth)
        
        Args:
            model_path: percorso del modello TorchScript (model_traced.pt) o checkpoint
            device: 'cuda' o 'cpu' (default: auto-detect)
        """
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device if torch.cuda.is_available() else 'cpu'
        
        self.model_path = model_path
        
        # Dimensioni usate nel training
        self.input_height = 240
        self.input_width = 320
        
        # Carica il modello
        self._load_model()
        
        # Preprocessing (stesso usato in training)
        self.transform = A.Compose([
            A.Resize(height=self.input_height, width=self.input_width, interpolation=cv2.INTER_LINEAR, p=1.0),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0),
            ToTensorV2(),
        ])

    def _load_model(self):
        """Carica il modello TorchScript o checkpoint tradizionale"""
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"❌ Modello non trovato: {self.model_path}")
        
        # Detecta il tipo di file
        is_torchscript = self.model_path.endswith('.pt')
        
        print(f"\n📦 Caricamento modello: {os.path.basename(self.model_path)}")
        print(f"   Tipo: {'TorchScript' if is_torchscript else 'Checkpoint PyTorch'}")
        print(f"   Device: {self.device}")
        
        try:
            if is_torchscript:
                # ============================================
                # CARICAMENTO TORCHSCRIPT (CONSIGLIATO)
                # ============================================
                print("   Caricamento TorchScript...")
                self.model = torch.jit.load(self.model_path, map_location=self.device)
                print("✅ Modello TorchScript caricato con successo!")
                
            else:
                # ============================================
                # CARICAMENTO CHECKPOINT TRADIZIONALE
                # ============================================
                print("   Caricamento checkpoint tradizionale...")
                loaded = torch.load(self.model_path, map_location=self.device, weights_only=False)
                
                if isinstance(loaded, torch.nn.Module):
                    # Full model salvato con torch.save(model, path)
                    self.model = loaded
                    print("✅ Full model caricato!")
                else:
                    # State dict - necessita istanziazione manuale
                    import segmentation_models_pytorch as smp
                    
                    # Se è un dict con metadata, estrai lo state_dict
                    if isinstance(loaded, dict) and 'model_state_dict' in loaded:
                        state = loaded['model_state_dict']
                    else:
                        state = loaded
                    
                    # Istanzia il modello
                    self.model = smp.Unet(
                        encoder_name='mobilenet_v2',
                        encoder_weights=None,
                        classes=1,
                        activation=None,
                    )
                    
                    # Carica i pesi
                    try:
                        res = self.model.load_state_dict(state, strict=True)
                        print("✅ State_dict caricato con strict=True!")
                    except Exception as e:
                        try:
                            res = self.model.load_state_dict(state, strict=False)
                            print(f"⚠️ Caricato con strict=False. Missing: {res.missing_keys}")
                        except Exception as e2:
                            raise RuntimeError(f"❌ Errore nel caricamento dello state_dict: {e2}")
        
        except Exception as e:
            raise RuntimeError(f"❌ Errore nel caricamento del modello: {str(e)}")
        
        # Porta modello su device e setta eval
        self.model.to(self.device)
        self.model.eval()
        print(f"✅ Modello pronto per le predizioni!\n")

    def predict(self, image_path, threshold=0.5, return_original=True):
        """
        Effettua una predizione su una singola immagine
        
        Args:
            image_path: percorso dell'immagine
            threshold: soglia per la mask binaria (0-1)
            return_original: se True, ritorna anche l'immagine originale
            
        Returns:
            prob_mask: mappa di probabilità ridimensionata all'originale
            binary_mask: mask binaria ridimensionata all'originale
            original_image: immagine originale (se return_original=True)
        """
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"❌ Immagine non trovata: {image_path}")
        
        original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_height, original_width = original_image.shape[:2]
        
        # Preprocessing
        sample = self.transform(image=image)
        image_tensor = sample['image'].unsqueeze(0).to(self.device).float()
        
        # Predizione
        with torch.no_grad():
            prediction = self.model(image_tensor)
        
        # Post-processing
        prob_map = torch.sigmoid(prediction).squeeze().cpu().numpy()
        binary_mask = (prob_map > threshold).astype(np.uint8)
        
        # Ridimensiona alle dimensioni originali
        prob_mask_resized = cv2.resize(prob_map, (original_width, original_height), interpolation=cv2.INTER_LINEAR)
        binary_mask_resized = cv2.resize(binary_mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)
        
        if return_original:
            return prob_mask_resized, binary_mask_resized, original_image
        else:
            return prob_mask_resized, binary_mask_resized

    def predict_batch(self, image_paths, threshold=0.5):
        """Predizione batch su multiple immagini"""
        results = []
        total = len(image_paths)
        
        for i, image_path in enumerate(image_paths):
            try:
                prob_mask, binary_mask, original_image = self.predict(image_path, threshold)
                results.append({
                    'image_path': image_path,
                    'prob_mask': prob_mask,
                    'binary_mask': binary_mask,
                    'original_image': original_image,
                    'status': 'success'
                })
                print(f" [{i+1}/{total}] ✅ {os.path.basename(image_path)}")
            except Exception as e:
                print(f" [{i+1}/{total}] ❌ {os.path.basename(image_path)}: {e}")
                results.append({'image_path': image_path, 'status': 'error', 'error': str(e)})
        
        return results

    def visualize(self, image_path, threshold=0.5, save_path=None):
        """Visualizza i risultati della predizione"""
        prob_mask, binary_mask, original_image = self.predict(image_path, threshold)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle(f'Lane Detection: {os.path.basename(image_path)}', fontsize=16)
        
        axes[0, 0].imshow(original_image)
        axes[0, 0].set_title('Immagine Originale')
        axes[0, 0].axis('off')
        
        im1 = axes[0, 1].imshow(prob_mask, cmap='viridis', vmin=0, vmax=1)
        axes[0, 1].set_title('Mappa di Probabilità')
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)
        
        axes[1, 0].imshow(binary_mask, cmap='gray')
        axes[1, 0].set_title(f'Mask Binaria (threshold={threshold})')
        axes[1, 0].axis('off')
        
        overlay_image = original_image.copy().astype(float) / 255.0
        red_overlay = np.zeros_like(overlay_image)
        red_overlay[:, :, 0] = binary_mask
        axes[1, 1].imshow(overlay_image)
        axes[1, 1].imshow(red_overlay, alpha=0.6)
        axes[1, 1].set_title('Overlay (Corsie in Rosso)')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Visualizzazione salvata in: {save_path}")
        
        plt.show()
        return prob_mask, binary_mask

    def save_predictions(self, image_paths, output_dir, threshold=0.5, format='png'):
        """Salva le predizioni in una directory"""
        os.makedirs(output_dir, exist_ok=True)
        results = self.predict_batch(image_paths, threshold)
        success_count = 0
        
        for result in results:
            if result['status'] == 'success':
                base_name = os.path.splitext(os.path.basename(result['image_path']))[0]
                
                mask_path = os.path.join(output_dir, f'{base_name}_mask.{format}')
                cv2.imwrite(mask_path, result['binary_mask'] * 255)
                
                prob_path = os.path.join(output_dir, f'{base_name}_prob.{format}')
                cv2.imwrite(prob_path, (result['prob_mask'] * 255).astype(np.uint8))
                
                success_count += 1
        
        print(f"✅ Predizioni salvate! ({success_count}/{len(results)} riuscite)")


def main():
    """Esempio di utilizzo con TorchScript"""
    
    # ==================== CONFIGURAZIONE ====================
    # Cambia questo percorso al tuo modello TorchScript
    MODEL_PATH = 'test08_nn/model_traced.pt'  # ← Modello TorchScript da Ubuntu
    
    TEST_IMAGE = 'image.png'
    TEST_IMAGES_DIR = './test_images'
    OUTPUT_DIR = './predictions'
    THRESHOLD = 0.5
    
    # ==================== CREAZIONE PREDICTOR ====================
    try:
        predictor = LanePredictor(
            model_path=MODEL_PATH,
            device='cuda'  # O 'cpu' se non hai GPU
        )
    except Exception as e:
        print(f"❌ Errore nell'inizializzazione: {e}")
        return
    
    print("\n" + "="*80)
    print("🤖 ROBOT LANE DETECTION - 640x480 (TorchScript)")
    print("="*80 + "\n")
    
    # ==================== PREDIZIONE SINGOLA IMMAGINE ====================
    if os.path.exists(TEST_IMAGE):
        print(f"🔮 Predizione singola su: {TEST_IMAGE}\n")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_path = os.path.join(OUTPUT_DIR, 'prediction_visualization.png')
        
        try:
            predictor.visualize(TEST_IMAGE, threshold=THRESHOLD, save_path=output_path)
        except Exception as e:
            print(f"❌ Errore nella predizione: {e}")
    else:
        print(f"⚠️ Immagine di test non trovata: {TEST_IMAGE}\n")
    
    # ==================== PREDIZIONE BATCH ====================
    if os.path.exists(TEST_IMAGES_DIR) and os.listdir(TEST_IMAGES_DIR):
        print(f"\n📂 Predizioni batch su: {TEST_IMAGES_DIR}\n")
        image_files = [
            os.path.join(TEST_IMAGES_DIR, f) 
            for f in os.listdir(TEST_IMAGES_DIR) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ]
        
        if image_files:
            predictor.save_predictions(
                image_files,
                output_dir=OUTPUT_DIR,
                threshold=THRESHOLD
            )
        else:
            print("⚠️ Nessuna immagine trovata nella directory")
    else:
        print(f"⚠️ Directory test non trovata: {TEST_IMAGES_DIR}")
    
    print("\n" + "="*80)
    print("✅ Completato!")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
