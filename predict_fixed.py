# predict.py - VERSIONE CORRETTA

import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2


class LanePredictor:
    """Classe per fare predizioni con U-Net addestrato"""
    
    def __init__(self, model_path, encoder='efficientnet-b4', encoder_weights='imagenet', device=None):
        """
        Args:
            model_path (str): Path al modello salvato (.pth)
            encoder (str): Nome encoder usato durante training
            encoder_weights (str): Pesi encoder usati durante training
            device (str): 'cuda' o 'cpu' (auto-detect se None)
        """
        
        # Auto-detect device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device if torch.cuda.is_available() else 'cpu'
        
        self.encoder = encoder
        self.encoder_weights = encoder_weights
        self.model_path = model_path
        self.input_size = 512  # Dimensione input della rete
        
        # Crea modello
        self.model = smp.Unet(
            encoder_name=encoder,
            encoder_weights=None,  # Non caricare pesi ImageNet
            classes=1,
            activation='sigmoid',
        )
        
        # Carica pesi addestrati
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
        except Exception as e:
            raise RuntimeError(f"❌ Errore nel caricamento del modello: {e}")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Ottieni preprocessing function dal encoder
        self.preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, encoder_weights)
        
        # Pipeline di preprocessing
        self.transform = A.Compose([
            A.Resize(self.input_size, self.input_size),
            A.Lambda(image=self.preprocessing_fn),
            ToTensorV2(),
        ])
        
        print(f"✅ Modello caricato da: {model_path}")
        print(f"   Device: {self.device}")
        print(f"   Encoder: {encoder}")
        print(f"   Input size: {self.input_size}x{self.input_size}")
    
    def predict(self, image_path, threshold=0.5, return_original=True):
        """
        Predice la mask per una singola immagine
        
        Args:
            image_path (str): Path all'immagine
            threshold (float): Soglia per binarizzazione (default 0.5)
            return_original (bool): Se True, ritorna anche immagine originale
            
        Returns:
            prob_mask (np.array): Mappa di probabilità (0-1) dimensioni originali
            binary_mask (np.array): Mask binaria (0 o 1) dimensioni originali
            original_image (np.array): Immagine originale (se return_original=True)
        """
        
        # Carica immagine
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"❌ Immagine non trovata: {image_path}")
        
        # Salva dimensioni originali PRIMA di qualsiasi modifica
        original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_height, original_width = original_image.shape[:2]
        
        # Applica preprocessing e resize
        sample = self.transform(image=image)
        image_tensor = sample['image'].unsqueeze(0).to(self.device).float()
        
        # Predizione
        with torch.no_grad():
            prediction = self.model(image_tensor)
            prob_mask = prediction.squeeze().cpu().numpy()
        
        # Binarizza con threshold
        binary_mask = (prob_mask > threshold).astype(np.uint8)
        
        # ✅ CORRETTO: Ridimensiona alle dimensioni originali
        # cv2.resize usa (width, height) non (height, width)
        prob_mask_resized = cv2.resize(prob_mask, (original_width, original_height))
        binary_mask_resized = cv2.resize(binary_mask, (original_width, original_height))
        
        if return_original:
            return prob_mask_resized, binary_mask_resized, original_image
        else:
            return prob_mask_resized, binary_mask_resized
    
    def predict_batch(self, image_paths, threshold=0.5):
        """
        Predice su una lista di immagini
        
        Args:
            image_paths (list): Lista di path alle immagini
            threshold (float): Soglia per binarizzazione
            
        Returns:
            list: Lista di dizionari con risultati
        """
        
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
                print(f"   [{i+1}/{total}] ✅ {os.path.basename(image_path)}")
            except Exception as e:
                print(f"   [{i+1}/{total}] ❌ {os.path.basename(image_path)}: {e}")
                results.append({
                    'image_path': image_path,
                    'status': 'error',
                    'error': str(e)
                })
        
        return results
    
    def visualize(self, image_path, threshold=0.5, save_path=None):
        """
        Visualizza risultato predizione
        
        Args:
            image_path (str): Path all'immagine
            threshold (float): Soglia binarizzazione
            save_path (str): Path dove salvare visualizzazione (opzionale)
            
        Returns:
            prob_mask, binary_mask
        """
        
        prob_mask, binary_mask, original_image = self.predict(image_path, threshold)
        
        # Crea figura con 4 subplot
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle(f'Lane Detection: {os.path.basename(image_path)}', fontsize=16)
        
        # 1. Immagine originale
        axes[0, 0].imshow(original_image)
        axes[0, 0].set_title('Immagine Originale', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')
        
        # 2. Mappa di probabilità
        im1 = axes[0, 1].imshow(prob_mask, cmap='viridis', vmin=0, vmax=1)
        axes[0, 1].set_title('Mappa di Probabilità', fontsize=12, fontweight='bold')
        axes[0, 1].axis('off')
        cbar1 = plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)
        cbar1.set_label('Probabilità', rotation=270, labelpad=15)
        
        # 3. Mask binaria
        axes[1, 0].imshow(binary_mask, cmap='gray')
        axes[1, 0].set_title(f'Mask Binaria (threshold={threshold})', fontsize=12, fontweight='bold')
        axes[1, 0].axis('off')
        
        # 4. Overlay su immagine originale
        overlay_image = original_image.copy().astype(float) / 255.0
        # Crea overlay in rosso
        red_overlay = np.zeros_like(overlay_image)
        red_overlay[:, :, 0] = binary_mask  # Canale rosso
        
        axes[1, 1].imshow(overlay_image)
        axes[1, 1].imshow(red_overlay, alpha=0.6)
        axes[1, 1].set_title('Overlay (Corsie in Rosso)', fontsize=12, fontweight='bold')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Visualizzazione salvata in: {save_path}")
        
        plt.show()
        
        return prob_mask, binary_mask
    
    def save_predictions(self, image_paths, output_dir, threshold=0.5, format='png'):
        """
        Salva predizioni su disco
        
        Args:
            image_paths (list): Lista di path alle immagini
            output_dir (str): Cartella dove salvare
            threshold (float): Soglia binarizzazione
            format (str): Formato salvataggio ('png' o 'jpg')
        """
        
        os.makedirs(output_dir, exist_ok=True)
        
        results = self.predict_batch(image_paths, threshold)
        
        print(f"\n💾 Salvataggio predizioni in: {output_dir}")
        
        for i, result in enumerate(results):
            if result['status'] == 'success':
                base_name = os.path.splitext(os.path.basename(result['image_path']))[0]
                
                # Salva mask binaria
                mask_path = os.path.join(output_dir, f'{base_name}_mask.{format}')
                cv2.imwrite(mask_path, result['binary_mask'] * 255)
                
                # Salva mappa di probabilità
                prob_path = os.path.join(output_dir, f'{base_name}_prob.{format}')
                cv2.imwrite(prob_path, (result['prob_mask'] * 255).astype(np.uint8))
        
        print(f"✅ Predizioni salvate!")


def main():
    """Esempio di utilizzo"""
    
    # ==================== CONFIGURAZIONE ====================
    MODEL_PATH = 'C:\\Users\\mfiumicelli\\downloads\\test05_nn\\unet_finetuned_molane_weights.pth'
    ENCODER = 'efficientnet-b4'
    ENCODER_WEIGHTS = 'imagenet'
    TEST_IMAGE = 'C:\\Users\\mfiumicelli\\Downloads\\strada9.png'
    TEST_IMAGES_DIR = 'C:\\Users\\mfiumicelli\\Downloads\\dataset_TuSimple\\tusimple_preprocessed\\test\\frames'
    OUTPUT_DIR = './'
    THRESHOLD = 0.5
    
    # ==================== CREAZIONE PREDICTOR ====================
    
    predictor = LanePredictor(
        model_path=MODEL_PATH,
        encoder=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        device='cuda'  # O 'cpu' se non hai GPU
    )
    
    # ==================== PREDIZIONE SINGOLA IMMAGINE ====================
    
    if os.path.exists(TEST_IMAGE):
        print(f"\n🔮 Predizione singola su: {TEST_IMAGE}")
        output_path = os.path.join(OUTPUT_DIR, 'prediction_visualization.png')
        predictor.visualize(TEST_IMAGE, threshold=THRESHOLD, save_path=output_path)
    
    # ==================== PREDIZIONE BATCH ====================
    """
    if os.path.exists(TEST_IMAGES_DIR):
        test_images = [
            os.path.join(TEST_IMAGES_DIR, f)
            for f in os.listdir(TEST_IMAGES_DIR)
            if f.lower().endswith(('.jpg', '.png', '.jpeg'))
        ]
        
        if test_images:
            print(f"\n📁 Predizione batch su {len(test_images)} immagini...")
            predictor.save_predictions(test_images, OUTPUT_DIR, threshold=THRESHOLD)
        else:
            print(f"⚠️ Nessuna immagine trovata in {TEST_IMAGES_DIR}")
    else:
        print(f"⚠️ Cartella non trovata: {TEST_IMAGES_DIR}")
    """

if __name__ == '__main__':
    main()
