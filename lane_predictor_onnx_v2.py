import os
import cv2
import numpy as np
import torch
import onnxruntime as ort
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.serialization import safe_globals
from pathlib import Path


class LanePredictor:
    """Classe per fare predizioni con U-Net addestrato - Supporta PyTorch (.pth) e ONNX (.onnx)"""
    
    def __init__(self, model_path, encoder='mobilenet_v2', encoder_weights=None, device=None):
        """
        Carica il modello (PyTorch o ONNX).
        Auto-detect dal file extension: .onnx → ONNX, .pth → PyTorch
        """
        
        self.model_path = model_path
        self.encoder = encoder
        self.encoder_weights = encoder_weights
        self.input_height = 240
        self.input_width = 320
        self.model = None
        self.session = None
        self.model_type = None  # 'pytorch' o 'onnx'
        
        # Determina il tipo di modello dal file extension
        if model_path.endswith('.onnx'):
            self._load_onnx_model(model_path, device)
        else:
            self._load_pytorch_model(model_path, device)
        
        # Preprocessing (stesso per entrambi)
        self.transform = A.Compose([
            A.Resize(height=self.input_height, width=self.input_width, interpolation=cv2.INTER_LINEAR, p=1.0),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0),
            ToTensorV2(),
        ])
    
    def _load_pytorch_model(self, model_path, device=None):
        """Carica modello PyTorch (.pth)"""
        print(f"\n[PyTorch Model] Caricamento da: {model_path}")
        
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device if torch.cuda.is_available() else 'cpu'
        
        # Carica il file
        try:
            with safe_globals([smp.Unet]):
                loaded = torch.load(model_path, map_location=self.device, weights_only=False)
        except Exception as e:
            loaded = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Istanzia il modello
        if isinstance(loaded, torch.nn.Module):
            self.model = loaded
            print(f"✅ Full model caricato direttamente")
        else:
            # State dict
            if isinstance(loaded, dict) and 'model_state_dict' in loaded:
                state = loaded['model_state_dict']
            else:
                state = loaded
            
            self.model = smp.Unet(
                encoder_name=self.encoder,
                encoder_weights=None,
                classes=1,
                activation=None,
                decoder_dropout=0.2
            )
            
            try:
                res = self.model.load_state_dict(state, strict=True)
                print(f"✅ State dict caricato con strict=True")
            except Exception as e:
                try:
                    res = self.model.load_state_dict(state, strict=False)
                    print(f"✅ State dict caricato con strict=False")
                except Exception as e2:
                    raise RuntimeError(f"Errore nel caricamento dello state_dict: {e2}")
        
        self.model.to(self.device)
        self.model.eval()
        self.model_type = 'pytorch'
        print(f"✅ Device: {self.device}\n")
    
    def _load_onnx_model(self, model_path, device=None):
        """Carica modello ONNX (.onnx)"""
        print(f"\n[ONNX Model] Caricamento da: {model_path}")
        
        if not Path(model_path).exists():
            raise FileNotFoundError(f"File ONNX non trovato: {model_path}")
        
        # Session options per CPU (ottimizzato)
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4
        sess_options.inter_op_num_threads = 1
        
        try:
            self.session = ort.InferenceSession(model_path, sess_options)
            
            # Mostra info
            inputs = self.session.get_inputs()
            outputs = self.session.get_outputs()
            
            print(f"✅ Modello ONNX caricato")
            print(f"   Input: {inputs[0].name} {inputs[0].shape}")
            print(f"   Output: {outputs[0].name} {outputs[0].shape}")
            print(f"✅ Device: CPU (ONNX Runtime)\n")
            
            self.model_type = 'onnx'
            
        except Exception as e:
            raise RuntimeError(f"Errore caricamento ONNX: {e}")
    
    def predict(self, image_path, threshold=0.5, return_original=True):
        """Predizione su singola immagine"""
        
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Immagine non trovata: {image_path}")
        
        original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_height, original_width = original_image.shape[:2]
        
        # Preprocessing
        sample = self.transform(image=image)
        image_tensor = sample['image']
        
        # Inference
        if self.model_type == 'pytorch':
            # PyTorch inference
            image_tensor = image_tensor.unsqueeze(0).to(self.device).float()
            with torch.no_grad():
                prediction = self.model(image_tensor)
            prob_map = torch.sigmoid(prediction).squeeze().cpu().numpy()
        
        else:  # ONNX
            # ONNX inference
            image_tensor_np = image_tensor.unsqueeze(0).numpy().astype(np.float32)
            outputs = self.session.run(None, {'input': image_tensor_np})
            
            # Estrai output
            if isinstance(outputs, list):
                output_array = outputs[0][0, 0, :, :]
            else:
                output_array = outputs[0, 0, :, :]
            
            # Applica sigmoid se necessario
            prob_map = 1.0 / (1.0 + np.exp(-output_array))
        
        # Postprocessing
        binary_mask = (prob_map > threshold).astype(np.uint8)
        
        # Ridimensiona alle dimensioni originali
        prob_mask_resized = cv2.resize(prob_map, (original_width, original_height), 
                                       interpolation=cv2.INTER_LINEAR)
        binary_mask_resized = cv2.resize(binary_mask, (original_width, original_height),
                                        interpolation=cv2.INTER_NEAREST)
        
        if return_original:
            return prob_mask_resized, binary_mask_resized, original_image
        else:
            return prob_mask_resized, binary_mask_resized
    
    def predict_batch(self, image_paths, threshold=0.5):
        """Predizione su batch di immagini"""
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
                print(f"  [{i + 1}/{total}] ✅ {os.path.basename(image_path)}")
            except Exception as e:
                print(f"  [{i + 1}/{total}] ❌ {os.path.basename(image_path)}: {e}")
                results.append({'image_path': image_path, 'status': 'error', 'error': str(e)})
        
        return results
    
    def visualize(self, image_path, threshold=0.5, save_path=None):
        """Visualizzazione predizione"""
        prob_mask, binary_mask, original_image = self.predict(image_path, threshold)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle(f'Lane Detection ({self.model_type.upper()}): {os.path.basename(image_path)}', fontsize=16)
        
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
    
    def save_predictions(self, image_paths, output_dir, threshold=0.5, format='png', save_visualization=True):
        """Salva predizioni"""
        os.makedirs(output_dir, exist_ok=True)
        results = self.predict_batch(image_paths, threshold)
        success_count = 0
        
        for result in results:
            if result['status'] == 'success':
                base_name = os.path.splitext(os.path.basename(result['image_path']))[0]
                
                if save_visualization:
                    self._save_complete_visualization_opencv(
                        result['original_image'],
                        result['prob_mask'],
                        result['binary_mask'],
                        base_name,
                        output_dir,
                        threshold
                    )
                
                success_count += 1
        
        print(f"\n✅ Predizioni salvate! ({success_count}/{len(results)} riuscite)")
        if save_visualization:
            print(f"📊 Visualizzazioni salvate in: {output_dir}")
    
    def _save_complete_visualization_opencv(self, original_image, prob_mask, binary_mask, base_name, output_dir, threshold):
        """Salva visualizzazione completa usando OpenCV"""
        
        original_uint8 = (original_image * 255).astype(np.uint8) if original_image.max() <= 1 else original_image.astype(np.uint8)
        h, w = original_uint8.shape[:2]
        
        # Pannello 1: Immagine Originale
        panel1 = original_uint8.copy()
        
        # Pannello 2: Mappa di Probabilità
        prob_uint8 = (prob_mask * 255).astype(np.uint8)
        panel2 = cv2.applyColorMap(prob_uint8, cv2.COLORMAP_VIRIDIS)
        
        # Pannello 3: Mask Binaria
        binary_uint8 = (binary_mask * 255).astype(np.uint8)
        panel3 = cv2.cvtColor(binary_uint8, cv2.COLOR_GRAY2BGR)
        
        # Pannello 4: Overlay
        panel4 = original_uint8.copy()
        red_mask = np.zeros_like(panel4)
        red_mask[:, :, 2] = binary_mask * 255  # Canale rosso
        panel4 = cv2.addWeighted(panel4, 0.7, red_mask, 0.3, 0)
        
        # Combina
        top_row = np.hstack([panel1, panel2])
        bottom_row = np.hstack([panel3, panel4])
        combined = np.vstack([top_row, bottom_row])
        
        # Aggiungi testo
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        color = (255, 255, 255)
        
        cv2.putText(combined, 'Immagine Originale', (20, 30), font, font_scale, color, thickness)
        cv2.putText(combined, 'Mappa di Probabilita', (w + 20, 30), font, font_scale, color, thickness)
        cv2.putText(combined, f'Mask Binaria', (20, h + 30), font, font_scale, color, thickness)
        cv2.putText(combined, 'Overlay', (w + 20, h + 30), font, font_scale, color, thickness)
        
        # Salva
        viz_path = os.path.join(output_dir, f'{base_name}_visualization.png')
        cv2.imwrite(viz_path, combined)


def main():
    """Esempio di utilizzo"""
    
    # ==================== CONFIGURAZIONE ====================
    
    # Usa il modello ONNX (CAMBIA QUI)
    MODEL_PATH = 'unet_mobilenet_int8_working.onnx'  # ← Modello ONNX
    
    # Oppure usa il modello PyTorch
    # MODEL_PATH = 'test08_nn/checkpoints_and_models_v2/last_unet_finetuned.pth'  # ← Modello PyTorch
    
    ENCODER = 'mobilenet_v2'
    TEST_IMAGE = 'C:\\Users\\mfiumicelli\\Documents\\turtlebot_dataset\\test_images\\image_20251031_103508_985.jpg'
    TEST_IMAGES_DIR = 'C:\\Users\\mfiumicelli\\Documents\\turtlebot_dataset\\test_images'
    OUTPUT_DIR = './predictions'
    THRESHOLD = 0.5
    
    # ==================== CREAZIONE PREDICTOR ====================
    
    predictor = LanePredictor(
        model_path=MODEL_PATH,
        encoder=ENCODER,
        encoder_weights=None,
        device='cpu'  # o 'cuda' se hai GPU
    )
    
    print("\n" + "="*80)
    print("ROBOT LANE DETECTION")
    print(f"Modello: {os.path.basename(MODEL_PATH)} ({predictor.model_type.upper()})")
    print("="*80 + "\n")
    
    # ==================== PREDIZIONE SINGOLA IMMAGINE ====================
    
    if os.path.exists(TEST_IMAGE):
        print(f"Predizione singola su: {TEST_IMAGE}\n")
        output_path = os.path.join(OUTPUT_DIR, 'prediction_visualization.png')
        predictor.visualize(TEST_IMAGE, threshold=THRESHOLD, save_path=output_path)
    else:
        print(f"⚠️ File non trovato: {TEST_IMAGE}")
    
    # ==================== PREDIZIONE BATCH ====================
    
    if os.path.exists(TEST_IMAGES_DIR):
        test_images = [
            os.path.join(TEST_IMAGES_DIR, f)
            for f in os.listdir(TEST_IMAGES_DIR)
            if f.lower().endswith(('.jpg', '.png', '.jpeg'))
        ]
        
        if test_images:
            print(f"\nPredizione batch su {len(test_images)} immagini...")
            predictor.save_predictions(
                test_images,
                OUTPUT_DIR,
                threshold=THRESHOLD,
                save_visualization=True
            )
        else:
            print(f"⚠️ Nessuna immagine trovata in {TEST_IMAGES_DIR}")
    else:
        print(f"⚠️ Cartella non trovata: {TEST_IMAGES_DIR}")


if __name__ == '__main__':
    main()
