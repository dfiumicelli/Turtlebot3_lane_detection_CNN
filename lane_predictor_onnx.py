"""
Lane Predictor - Versione potenziata con supporto ONNX
Supporta sia modelli PyTorch che ONNX consolidati
Con misurazione FPS
"""

import os
import time
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
    """
    Classe per fare predizioni con U-Net addestrato.
    Supporta sia modelli PyTorch (.pth) che ONNX (.onnx).
    
    Inputs: 320x240
    Outputs: 1 (segmentazione binaria corsie)
    """
    
    def __init__(self, model_path, encoder='mobilenet_v2', encoder_weights=None, device=None, use_onnx=None):
        """
        Carica il modello (PyTorch o ONNX).
        
        Args:
            model_path: Path del modello (.pth o .onnx)
            encoder: Nome encoder (solo per PyTorch)
            encoder_weights: Pesi encoder (solo per PyTorch)
            device: 'cpu' o 'cuda' (solo per PyTorch)
            use_onnx: Se True forza ONNX, se False forza PyTorch, se None auto-detect
        """
        
        self.model_path = model_path
        self.encoder = encoder
        self.encoder_weights = encoder_weights
        self.input_height = 240
        self.input_width = 320
        self.model = None
        self.session = None
        self.model_type = None  # 'pytorch' o 'onnx'
        
        # Determina il tipo di modello
        if use_onnx is None:
            # Auto-detect dal file extension
            if model_path.endswith('.onnx'):
                use_onnx = True
            else:
                use_onnx = False
        
        if use_onnx:
            self._load_onnx_model(model_path)
        else:
            self._load_pytorch_model(model_path, device)
        
        # Setup preprocessing (stesso per entrambi)
        self.transform = A.Compose([
            A.Resize(height=self.input_height, width=self.input_width, interpolation=cv2.INTER_LINEAR, p=1.0),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0),
            ToTensorV2(),
        ])
    
    def _load_pytorch_model(self, model_path, device=None):
        """Carica modello PyTorch"""
        print(f"\n[PyTorch] Caricamento da: {model_path}")
        
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device if torch.cuda.is_available() else 'cpu'
        
        # Carica il file
        try:
            with safe_globals([smp.Unet]):
                loaded = torch.load(model_path, map_location=self.device, weights_only=False)
        except:
            loaded = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Istanzia il modello
        if isinstance(loaded, torch.nn.Module):
            self.model = loaded
            print(f"✓ Full model caricato direttamente")
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
                self.model.load_state_dict(state, strict=True)
                print(f"✓ State dict caricato con strict=True")
            except:
                self.model.load_state_dict(state, strict=False)
                print(f"✓ State dict caricato con strict=False")
        
        self.model.to(self.device)
        self.model.eval()
        self.model_type = 'pytorch'
        
        # Conta parametri
        param_count = sum(p.numel() for p in self.model.parameters())
        print(f"✓ Parametri: {param_count:,}")
        print(f"✓ Device: {self.device}\n")
    
    def _load_onnx_model(self, model_path):
        """Carica modello ONNX"""
        print(f"\n[ONNX] Caricamento da: {model_path}")
        
        if not Path(model_path).exists():
            raise FileNotFoundError(f"File ONNX non trovato: {model_path}")
        
        # Session options per CPU
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4
        sess_options.inter_op_num_threads = 1
        
        try:
            self.session = ort.InferenceSession(model_path, sess_options)
            
            # Mostra info
            inputs = self.session.get_inputs()
            outputs = self.session.get_outputs()
            
            print(f"✓ Modello caricato")
            print(f"✓ Input: {inputs[0].name} {inputs[0].shape}")
            print(f"✓ Output: {outputs[0].name} {outputs[0].shape}")
            print(f"✓ Device: CPU (ONNX Runtime optimized)\n")
            
            self.model_type = 'onnx'
            
        except Exception as e:
            raise RuntimeError(f"Errore caricamento ONNX: {e}")
    
    def predict(self, image_path, threshold=0.5, return_original=True, measure_time=False):
        """
        Predizione su singola immagine.
        
        Returns:
            Se measure_time=False: (prob_mask, binary_mask) o + original_image
            Se measure_time=True: + inference_time, fps
        """
        
        # Leggi immagine
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Immagine non trovata: {image_path}")
        
        original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_height, original_width = original_image.shape[:2]
        
        # Preprocessing
        sample = self.transform(image=image)
        image_tensor = sample['image']
        
        # Inference con misurazione tempo
        if measure_time:
            start_time = time.perf_counter()
        
        if self.model_type == 'pytorch':
            # PyTorch inference
            image_tensor = image_tensor.unsqueeze(0).to(self.device).float()
            with torch.no_grad():
                prediction = self.model(image_tensor)
            
            if self.device == 'cuda':
                torch.cuda.synchronize()
            
            prediction = prediction.squeeze().cpu().numpy()
        
        else:  # ONNX
            # ONNX inference
            image_tensor = image_tensor.unsqueeze(0).numpy().astype(np.float32)
            outputs = self.session.run(None, {'input': image_tensor})
            
            # Estrai output
            if isinstance(outputs, list):
                prediction = outputs[0][0, 0, :, :]  # (1, 1, H, W) -> (H, W)
            else:
                prediction = outputs[0, 0, :, :]
        
        if measure_time:
            inference_time = time.perf_counter() - start_time
            fps = 1.0 / inference_time if inference_time > 0 else 0
        
        # Postprocessing
        prob_map = 1.0 / (1.0 + np.exp(-prediction)) if self.model_type == 'pytorch' else prediction
        binary_mask = (prob_map > threshold).astype(np.uint8)
        
        # Ridimensiona alle dimensioni originali
        prob_mask_resized = cv2.resize(prob_map, (original_width, original_height),
                                       interpolation=cv2.INTER_LINEAR)
        binary_mask_resized = cv2.resize(binary_mask, (original_width, original_height),
                                         interpolation=cv2.INTER_NEAREST)
        
        # Return
        if measure_time:
            if return_original:
                return prob_mask_resized, binary_mask_resized, original_image, inference_time, fps
            else:
                return prob_mask_resized, binary_mask_resized, inference_time, fps
        else:
            if return_original:
                return prob_mask_resized, binary_mask_resized, original_image
            else:
                return prob_mask_resized, binary_mask_resized
    
    def benchmark_inference(self, image_path, num_iterations=100, warmup=10):
        """Benchmark dell'inferenza"""
        
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Immagine non trovata: {image_path}")
        
        # Preprocessing
        sample = self.transform(image=image)
        image_tensor = sample['image']
        
        # Warmup
        print(f"Warmup: {warmup} iterazioni...")
        for _ in range(warmup):
            if self.model_type == 'pytorch':
                image_t = image_tensor.unsqueeze(0).to(self.device).float()
                with torch.no_grad():
                    _ = self.model(image_t)
                if self.device == 'cuda':
                    torch.cuda.synchronize()
            else:
                image_t = image_tensor.unsqueeze(0).numpy().astype(np.float32)
                _ = self.session.run(None, {'input': image_t})
        
        # Benchmark
        print(f"Benchmark: {num_iterations} iterazioni...")
        inference_times = []
        
        for i in range(num_iterations):
            if self.model_type == 'pytorch':
                image_t = image_tensor.unsqueeze(0).to(self.device).float()
                start = time.perf_counter()
                with torch.no_grad():
                    _ = self.model(image_t)
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
            else:
                image_t = image_tensor.unsqueeze(0).numpy().astype(np.float32)
                start = time.perf_counter()
                _ = self.session.run(None, {'input': image_t})
                elapsed = time.perf_counter() - start
            
            inference_times.append(elapsed)
            
            if (i + 1) % 20 == 0:
                print(f"  {i + 1}/{num_iterations}")
        
        # Calcola statistiche
        inference_times = np.array(inference_times)
        fps_values = 1.0 / inference_times
        
        results = {
            'model_type': self.model_type,
            'inference_time_mean': np.mean(inference_times),
            'inference_time_std': np.std(inference_times),
            'inference_time_min': np.min(inference_times),
            'inference_time_max': np.max(inference_times),
            'fps_mean': np.mean(fps_values),
            'fps_std': np.std(fps_values),
            'fps_min': np.min(fps_values),
            'fps_max': np.max(fps_values),
            'num_iterations': num_iterations
        }
        
        # Stampa risultati
        print(f"\n{'='*70}")
        print(f"BENCHMARK INFERENCE - {self.model_type.upper()}")
        print(f"{'='*70}")
        print(f"Tempo medio: {results['inference_time_mean']*1000:.2f} ms")
        print(f"Std dev: {results['inference_time_std']*1000:.2f} ms")
        print(f"FPS medio: {results['fps_mean']:.2f} FPS ⭐")
        print(f"FPS min: {results['fps_min']:.2f} FPS")
        print(f"FPS max: {results['fps_max']:.2f} FPS")
        print(f"{'='*70}\n")
        
        return results
    
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
                print(f"  [{i + 1}/{total}] ✓ {os.path.basename(image_path)}")
            except Exception as e:
                print(f"  [{i + 1}/{total}] ✗ {os.path.basename(image_path)}: {e}")
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
            print(f"✓ Visualizzazione salvata in: {save_path}")
        
        plt.show()
        return prob_mask, binary_mask


# ============================================================================
# ESEMPIO DI UTILIZZO
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*80)
    print("LANE DETECTION - PYTORCH vs ONNX BENCHMARK")
    print("="*80)
    
    # Configurazione
    PYTORCH_MODEL = 'test08_nn/checkpoints_and_models_v2/last_unet_finetuned.pth'
    ONNX_MODEL = 'unet_mobilenet_int8_working_consolidated.onnx'
    TEST_IMAGE = 'C:\\Users\\mfiumicelli\\Documents\\turtlebot_dataset\\test_images\\image_20251031_103508_985.jpg'  # Cambia con il tuo percorso
    THRESHOLD = 0.5
    
    # ========== TEST PYTORCH ==========
    if os.path.exists(PYTORCH_MODEL):
        print("\n[1/2] Testing PyTorch model...")
        predictor_pytorch = LanePredictor(PYTORCH_MODEL, use_onnx=False)
        
        if os.path.exists(TEST_IMAGE):
            prob_mask, binary_mask, original_image, inf_time, fps = predictor_pytorch.predict(
                TEST_IMAGE, threshold=THRESHOLD, measure_time=True
            )
            print(f"⏱️ Tempo: {inf_time*1000:.2f} ms")
            print(f"🎯 FPS: {fps:.2f} FPS")
            
            # Benchmark
            benchmark_pt = predictor_pytorch.benchmark_inference(TEST_IMAGE, num_iterations=100)
    
    # ========== TEST ONNX ==========
    if os.path.exists(ONNX_MODEL):
        print("\n[2/2] Testing ONNX model...")
        predictor_onnx = LanePredictor(ONNX_MODEL, use_onnx=True)
        
        if os.path.exists(TEST_IMAGE):
            prob_mask, binary_mask, original_image, inf_time, fps = predictor_onnx.predict(
                TEST_IMAGE, threshold=THRESHOLD, measure_time=True
            )
            print(f"⏱️ Tempo: {inf_time*1000:.2f} ms")
            print(f"🎯 FPS: {fps:.2f} FPS")
            
            # Benchmark
            benchmark_onnx = predictor_onnx.benchmark_inference(TEST_IMAGE, num_iterations=100)
    
    print("\n" + "="*80)
    print("✓ Test completato!")
    print("="*80 + "\n")
