"""
Lane Predictor con misurazione FPS dell'inference
Versione estesa con benchmark e timing utilities
"""

import os
import time
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.serialization import safe_globals


class LanePredictor:
    """Classe per fare predizioni con U-Net addestrato - ROBOT 640x480 (supporta full model)"""
    
    def __init__(self, model_path, encoder='mobilenet_v2', encoder_weights=None, device=None):
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device if torch.cuda.is_available() else 'cpu'
        
        self.encoder = encoder
        self.encoder_weights = encoder_weights
        self.model_path = model_path
        
        # Dimensioni usate nel file originale
        self.input_height = 240
        self.input_width = 320
        
        # Prova a caricare il file: può essere un modello intero oppure uno state_dict / checkpoint
        try:
            # allowlist temporanea della classe Unet di segmentation_models_pytorch
            with safe_globals([smp.Unet]):
                loaded = torch.load(model_path, map_location=self.device, weights_only=False)
        except Exception as e:
            # fallback: prova comunque a caricare forzando weights_only=False
            # (usa solo se ti fidi del file)
            loaded = torch.load(model_path, map_location=self.device, weights_only=False)
        
        if isinstance(loaded, torch.nn.Module):
            # Full model salvato con torch.save(model, path)
            self.model = loaded
            print(f"Full model caricato direttamente da: {model_path}")
        else:
            # Se è un dict con metadata cerca la chiave 'model_state_dict', altrimenti prendi il dict diretto
            if isinstance(loaded, dict) and 'model_state_dict' in loaded:
                state = loaded['model_state_dict']
            else:
                state = loaded
            
            # Istanzia il modello con gli stessi argomenti usati in training (minimo necessario)
            self.model = smp.Unet(
                encoder_name=encoder,
                encoder_weights=None,  # i pesi allenati verranno caricati dal checkpoint
                classes=1,
                activation=None,
            )
            
            # Carica lo state_dict (prima strict=True, poi fallback strict=False)
            try:
                res = self.model.load_state_dict(state, strict=True)
                print(f"✅ State_dict caricato con strict=True. Missing: {getattr(res, 'missing_keys', None)} Unexpected: {getattr(res, 'unexpected_keys', None)}")
            except Exception as e:
                try:
                    res = self.model.load_state_dict(state, strict=False)
                    print(f"Loaded with strict=False. Missing: {res.missing_keys} Unexpected: {res.unexpected_keys}")
                except Exception as e2:
                    raise RuntimeError(f"Errore nel caricamento dello state_dict: {e2}")
        
        print(f"✅ Modello ricostruito e pesi caricati da: {model_path}")
        
        # Porta modello su device e setta eval
        self.model.to(self.device)
        self.model.eval()
        
        # Preprocessing (stesso usato in training)
        self.transform = A.Compose([
            A.Resize(height=self.input_height, width=self.input_width, interpolation=cv2.INTER_LINEAR, p=1.0),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0),
            ToTensorV2(),
        ])
    
    def predict(self, image_path, threshold=0.5, return_original=True, measure_time=False):
        """
        Predizione su singola immagine con opzione timing.
        
        Args:
            image_path: Path dell'immagine
            threshold: Soglia per la mask binaria
            return_original: Se ritornare l'immagine originale
            measure_time: Se misurare il tempo di inference
        
        Returns:
            Se measure_time=False: (prob_mask, binary_mask) o (prob_mask, binary_mask, original_image)
            Se measure_time=True: (prob_mask, binary_mask, original_image, inference_time, fps)
        """
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Immagine non trovata: {image_path}")
        
        original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_height, original_width = original_image.shape[:2]
        
        # Preprocessing
        sample = self.transform(image=image)
        image_tensor = sample['image'].unsqueeze(0).to(self.device).float()
        
        # Misura il tempo di inferenza (solo rete neurale)
        if measure_time:
            start_time = time.time()
        
        with torch.no_grad():
            prediction = self.model(image_tensor)
        
        if measure_time:
            # Sincronizza GPU se disponibile
            if self.device == 'cuda':
                torch.cuda.synchronize()
            
            inference_time = time.time() - start_time
            fps = 1.0 / inference_time if inference_time > 0 else 0
        
        # Postprocessing
        prob_map = torch.sigmoid(prediction).squeeze().cpu().numpy()
        binary_mask = (prob_map > threshold).astype(np.uint8)
        
        prob_mask_resized = cv2.resize(prob_map, (original_width, original_height), 
                                        interpolation=cv2.INTER_LINEAR)
        binary_mask_resized = cv2.resize(binary_mask, (original_width, original_height),
                                         interpolation=cv2.INTER_NEAREST)
        
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
    
    def benchmark_inference(self, image_path, num_iterations=100, warmup=10, measure_full_pipeline=False):
        """
        Esegue benchmark dell'inferenza per calcolare FPS medi.
        
        Args:
            image_path: Path dell'immagine di test
            num_iterations: Numero di iterazioni per il benchmark
            warmup: Numero di iterazioni di warmup (per GPU)
            measure_full_pipeline: Se True, misura anche preprocessing e postprocessing
        
        Returns:
            dict con statistiche: fps_mean, fps_std, inference_time_mean, etc.
        """
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Immagine non trovata: {image_path}")
        
        original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Preprocessing (fuori dal benchmark)
        sample = self.transform(image=image)
        image_tensor = sample['image'].unsqueeze(0).to(self.device).float()
        
        # Warmup (importante per GPU)
        print(f"Warmup: {warmup} iterazioni...")
        for _ in range(warmup):
            with torch.no_grad():
                _ = self.model(image_tensor)
        
        if self.device == 'cuda':
            torch.cuda.synchronize()
        
        # Benchmark
        print(f"Benchmark: {num_iterations} iterazioni...")
        inference_times = []
        pipeline_times = []
        
        for i in range(num_iterations):
            if measure_full_pipeline:
                # Misura tutto il pipeline
                pipeline_start = time.time()
            
            # Solo inference (rete neurale)
            inference_start = time.time()
            
            with torch.no_grad():
                prediction = self.model(image_tensor)
            
            # Sincronizza GPU se disponibile
            if self.device == 'cuda':
                torch.cuda.synchronize()
            
            inference_time = time.time() - inference_start
            inference_times.append(inference_time)
            
            # Postprocessing
            prob_map = torch.sigmoid(prediction).squeeze().cpu().numpy()
            binary_mask = (prob_map > 0.5).astype(np.uint8)
            
            if measure_full_pipeline:
                pipeline_time = time.time() - pipeline_start
                pipeline_times.append(pipeline_time)
            
            if (i + 1) % 20 == 0:
                print(f"  Completate {i + 1}/{num_iterations} iterazioni")
        
        # Calcola statistiche
        inference_times = np.array(inference_times)
        fps_values = 1.0 / inference_times
        
        results = {
            'inference_time_mean': np.mean(inference_times),
            'inference_time_std': np.std(inference_times),
            'inference_time_min': np.min(inference_times),
            'inference_time_max': np.max(inference_times),
            'inference_time_median': np.median(inference_times),
            'fps_mean': np.mean(fps_values),
            'fps_std': np.std(fps_values),
            'fps_min': np.min(fps_values),
            'fps_max': np.max(fps_values),
            'fps_median': np.median(fps_values),
            'device': self.device,
            'input_resolution': f"{self.input_width}x{self.input_height}",
            'num_iterations': num_iterations,
            'encoder': self.encoder,
        }
        
        if measure_full_pipeline and pipeline_times:
            pipeline_times = np.array(pipeline_times)
            pipeline_fps = 1.0 / pipeline_times
            results['pipeline_time_mean'] = np.mean(pipeline_times)
            results['pipeline_time_std'] = np.std(pipeline_times)
            results['pipeline_fps_mean'] = np.mean(pipeline_fps)
            results['pipeline_fps_std'] = np.std(pipeline_fps)
        
        # Stampa risultati
        self._print_benchmark_results(results)
        
        return results
    
    def _print_benchmark_results(self, results):
        """Stampa i risultati del benchmark in formato leggibile"""
        print("\n" + "="*70)
        print("RISULTATI BENCHMARK INFERENCE")
        print("="*70)
        print(f"Device: {results['device']}")
        print(f"Encoder: {results['encoder']}")
        print(f"Risoluzione input: {results['input_resolution']}")
        print(f"Numero iterazioni: {results['num_iterations']}")
        print(f"\nTempo di inferenza (rete neurale):")
        print(f"  Media:   {results['inference_time_mean']*1000:.2f} ms")
        print(f"  Std:     {results['inference_time_std']*1000:.2f} ms")
        print(f"  Mediana: {results['inference_time_median']*1000:.2f} ms")
        print(f"  Min:     {results['inference_time_min']*1000:.2f} ms")
        print(f"  Max:     {results['inference_time_max']*1000:.2f} ms")
        print(f"\nFPS (rete neurale):")
        print(f"  Media:   {results['fps_mean']:.2f} FPS ⭐")
        print(f"  Std:     {results['fps_std']:.2f} FPS")
        print(f"  Mediana: {results['fps_median']:.2f} FPS")
        print(f"  Min:     {results['fps_min']:.2f} FPS")
        print(f"  Max:     {results['fps_max']:.2f} FPS")
        
        if 'pipeline_time_mean' in results:
            print(f"\nTempo pipeline completo (preprocessing + inference + postprocessing):")
            print(f"  Media: {results['pipeline_time_mean']*1000:.2f} ms")
            print(f"  Std:   {results['pipeline_time_std']*1000:.2f} ms")
            print(f"\nFPS pipeline completo:")
            print(f"  Media: {results['pipeline_fps_mean']:.2f} FPS ⭐")
            print(f"  Std:   {results['pipeline_fps_std']:.2f} FPS")
        
        print("="*70 + "\n")
    
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
                print(f" [{i + 1}/{total}] {os.path.basename(image_path)}")
            except Exception as e:
                print(f" [{i + 1}/{total}] {os.path.basename(image_path)}: {e}")
                results.append({'image_path': image_path, 'status': 'error', 'error': str(e)})
        return results
    
    def visualize(self, image_path, threshold=0.5, save_path=None):
        """Visualizzazione predizione con matplotlib"""
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
    
    def save_predictions(self, image_paths, output_dir, threshold=0.5, save_visualization=True):
        """Salva predizioni batch"""
        os.makedirs(output_dir, exist_ok=True)
        results = self.predict_batch(image_paths, threshold)
        success_count = 0
        
        for result in results:
            if result['status'] == 'success':
                base_name = os.path.splitext(os.path.basename(result['image_path']))[0]
                success_count += 1
        
        print(f"✅ Predizioni salvate! ({success_count}/{len(results)} riuscite)")
        if save_visualization:
            print(f"📊 Visualizzazioni complete salvate in: {output_dir}")


def main():
    """Esempio di utilizzo con misurazione FPS"""
    
    # ==================== CONFIGURAZIONE ====================
    MODEL_PATH = 'test08_nn/checkpoints_and_models_v2/last_unet_finetuned.pth'
    ENCODER = 'mobilenet_v2'
    TEST_IMAGE = '/home/dfiumicelli/Documenti/turtlebot_dataset/test_images/image_20251031_103508_985.jpg'
    TEST_IMAGES_DIR = '/home/dfiumicelli/Documenti/turtlebot_dataset/test_images'
    OUTPUT_DIR = './predictions'
    THRESHOLD = 0.5
    
    # ==================== CREAZIONE PREDICTOR ====================
    predictor = LanePredictor(
        model_path=MODEL_PATH,
        encoder=ENCODER,
        encoder_weights=None,
        device='cuda'  # O 'cpu' se non hai GPU
    )
    
    print("\n" + "="*80)
    print("ROBOT LANE DETECTION - FPS BENCHMARK")
    print("="*80 + "\n")
    
    # ==================== PREDIZIONE SINGOLA CON TIMING ====================
    if os.path.exists(TEST_IMAGE):
        print(f"Predizione singola su: {TEST_IMAGE}\n")
        prob_mask, binary_mask, original_image, inf_time, fps = predictor.predict(
            TEST_IMAGE, threshold=THRESHOLD, measure_time=True
        )
        print(f"⏱️  Tempo inferenza: {inf_time*1000:.2f} ms")
        print(f"🎯 FPS: {fps:.2f} FPS\n")
    
    # ==================== BENCHMARK COMPLETO ====================
    if os.path.exists(TEST_IMAGE):
        print("\n" + "="*80)
        print("ESECUZIONE BENCHMARK...")
        print("="*80 + "\n")
        
        benchmark_results = predictor.benchmark_inference(
            TEST_IMAGE, 
            num_iterations=100, 
            warmup=10,
            measure_full_pipeline=True
        )
        
        # Salva risultati in file
        import json
        results_path = os.path.join(OUTPUT_DIR, 'benchmark_results.json')
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(results_path, 'w') as f:
            json.dump(benchmark_results, f, indent=2)
        print(f"📊 Risultati benchmark salvati in: {results_path}\n")
    
    # ==================== PREDIZIONE BATCH ====================
    if os.path.exists(TEST_IMAGES_DIR):
        test_images = [
            os.path.join(TEST_IMAGES_DIR, f)
            for f in os.listdir(TEST_IMAGES_DIR)
            if f.lower().endswith(('.jpg', '.png', '.jpeg'))
        ]
        
        if test_images:
            print(f"Predizione batch su {len(test_images)} immagini...")
            predictor.save_predictions(
                test_images,
                OUTPUT_DIR,
                threshold=THRESHOLD,
                save_visualization=True
            )
        else:
            print(f"Nessuna immagine trovata in {TEST_IMAGES_DIR}")
    else:
        print(f"Cartella non trovata: {TEST_IMAGES_DIR}")


if __name__ == '__main__':
    main()
