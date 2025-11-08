#!/usr/bin/env python3
"""
Predictor per U-Net 3-CLASSI - SHAPE-BASED (RGB + Sobel Edges)
✅ Riconosce CONTINUA vs DISCONTINUA in base alla FORMA, NON al colore
✅ Input: RGB + Sobel Edges (4 canali)
✅ Robustezza assoluta a variazioni di colore
✅ FIX: Usa Sobel edges al posto di HSV Hue!
"""

import os
import cv2
import numpy as np
import torch
import onnxruntime as ort
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path


class UNetWith4Channels(torch.nn.Module):
    """U-Net che prende 4 canali in input (RGB + Edges)"""
    
    def __init__(self, encoder_name='mobilenet_v2', num_classes=3, dropout=0.2):
        super().__init__()
        
        self.unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=3,
            classes=num_classes,
            activation=None,
            decoder_dropout=dropout,
        )
        
        self.conv_4to3 = torch.nn.Conv2d(4, 3, kernel_size=1, padding=0)
    
    def forward(self, x):
        x = self.conv_4to3(x)
        x = self.unet(x)
        return x


class LanePredictor3ClassShapeBased:
    """Predictor U-Net 3-classi SHAPE-BASED (RGB + Sobel Edges)"""
    
    def __init__(self, model_path, encoder='mobilenet_v2', device=None):
        """Carica il modello (.pth o .onnx)"""
        self.model_path = model_path
        self.encoder = encoder
        self.input_height = 240
        self.input_width = 320
        self.model = None
        self.session = None
        self.model_type = None
        
        self.num_classes = 3
        self.class_names = ['Background', 'Continua', 'Discontinua']
        self.class_colors = {
            0: (0, 0, 0),        # Nero
            1: (255, 255, 255),  # Bianco (Continua)
            2: (100, 100, 100)   # Grigio (Discontinua)
        }
        
        if model_path.endswith('.onnx'):
            self._load_onnx_model(model_path, device)
        else:
            self._load_pytorch_model(model_path, device)
        
        self.transform = A.Compose([
            A.Resize(height=self.input_height, width=self.input_width, 
                    interpolation=cv2.INTER_LINEAR, p=1.0),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), 
                       max_pixel_value=255.0),
            ToTensorV2(),
        ])
    
    def _load_pytorch_model(self, model_path, device=None):
        """Carica modello PyTorch (.pth) con 4 canali (RGB + Edges)"""
        print(f"\n[PyTorch Model] Caricamento da: {model_path}")
        
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device if torch.cuda.is_available() else 'cpu'
        
        try:
            loaded = torch.load(model_path, map_location=self.device, weights_only=False)
        except Exception as e:
            loaded = torch.load(model_path, map_location=self.device, weights_only=False)
        
        if isinstance(loaded, torch.nn.Module):
            self.model = loaded
            print(f"✅ Full model caricato direttamente")
        else:
            if isinstance(loaded, dict) and 'model_state_dict' in loaded:
                state = loaded['model_state_dict']
            else:
                state = loaded
            
            self.model = UNetWith4Channels(
                encoder_name=self.encoder,
                num_classes=self.num_classes,
                dropout=0.2
            )
            
            try:
                self.model.load_state_dict(state, strict=True)
                print(f"✅ State dict caricato (strict=True)")
            except Exception as e:
                try:
                    self.model.load_state_dict(state, strict=False)
                    print(f"✅ State dict caricato (strict=False)")
                except Exception as e2:
                    raise RuntimeError(f"Errore caricamento state_dict: {e2}")
        
        self.model.to(self.device)
        self.model.eval()
        self.model_type = 'pytorch'
        print(f"✅ Device: {self.device}\n")
    
    def _load_onnx_model(self, model_path, device=None):
        """Carica modello ONNX (.onnx) con 4 canali (RGB + Edges)"""
        print(f"\n[ONNX Model] Caricamento da: {model_path}")
        
        if not Path(model_path).exists():
            raise FileNotFoundError(f"File ONNX non trovato: {model_path}")
        
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        try:
            self.session = ort.InferenceSession(model_path, sess_options)
            
            inputs = self.session.get_inputs()
            outputs = self.session.get_outputs()
            print(f"✅ Modello ONNX caricato")
            print(f"   Input: {inputs[0].name} {inputs[0].shape}")
            print(f"   Output: {outputs[0].name} {outputs[0].shape}")
            print(f"✅ Device: CPU (ONNX Runtime)\n")
            
            self.model_type = 'onnx'
        except Exception as e:
            raise RuntimeError(f"Errore caricamento ONNX: {e}")
    
    def _compute_sobel_edges(self, image):
        """⭐ Calcola Sobel edges per riconoscimento di pattern"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Sobel edges (rileva i bordi)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edges = np.sqrt(sobelx**2 + sobely**2)
        
        # Normalizza
        edges = (edges / edges.max()).astype(np.float32) if edges.max() > 0 else edges.astype(np.float32)
        
        return edges
    
    def predict(self, image_path, return_original=True):
        """⭐ Predizione con 4 CANALI (RGB + Sobel Edges)"""
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Immagine non trovata: {image_path}")
        
        original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_height, original_width = original_image.shape[:2]
        
        # ⭐ PREPROCESSING RGB
        sample = self.transform(image=original_image)
        image_rgb = sample['image']  # (3, H, W)
        
        # ⭐ CALCOLA SOBEL EDGES (4° canale!)
        edges = self._compute_sobel_edges(original_image)
        
        # Ridimensiona edges
        edges_resized = cv2.resize(edges, (self.input_width, self.input_height), 
                                  interpolation=cv2.INTER_LINEAR)
        edges_tensor = torch.from_numpy(edges_resized).unsqueeze(0)  # (1, H, W)
        
        # ⭐ CONCATENA RGB + EDGES = 4 canali
        image_with_edges = torch.cat([image_rgb, edges_tensor], dim=0)  # (4, H, W)
        
        # Inference
        if self.model_type == 'pytorch':
            image_with_edges = image_with_edges.unsqueeze(0).to(self.device).float()
            with torch.no_grad():
                prediction = self.model(image_with_edges)
            
            class_map = torch.argmax(prediction, dim=1).squeeze().cpu().numpy()
        
        else:  # ONNX
            image_with_edges_np = image_with_edges.unsqueeze(0).numpy().astype(np.float32)
            outputs = self.session.run(None, {'input': image_with_edges_np})
            
            output_array = outputs[0][0]
            class_map = np.argmax(output_array, axis=0)
        
        # Ridimensiona alle dimensioni originali
        class_map_resized = cv2.resize(class_map.astype(np.uint8), 
                                      (original_width, original_height),
                                      interpolation=cv2.INTER_NEAREST)
        
        if return_original:
            return class_map_resized, original_image
        else:
            return class_map_resized
    
    def class_map_to_rgb(self, class_map):
        """Converte mappa di classi a immagine RGB colorata"""
        rgb_map = np.zeros((class_map.shape[0], class_map.shape[1], 3), dtype=np.uint8)
        
        for class_id, color in self.class_colors.items():
            mask = class_map == class_id
            rgb_map[mask] = color
        
        return rgb_map
    
    def predict_batch(self, image_paths):
        """Predizione su batch di immagini"""
        results = []
        total = len(image_paths)
        
        for i, image_path in enumerate(image_paths):
            try:
                class_map, original_image = self.predict(image_path)
                results.append({
                    'image_path': image_path,
                    'class_map': class_map,
                    'original_image': original_image,
                    'status': 'success'
                })
                print(f" [{i + 1}/{total}] ✅ {os.path.basename(image_path)}")
            except Exception as e:
                print(f" [{i + 1}/{total}] ❌ {os.path.basename(image_path)}: {e}")
                results.append({'image_path': image_path, 'status': 'error', 'error': str(e)})
        
        return results
    
    def visualize(self, image_path, save_path=None):
        """Visualizzazione predizione"""
        class_map, original_image = self.predict(image_path)
        rgb_map = self.class_map_to_rgb(class_map)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle(f'Lane Segmentation 3-Class SHAPE-BASED ({self.model_type.upper()}): {os.path.basename(image_path)}', 
                    fontsize=16)
        
        axes[0, 0].imshow(original_image)
        axes[0, 0].set_title('Immagine Originale')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(rgb_map)
        axes[0, 1].set_title('Segmentazione (Continua vs Discontinua)')
        axes[0, 1].axis('off')
        
        im = axes[1, 0].imshow(class_map, cmap='tab10', vmin=0, vmax=self.num_classes-1)
        axes[1, 0].set_title('Mappa Classi')
        axes[1, 0].axis('off')
        plt.colorbar(im, ax=axes[1, 0], ticks=range(self.num_classes))
        
        overlay = original_image.copy().astype(float) / 255.0
        overlay_rgb = rgb_map.astype(float) / 255.0
        blended = cv2.addWeighted(overlay, 0.7, overlay_rgb, 0.3, 0)
        axes[1, 1].imshow(blended)
        axes[1, 1].set_title('Overlay (70% img + 30% seg)')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Visualizzazione salvata: {save_path}")
        
        plt.show()
        return class_map, original_image
    
    def save_predictions(self, image_paths, output_dir, save_visualization=True):
        """Salva predizioni e visualizzazioni"""
        os.makedirs(output_dir, exist_ok=True)
        
        results = self.predict_batch(image_paths)
        success_count = 0
        
        for result in results:
            if result['status'] == 'success':
                base_name = os.path.splitext(os.path.basename(result['image_path']))[0]
                
                rgb_map = self.class_map_to_rgb(result['class_map'])
                rgb_path = os.path.join(output_dir, f'{base_name}_segmentation_rgb.png')
                cv2.imwrite(rgb_path, cv2.cvtColor(rgb_map, cv2.COLOR_RGB2BGR))
                
                class_path = os.path.join(output_dir, f'{base_name}_class_map.npy')
                np.save(class_path, result['class_map'])
                
                if save_visualization:
                    self._save_complete_visualization(
                        result['original_image'],
                        result['class_map'],
                        base_name,
                        output_dir
                    )
                
                success_count += 1
        
        print(f"\n✅ Predizioni salvate! ({success_count}/{len(results)} riuscite)")
        if save_visualization:
            print(f"📊 Visualizzazioni salvate in: {output_dir}")
    
    def _save_complete_visualization(self, original_image, class_map, base_name, output_dir):
        """Salva visualizzazione completa"""
        original_uint8 = (original_image * 255).astype(np.uint8) if original_image.max() <= 1 else original_image.astype(np.uint8)
        original_bgr = cv2.cvtColor(original_uint8, cv2.COLOR_RGB2BGR)
        
        h, w = original_bgr.shape[:2]
        
        panel1 = original_bgr.copy()
        
        rgb_map = self.class_map_to_rgb(class_map)
        panel2 = cv2.cvtColor(rgb_map, cv2.COLOR_RGB2BGR)
        
        class_map_uint8 = (class_map * 80).astype(np.uint8)
        try:
            panel3 = cv2.applyColorMap(class_map_uint8, cv2.COLORMAP_TURBO)
        except AttributeError:
            panel3 = cv2.applyColorMap(class_map_uint8, cv2.COLORMAP_JET)
        
        rgb_map_bgr = cv2.cvtColor(rgb_map, cv2.COLOR_RGB2BGR)
        panel4 = cv2.addWeighted(original_bgr, 0.7, rgb_map_bgr, 0.3, 0)
        
        top_row = np.hstack([panel1, panel2])
        bottom_row = np.hstack([panel3, panel4])
        combined = np.vstack([top_row, bottom_row])
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        color = (255, 255, 255)
        
        cv2.putText(combined, 'Immagine Originale', (20, 30), font, font_scale, color, thickness)
        cv2.putText(combined, 'Segmentazione SHAPE-BASED', (w + 20, 30), font, font_scale, color, thickness)
        cv2.putText(combined, 'Mappa Classi', (20, h + 30), font, font_scale, color, thickness)
        cv2.putText(combined, 'Overlay', (w + 20, h + 30), font, font_scale, color, thickness)
        
        legend_y = combined.shape[0] - 80
        cv2.putText(combined, 'Legenda:', (20, legend_y), font, font_scale, (255, 255, 255), thickness)
        cv2.putText(combined, '- Nero: Background', (20, legend_y + 30), font, 0.6, (0, 0, 0), 1)
        cv2.putText(combined, '- Bianco: Continua', (20, legend_y + 60), font, 0.6, (255, 255, 255), 1)
        cv2.putText(combined, '- Grigio: Discontinua', (20, legend_y + 90), font, 0.6, (100, 100, 100), 1)
        
        viz_path = os.path.join(output_dir, f'{base_name}_visualization.png')
        cv2.imwrite(viz_path, combined)


def main():
    """Esempio utilizzo"""
    
    # ⭐ MODELLO SHAPE-BASED (4 canali: RGB + Sobel Edges)
    MODEL_PATH = 'full_model_v4.onnx'  # o .pth
    ENCODER = 'mobilenet_v2'

    TEST_IMAGE = '/home/dfiumicelli/test_images/image_20251030_151942_149.jpg'  # ← Modifica con tuo percorso

    # Test su cartella
    TEST_IMAGES_DIR = '/home/dfiumicelli/Documenti/turtlebot_dataset/test_images'  # ← Modifica con tuo percorso

    OUTPUT_DIR = './predictions_3class'
    
    print("\n" + "="*80)
    print("🚀 ROBOT LANE SEGMENTATION (3-CLASSI SHAPE-BASED)")
    print("="*80 + "\n")
    print("✅ Riconosce: CONTINUA vs DISCONTINUA (in base alla forma)")
    print("✅ Robusto a variazioni di colore\n")
    
    predictor = LanePredictor3ClassShapeBased(
        model_path=MODEL_PATH,
        encoder=ENCODER,
        device='cpu'
    )
    
    if os.path.exists(TEST_IMAGE):
        print(f"Predizione: {TEST_IMAGE}\n")
        predictor.visualize(TEST_IMAGE, save_path=os.path.join(OUTPUT_DIR, 'prediction.png'))
    
    if os.path.exists(TEST_IMAGES_DIR):
        test_images = [
            os.path.join(TEST_IMAGES_DIR, f)
            for f in os.listdir(TEST_IMAGES_DIR)
            if f.lower().endswith(('.jpg', '.png', '.jpeg'))
        ]
        
        if test_images:
            print(f"\nBatch: {len(test_images)} immagini\n")
            predictor.save_predictions(test_images, OUTPUT_DIR, save_visualization=True)


if __name__ == '__main__':
    main()
