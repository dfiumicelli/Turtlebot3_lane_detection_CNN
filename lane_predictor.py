import os
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
                print(
                    f"✅ State_dict caricato con strict=True. Missing: {getattr(res, 'missing_keys', None)} Unexpected: {getattr(res, 'unexpected_keys', None)}")
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

    def predict(self, image_path, threshold=0.5, return_original=True):
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Immagine non trovata: {image_path}")

        original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_height, original_width = original_image.shape[:2]

        sample = self.transform(image=image)
        image_tensor = sample['image'].unsqueeze(0).to(self.device).float()

        with torch.no_grad():
            prediction = self.model(image_tensor)
            prob_map = torch.sigmoid(prediction).squeeze().cpu().numpy()

        binary_mask = (prob_map > threshold).astype(np.uint8)

        prob_mask_resized = cv2.resize(prob_map, (original_width, original_height), interpolation=cv2.INTER_LINEAR)
        binary_mask_resized = cv2.resize(binary_mask, (original_width, original_height),
                                         interpolation=cv2.INTER_NEAREST)

        if return_original:
            return prob_mask_resized, binary_mask_resized, original_image
        else:
            return prob_mask_resized, binary_mask_resized

    def predict_batch(self, image_paths, threshold=0.5):
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
        prob_mask, binary_mask, original_image = self.predict(image_path, threshold)

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle(f'Lane Detection: {os.path.basename(image_path)}', fontsize=16)

        axes[0, 0].imshow(original_image);
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

        os.makedirs(output_dir, exist_ok=True)
        results = self.predict_batch(image_paths, threshold)
        success_count = 0

        for result in results:
            if result['status'] == 'success':
                base_name = os.path.splitext(os.path.basename(result['image_path']))[0]

                # Salva visualizzazione completa (come nella predizione singola)
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

        print(f"✅ Predizioni salvate! ({success_count}/{len(results)} riuscite)")
        if save_visualization:
            print(f"📊 Visualizzazioni complete salvate in: {output_dir}")

    def _save_complete_visualization_opencv(self, original_image, prob_mask, binary_mask, base_name, output_dir,
                                            threshold):
        # Converti original_image (RGB) in uint8 se necessario
        original_uint8 = (original_image * 255).astype(
            np.uint8) if original_image.max() <= 1 else original_image.astype(np.uint8)

        h, w = original_uint8.shape[:2]

        # ========== PANNELLO 1: Immagine Originale ==========
        panel1 = original_uint8.copy()

        # ========== PANNELLO 2: Mappa di Probabilità (applicare colormap) ==========
        prob_uint8 = (prob_mask * 255).astype(np.uint8)
        panel2 = cv2.applyColorMap(prob_uint8, cv2.COLORMAP_VIRIDIS)

        # ========== PANNELLO 3: Mask Binaria ==========
        binary_uint8 = (binary_mask * 255).astype(np.uint8)
        panel3 = cv2.cvtColor(binary_uint8, cv2.COLOR_GRAY2BGR)

        # ========== PANNELLO 4: Overlay (rosso su original) ==========
        panel4 = original_uint8.copy()
        # Crea overlay rosso
        red_mask = np.zeros_like(panel4)
        red_mask[:, :, 0] = binary_mask * 255  # Canale R (in BGR, R è l'indice 2, ma OpenCV usa BGR)
        red_mask[:, :, 2] = binary_mask * 255  # Canale B blu diventa rosso in BGR
        panel4 = cv2.addWeighted(panel4, 0.7, red_mask, 0.3, 0)

        # ========== Ridimensiona tutti i pannelli alla stessa dimensione ==========
        panel_size = (w, h)

        # ========== Combina 2x2 ==========
        top_row = np.hstack([panel1, panel2])
        bottom_row = np.hstack([panel3, panel4])
        combined = np.vstack([top_row, bottom_row])

        # ========== Aggiungi titoli (testo su immagine) ==========
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        color = (255, 255, 255)  # Bianco

        # Titolo sopra ogni pannello
        cv2.putText(combined, 'Immagine Originale', (20, 30), font, font_scale, color, thickness)
        cv2.putText(combined, 'Mappa di Probabilita', (w + 20, 30), font, font_scale, color, thickness)
        cv2.putText(combined, f'Mask Binaria (threshold={threshold})', (20, h + 30), font, font_scale, color, thickness)
        cv2.putText(combined, 'Overlay (Corsie in Rosso)', (w + 20, h + 30), font, font_scale, color, thickness)

        # Titolo principale
        cv2.putText(combined, f'Lane Detection: {base_name}', (20, h * 2 + 50), font, 1.0, color, thickness)

        # ========== Salva con OpenCV (MOLTO PIU' VELOCE) ==========
        viz_path = os.path.join(output_dir, f'{base_name}_visualization.png')
        cv2.imwrite(viz_path, combined)


def main():
    """Esempio di utilizzo"""

    # ==================== CONFIGURAZIONE ====================

    MODEL_PATH = 'test08_nn/checkpoints_and_models_v2/last_unet_finetuned.pth'
    ENCODER = 'mobilenet_v2'

    TEST_IMAGE = 'C:\\Users\\mfiumicelli\\Documents\\turtlebot_dataset\\test_images\\image_20251031_103508_985.jpg'
    TEST_IMAGES_DIR = 'C:\\Users\\mfiumicelli\\Documents\\turtlebot_dataset\\test_images'

    OUTPUT_DIR = './predictions'
    THRESHOLD = 0.5

    # ==================== CREAZIONE PREDICTOR ====================

    predictor = LanePredictor(
        model_path=MODEL_PATH,
        encoder=ENCODER,
        encoder_weights=None,  # ← Modello già addestrato
        device='cuda'  # O 'cpu' se non hai GPU
    )

    print("\n" + "=" * 80)
    print("ROBOT LANE DETECTION")
    print("=" * 80 + "\n")

    # ==================== PREDIZIONE SINGOLA IMMAGINE ====================

    if os.path.exists(TEST_IMAGE):
        print(f"Predizione singola su: {TEST_IMAGE}\n")
        output_path = os.path.join(OUTPUT_DIR, 'prediction_visualization.png')
        predictor.visualize(TEST_IMAGE, threshold=THRESHOLD, save_path=output_path)

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
            print(f"Nessuna immagine trovata in {TEST_IMAGES_DIR}")
    else:
        print(f"Cartella non trovata: {TEST_IMAGES_DIR}")


if __name__ == '__main__':
    main()