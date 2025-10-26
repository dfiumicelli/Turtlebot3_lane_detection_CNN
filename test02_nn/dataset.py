# dataset_COMBINED.py - COMBINA TRAINING + TEST SET

import os
import cv2
import numpy as np
from torch.utils.data import Dataset

class LaneSegmentationDataset(Dataset):
    """
    Dataset personalizzato che carica:
    - ✅ Training set (frames + lane-masks)
    - ✅ Test set (frames + lane-masks)
    
    Combina entrambi per avere più dati per il training.
    Il split 80/20 interno creerà nuovi train/val set.
    """
    
    def __init__(self, images_dir, masks_dir, test_images_dir=None, test_masks_dir=None, transform=None, preprocessing=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.preprocessing = preprocessing
        
        # ✅ Carica TRAINING SET
        self.images_fps = sorted([
            os.path.join(images_dir, image_id)
            for image_id in os.listdir(images_dir)
            if image_id.endswith(('.jpg', '.png', '.jpeg'))
        ])
        
        self.masks_fps = sorted([
            os.path.join(masks_dir, image_id)
            for image_id in os.listdir(masks_dir)
            if image_id.endswith(('.jpg', '.png', '.jpeg'))
        ])
        
        # ✅ Carica TEST SET (se fornito)
        if test_images_dir and test_masks_dir:
            print(f"\n📂 Caricamento TEST SET da: {test_images_dir}")
            
            test_images = sorted([
                os.path.join(test_images_dir, image_id)
                for image_id in os.listdir(test_images_dir)
                if image_id.endswith(('.jpg', '.png', '.jpeg'))
            ])
            
            test_masks = sorted([
                os.path.join(test_masks_dir, image_id)
                for image_id in os.listdir(test_masks_dir)
                if image_id.endswith(('.jpg', '.png', '.jpeg'))
            ])
            
            print(f"✅ Test set immagini trovate: {len(test_images)}")
            print(f"✅ Test set maschere trovate: {len(test_masks)}")
            
            # Combina training + test set
            self.images_fps.extend(test_images)
            self.masks_fps.extend(test_masks)
            
            print(f"\n✅ TOTALE IMMAGINI (Training + Test): {len(self.images_fps)}")
        else:
            print(f"⚠️ Test set NON fornito - usa solo training set")
        
        # Verifica numero immagini == numero maschere
        assert len(self.images_fps) == len(self.masks_fps), \
            f"Numero immagini ({len(self.images_fps)}) != numero maschere ({len(self.masks_fps)})"
    
    def __len__(self):
        return len(self.images_fps)
    
    def __getitem__(self, idx):
        # Leggi immagine RGB
        image = cv2.imread(self.images_fps[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Leggi mask (grayscale)
        mask = cv2.imread(self.masks_fps[idx], cv2.IMREAD_GRAYSCALE)
        
        # ✅ Binarizza ma NON dividere per 255!
        mask = (mask > 127).astype(np.float32)
        
        # Applica augmentation (se presente)
        if self.transform:
            sample = self.transform(image=image, mask=mask)
            image = sample['image']
            mask = sample['mask']
        
        # Applica preprocessing dell'encoder (se presente)
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image = sample['image']
            mask = sample['mask']
        
        return image, mask
