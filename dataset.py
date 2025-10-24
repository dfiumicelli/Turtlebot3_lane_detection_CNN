# dataset.py
# Script per definire il Dataset PyTorch per lane detection

import os
import cv2
import numpy as np
from torch.utils.data import Dataset


class LaneSegmentationDataset(Dataset):
    """
    Dataset personalizzato per lane detection con U-Net
    
    Args:
        images_dir (str): Path alla cartella con le immagini
        masks_dir (str): Path alla cartella con le mask
        transform: Augmentation da applicare (albumentations)
        preprocessing: Preprocessing specifico dell'encoder (da SMP)
    """
    
    def __init__(self, images_dir, masks_dir, transform=None, preprocessing=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.preprocessing = preprocessing
        
        # Lista di tutti i file immagine
        self.images_fps = sorted([
            os.path.join(images_dir, image_id) 
            for image_id in os.listdir(images_dir)
            if image_id.endswith(('.jpg', '.png', '.jpeg'))
        ])
        
        # Lista di tutti i file mask
        self.masks_fps = sorted([
            os.path.join(masks_dir, image_id) 
            for image_id in os.listdir(masks_dir)
            if image_id.endswith(('.jpg', '.png', '.jpeg'))
        ])
        
        # Verifica che il numero di immagini e mask corrisponda
        assert len(self.images_fps) == len(self.masks_fps), \
            f"Numero immagini ({len(self.images_fps)}) != numero masks ({len(self.masks_fps)})"
        
        print(f"✅ Dataset inizializzato: {len(self.images_fps)} immagini trovate")
    
    def __len__(self):
        return len(self.images_fps)
    
    def __getitem__(self, idx):
        # Leggi immagine RGB
        image = cv2.imread(self.images_fps[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Leggi mask (grayscale)
        mask = cv2.imread(self.masks_fps[idx], cv2.IMREAD_GRAYSCALE)
        
        # Converti mask binaria: 0 (background) e 1 (lane)
        # Assumi che le corsie siano bianche (255) nella mask
        mask = (mask > 127).astype(np.uint8)  # Prima binarizza come uint8
        mask = mask.astype(np.float32)
        
        # Applica augmentation (se presente)
        if self.transform:
            sample = self.transform(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # Applica preprocessing dell'encoder (se presente)
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        return image, mask
    
    def get_sample(self, idx):
        """Ottieni un sample senza preprocessing per visualizzazione"""
        image = cv2.imread(self.images_fps[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[idx], cv2.IMREAD_GRAYSCALE)
        return image, mask
