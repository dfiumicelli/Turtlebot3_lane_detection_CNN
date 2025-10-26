# dataset_MOLANE.py - Carica MoLane test/target + val/target

import os
import cv2
import numpy as np
from torch.utils.data import Dataset

class LaneSegmentationDataset(Dataset):
    """
    Dataset per MoLane che carica:
    - Validation set: /kaggle/input/carlane-benchmark/CARLANE/MoLane/data/val/target
    - Test set: /kaggle/input/carlane-benchmark/CARLANE/MoLane/data/test/target
    
    Con le maschere appena create:
    - /kaggle/working/molane_val_masks
    - /kaggle/working/molane_test_masks
    """
    
    def __init__(self, images_dir, masks_dir, transform=None, preprocessing=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.preprocessing = preprocessing
        self.img_size = (512, 512)
        
        # Carica RICORSIVAMENTE tutte le immagini dalle sottocartelle
        self.images_fps = []
        self.masks_fps = []
        
        for root, dirs, files in os.walk(images_dir):
            for file in sorted(files):
                if file.endswith(('.jpg', '.png', '.jpeg')):
                    img_path = os.path.join(root, file)
                    
                    # Mantieni il percorso relativo per trovare la maschera corrispondente
                    rel_path = os.path.relpath(img_path, images_dir)
                    
                    # Maschera corrisponde: stessa struttura, estensione .png
                    mask_rel_path = rel_path.replace('image', 'label').replace('.jpg', '.png')
                    mask_path = os.path.join(masks_dir, mask_rel_path)
                    
                    if os.path.exists(mask_path):
                        self.images_fps.append(img_path)
                        self.masks_fps.append(mask_path)
        
        print(f"✅ Dataset MoLane caricato: {len(self.images_fps)} immagini con maschere")
        
        assert len(self.images_fps) == len(self.masks_fps), \
            f"Numero immagini ({len(self.images_fps)}) != numero maschere ({len(self.masks_fps)})"
    
    def __len__(self):
        return len(self.images_fps)
    
    def __getitem__(self, idx):
        image = cv2.imread(self.images_fps[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(self.masks_fps[idx], cv2.IMREAD_GRAYSCALE)
        
        # ✅ RIDIMENSIONA PRIMA!
        image = cv2.resize(image, self.img_size, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)
        
        mask = (mask > 127).astype(np.float32)
        
        # ✅ ORA le dimensioni sono uguali!
        if self.transform:
            sample = self.transform(image=image, mask=mask)
        
        return image, mask
