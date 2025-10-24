# dataset.py - VERSIONE CORRETTA (Fix Float64 → Float32)

import os
import cv2
import numpy as np
from torch.utils.data import Dataset


class LaneSegmentationDataset(Dataset):
    """
    Dataset personalizzato per lane detection con U-Net
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

        # Verifica che numero immagini == numero maschere
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

        # ✅ CORREZIONE CRITICA: Converti ESATTAMENTE a float32
        # Step 1: Binarizza a uint8 (0 o 255)
        mask = (mask > 127).astype(np.uint8)

        # Step 2: Converti a float32 esplicitamente
        mask = np.float32(mask)

        # Step 3: Normalizza a [0, 1]
        mask = mask / 255.0

        # Applica augmentation (se presente)
        if self.transform:
            sample = self.transform(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

            # ✅ Verifica e correggi il tipo di dato dopo augmentation
            # Albumentations potrebbe avere convertito a float64
            mask = np.float32(mask)

        # Applica preprocessing dell'encoder (se presente)
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

            # ✅ Verifica e correggi il tipo di dato dopo preprocessing
            mask = np.float32(mask)

        return image, mask