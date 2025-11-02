import os
import cv2
import numpy as np
from torch.utils.data import Dataset

class LaneSegmentationDataset(Dataset):

    def __init__(self, images_dir, masks_dir, test_images_dir=None, test_masks_dir=None, 
                 transform=None, preprocessing=None, img_size=(640, 480)):
        
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.preprocessing = preprocessing
        self.img_size = img_size
        
        #Carica TRAINING SET
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
        
        #Carica TEST SET (se fornito)
        if test_images_dir and test_masks_dir:
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
            
            self.images_fps.extend(test_images)
            self.masks_fps.extend(test_masks)
        
        print(f"Dataset caricato:")
        print(f"   • Immagini: {len(self.images_fps)}")
        print(f"   • Target size: {self.img_size[0]}x{self.img_size[1]} (robot compatible)")
        
        assert len(self.images_fps) == len(self.masks_fps), \
            f"Mismatch: {len(self.images_fps)} immagini != {len(self.masks_fps)} maschere"
    
    def __len__(self):
        return len(self.images_fps)
    
    def __getitem__(self, idx):
        # Leggi immagine
        image = cv2.imread(self.images_fps[idx])
        
        if image is None:
            print(f"Non riesco a leggere: {self.images_fps[idx]}")
            return self.__getitem__(0)
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Leggi maschera
        mask = cv2.imread(self.masks_fps[idx], cv2.IMREAD_GRAYSCALE)
        
        if mask is None:
            print(f"Non riesco a leggere: {self.masks_fps[idx]}")
            return self.__getitem__(0)

        # Original: 1280x720 → Target: 640x480
        image = cv2.resize(image, self.img_size, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)
        
        # Binarizza maschera
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
