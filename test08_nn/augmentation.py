# augmentation.py - CORRETTA per Albumentations 1.4+

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

def get_training_augmentation_fisheye_heavy():

    train_transform = [
        #OPTICAL DISTORTION - Parametri CORRETTI
        A.OpticalDistortion(
            distort_limit=1.0,      # Distorsione più forte
            interpolation=cv2.INTER_LINEAR,
            p=0.7                   # 70% probabilità
        ),
        
        #GRID DISTORTION - Parametri CORRETTI
        A.GridDistortion(
            num_steps=5,
            distort_limit=0.5,      # Distorsione più forte
            interpolation=cv2.INTER_LINEAR,
            p=0.5
        ),
        
        #ELASTIC TRANSFORM - Parametri CORRETTI
        A.ElasticTransform(
            alpha=50,               # Intensità della deformazione
            sigma=5,                # Smoothness della deformazione
            interpolation=cv2.INTER_LINEAR,
            p=0.3
        ),
        
        # Random crop + resize - PARAMETRO CORRETTO: size
        A.Resize(height=480, width=640, interpolation=cv2.INTER_LINEAR, p=1.0),

        
        # Flip orizzontale
        A.HorizontalFlip(p=0.5),
        
        # Affine (non ShiftScaleRotate)
        A.Affine(
            translate_percent=0.1,
            scale=0.15,
            rotate=15,
            interpolation=cv2.INTER_LINEAR,
            p=0.7
        ),
        
        # Perspective
        A.Perspective(
            scale=(0.05, 0.1),
            keep_size=True,
            interpolation=cv2.INTER_LINEAR,
            p=0.3
        ),
        
        # Brightness/Contrast
        A.RandomBrightnessContrast(
            brightness_limit=0.3,
            contrast_limit=0.3,
            p=0.5
        ),
        
        # CLAHE per migliorare contrasto
        A.CLAHE(
            clip_limit=2.0,
            tile_grid_size=(8, 8),
            p=0.4
        ),
        
        # Blur
        A.GaussianBlur(
            blur_limit=(3, 5),
            p=0.2
        ),
        
        # Dropout
        A.CoarseDropout(
            num_holes_range=(1, 8),
            hole_height_range=(8, 32),
            hole_width_range=(8, 32),
            p=0.3
        ),
        
        # Weather
        A.RandomRain(
            slant_range=(-10, 10),
            drop_length=20,
            drop_width=1,
            drop_color=(200, 200, 200),
            blur_value=3,
            brightness_coefficient=0.8,
            p=0.1
        ),
        
        A.RandomFog(
            fog_coef_range=(0.3, 1),
            p=0.1
        ),
        
        # Normalizzazione ImageNet
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            max_pixel_value=255.0,
        ),
        
        ToTensorV2(),
    ]
    
    return A.Compose(train_transform)


def get_training_augmentation():

    train_transform = [
        # Fisheye distortion moderato
        A.OpticalDistortion(
            distort_limit=0.5,
            interpolation=cv2.INTER_LINEAR,
            p=0.5
        ),
        
        # Grid distortion
        A.GridDistortion(
            num_steps=5,
            distort_limit=0.3,
            interpolation=cv2.INTER_LINEAR,
            p=0.3
        ),
        
        # Random crop + resize
        A.RandomResizedCrop(
            size=(640, 480),        # ← PARAMETRO CORRETTO!
            scale=(0.8, 1.0),
            ratio=(0.66, 1.5),
            interpolation=cv2.INTER_LINEAR,
            p=0.5
        ),
        
        # Flip orizzontale
        A.HorizontalFlip(p=0.5),
        
        # Affine
        A.Affine(
            translate_percent=0.1,
            scale=0.15,
            rotate=15,
            interpolation=cv2.INTER_LINEAR,
            p=0.7
        ),
        
        # Perspective
        A.Perspective(
            scale=(0.05, 0.1),
            keep_size=True,
            interpolation=cv2.INTER_LINEAR,
            p=0.3
        ),
        
        # Brightness/Contrast
        A.RandomBrightnessContrast(
            brightness_limit=0.3,
            contrast_limit=0.3,
            p=0.5
        ),
        
        # CLAHE
        A.CLAHE(
            clip_limit=2.0,
            tile_grid_size=(8, 8),
            p=0.4
        ),
        
        # Blur
        A.GaussianBlur(
            blur_limit=(3, 5),
            p=0.2
        ),
        
        # Dropout
        A.CoarseDropout(
            num_holes_range=(1, 8),
            hole_height_range=(8, 32),
            hole_width_range=(8, 32),
            p=0.3
        ),
        
        # Normalizzazione
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            max_pixel_value=255.0,
        ),
        
        ToTensorV2(),
    ]
    
    return A.Compose(train_transform)


def get_validation_augmentation():
    
    val_transform = [
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            max_pixel_value=255.0,
        ),
        A.Resize(height=480, width=640, interpolation=cv2.INTER_LINEAR, p=1.0),

        ToTensorV2(),
    ]
    
    return A.Compose(val_transform)


