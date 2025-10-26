# augmentation.py - VERSIONE CON NORMALIZZAZIONE ImageNet

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

def get_training_augmentation():
    """
    Data augmentation per training set.
    ✅ INCLUDE: Normalizzazione ImageNet + ToTensorV2
    """
    train_transform = [
        A.Resize(256, 256),
        A.HorizontalFlip(p=0.5),
        A.Affine(
            scale=(0.9, 1.1),
            translate_percent=(0.0625, 0.0625),
            rotate=(-10, 10),
            p=0.5
        ),
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=1.0
            ),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=20,
                p=1.0
            ),
        ], p=0.5),
        A.GaussianBlur(blur_limit=3, p=0.2),
        
        # ✅ CRITICO: Normalizzazione ImageNet
        # Mean e Std di ImageNet per RGB
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
        ),
        
        # ✅ Conversione a PyTorch format (CxHxW)
        ToTensorV2(),
    ]
    
    return A.Compose(train_transform)


def get_validation_augmentation():
    """
    Augmentation per validation set.
    ✅ INCLUDE: Normalizzazione ImageNet + ToTensorV2
    """
    val_transform = [
        A.Resize(256, 256),
        
        # ✅ CRITICO: Normalizzazione ImageNet
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
        ),
        
        # ✅ Conversione a PyTorch format (CxHxW)
        ToTensorV2(),
    ]
    return A.Compose(val_transform)


def get_preprocessing(preprocessing_fn):
    """
    Preprocessing specifico dell'encoder.
    ⚠️ NON usare questo - lo facciamo direttamente con Normalize
    """
    _transform = [
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ]
    return A.Compose(_transform)


def get_heavy_augmentation():
    """
    Augmentation aggressiva per dataset piccoli.
    ✅ INCLUDE: Normalizzazione ImageNet + ToTensorV2
    """
    heavy_transform = [
        A.Resize(256, 256),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.Affine(
            scale=(0.8, 1.2),
            translate_percent=(0.1, 0.1),
            rotate=(-15, 15),
            shear=(-5, 5),
            p=0.7
        ),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=30, p=1.0),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1.0),
        ], p=0.7),
        A.OneOf([
            A.GaussianBlur(blur_limit=5, p=1.0),
            A.MotionBlur(blur_limit=5, p=1.0),
        ], p=0.3),
        
        # ✅ CRITICO: Normalizzazione ImageNet
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
        ),
        
        # ✅ Conversione a PyTorch format (CxHxW)
        ToTensorV2(),
    ]
    
    return A.Compose(heavy_transform)
