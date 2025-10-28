# augmentation_FISHEYE.py - Con supporto per distorsione fisheye

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

def get_training_augmentation_fisheye():
    """
    ✅ Augmentation per FISHEYE camera
    Aggiunge distorsioni ottiche per simulare l'effetto fisheye
    """
    
    train_transform = [
        # Resize base
        A.Resize(512, 512),
        
        # ⭐ OPTICAL DISTORTION - Simula fisheye e altre distorsioni ottiche
        A.OpticalDistortion(
            distort_limit=0.5,      # Range di distorsione (-0.5 a 0.5)
            shift_limit=0.2,        # Shift del centro della distorsione
            interpolation=cv2.INTER_LINEAR,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            mask_value=0,
            p=0.5                   # 50% probabilità
        ),
        
        # ⭐ GRID DISTORTION - Distorsione della griglia (effetto barrel/pincushion)
        A.GridDistortion(
            num_steps=5,            # Numero di step della griglia
            distort_limit=0.3,      # Limite di distorsione
            interpolation=cv2.INTER_LINEAR,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            mask_value=0,
            p=0.3                   # 30% probabilità
        ),
        
        # RandomResizedCrop
        A.RandomResizedCrop(
            size=(512, 512),
            scale=(0.8, 1.0),
            ratio=(0.9, 1.1),
            interpolation=cv2.INTER_LINEAR,
            p=0.5
        ),
        
        # Flip orizzontale
        A.HorizontalFlip(p=0.5),
        
        # ShiftScaleRotate
        A.Affine(
            shift_limit=0.1,
            scale_limit=0.15,
            rotate_limit=15,
            interpolation=cv2.INTER_LINEAR,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            mask_value=0,
            p=0.7
        ),
        
        # Perspective
        A.Perspective(
            scale=(0.05, 0.1),
            keep_size=True,
            fit_output=False,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            mask_value=0,
            interpolation=cv2.INTER_LINEAR,
            p=0.3
        ),
        
        # RandomBrightnessContrast
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
        
        # GaussianBlur
        A.GaussianBlur(
            blur_limit=(3, 5),
            p=0.2
        ),
        
        # CoarseDropout
        A.CoarseDropout(
            num_holes_range=(1, 8),
            hole_height_range=(8, 32),
            hole_width_range=(8, 32),
            fill="random_uniform",
            mask_value=0,
            p=0.3
        ),
        
        # RandomRain
        A.RandomRain(
            slant_range=(-10, 10),
            drop_length=20,
            drop_width=1,
            drop_color=(200, 200, 200),
            blur_value=3,
            brightness_coefficient=0.8,
            rain_type="drizzle",
            p=0.1
        ),
        
        # RandomFog
        A.RandomFog(
            alpha_coef=0.08,
            fog_coef_range=(0.3, 1),
            p=0.1
        ),
        
        # Normalizzazione ImageNet
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            max_pixel_value=255.0,
        ),
        
        # Conversione a PyTorch format
        ToTensorV2(),
    ]
    
    return A.Compose(train_transform)


def get_training_augmentation_fisheye_heavy():
    """
    ✅ Augmentation AGGRESSIVA per FISHEYE estremo
    Usa quando il fisheye del robot è molto marcato
    """
    
    train_transform = [
        # Resize base
        A.Resize(512, 512),
        
        # ⭐ OPTICAL DISTORTION HEAVY - Fisheye molto marcato
        A.OpticalDistortion(
            distort_limit=1.0,      # Distorsione più forte
            shift_limit=0.3,
            interpolation=cv2.INTER_LINEAR,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            mask_value=0,
            p=0.7                   # 70% probabilità
        ),
        
        # ⭐ GRID DISTORTION HEAVY
        A.GridDistortion(
            num_steps=5,
            distort_limit=0.5,      # Distorsione più forte
            interpolation=cv2.INTER_LINEAR,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            mask_value=0,
            p=0.5
        ),
        
        # ⭐ ELASTIC TRANSFORM - Deformazioni elastiche
        A.ElasticTransform(
            alpha=50,               # Intensità della deformazione
            sigma=5,                # Smoothness della deformazione
            interpolation=cv2.INTER_LINEAR,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            mask_value=0,
            p=0.3
        ),
        
        # Flip orizzontale
        A.HorizontalFlip(p=0.5),
        
        # ShiftScaleRotate
        A.Affine(
            shift_limit=0.1,
            scale_limit=0.15,
            rotate_limit=15,
            interpolation=cv2.INTER_LINEAR,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            mask_value=0,
            p=0.7
        ),
        
        # RandomBrightnessContrast
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
        
        # Normalizzazione ImageNet
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            max_pixel_value=255.0,
        ),
        
        # Conversione a PyTorch format
        ToTensorV2(),
    ]
    
    return A.Compose(train_transform)


def get_training_augmentation_improved():
    """
    ✅ VERSIONE STANDARD (senza fisheye)
    Data augmentation per lane detection
    """
    
    train_transform = [
        # Resize base
        A.Resize(512, 512),
        
        # RandomResizedCrop
        A.RandomResizedCrop(
            size=(512, 512),
            scale=(0.8, 1.0),
            ratio=(0.9, 1.1),
            interpolation=cv2.INTER_LINEAR,
            p=0.5
        ),
        
        # Flip orizzontale
        A.HorizontalFlip(p=0.5),
        
        # ShiftScaleRotate
        A.Affine(
            shift_limit=0.1,
            scale_limit=0.15,
            rotate_limit=15,
            interpolation=cv2.INTER_LINEAR,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            mask_value=0,
            p=0.7
        ),
        
        # Perspective
        A.Perspective(
            scale=(0.05, 0.1),
            keep_size=True,
            fit_output=False,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            mask_value=0,
            interpolation=cv2.INTER_LINEAR,
            p=0.3
        ),
        
        # RandomBrightnessContrast
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
        
        # GaussianBlur
        A.GaussianBlur(
            blur_limit=(3, 5),
            p=0.2
        ),
        
        # CoarseDropout
        A.CoarseDropout(
            num_holes_range=(1, 8),
            hole_height_range=(8, 32),
            hole_width_range=(8, 32),
            fill="random_uniform",
            mask_value=0,
            p=0.3
        ),
        
        # RandomRain
        A.RandomRain(
            slant_range=(-10, 10),
            drop_length=20,
            drop_width=1,
            drop_color=(200, 200, 200),
            blur_value=3,
            brightness_coefficient=0.8,
            rain_type="drizzle",
            p=0.1
        ),
        
        # RandomFog
        A.RandomFog(
            alpha_coef=0.08,
            fog_coef_range=(0.3, 1),
            p=0.1
        ),
        
        # Normalizzazione ImageNet
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            max_pixel_value=255.0,
        ),
        
        # Conversione a PyTorch format
        ToTensorV2(),
    ]
    
    return A.Compose(train_transform)


def get_validation_augmentation():
    """
    Augmentation per validation - solo resize + normalize
    """
    
    val_transform = [
        A.Resize(512, 512),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ]
    
    return A.Compose(val_transform)


# ==================== TESTING ====================

if __name__ == '__main__':
    import numpy as np
    
    print("\n" + "="*80)
    print("🔍 TESTING FISHEYE AUGMENTATION")
    print("="*80 + "\n")
    
    # Immagine dummy
    dummy_image = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    dummy_mask = np.random.randint(0, 2, (720, 1280), dtype=np.float32)
    
    try:
        # Test fisheye moderate
        print("📊 Testing Fisheye Moderate Augmentation...")
        fisheye_aug = get_training_augmentation_fisheye()
        augmented = fisheye_aug(image=dummy_image, mask=dummy_mask)
        print(f"   Input shape: {dummy_image.shape}")
        print(f"   Output shape: {augmented['image'].shape}")
        print(f"   ✅ Fisheye moderate OK!\n")
        
        # Test fisheye heavy
        print("📊 Testing Fisheye Heavy Augmentation...")
        fisheye_heavy_aug = get_training_augmentation_fisheye_heavy()
        augmented_heavy = fisheye_heavy_aug(image=dummy_image, mask=dummy_mask)
        print(f"   Output shape: {augmented_heavy['image'].shape}")
        print(f"   ✅ Fisheye heavy OK!\n")
        
        # Test standard
        print("📊 Testing Standard Augmentation...")
        standard_aug = get_training_augmentation_improved()
        augmented_std = standard_aug(image=dummy_image, mask=dummy_mask)
        print(f"   Output shape: {augmented_std['image'].shape}")
        print(f"   ✅ Standard OK!\n")
        
        print("="*80)
        print("✅ TUTTI I TEST COMPLETATI CON SUCCESSO!")
        print("="*80)
        print("\n📝 COME USARE:")
        print("1. Per fisheye moderato: get_training_augmentation_fisheye()")
        print("2. Per fisheye marcato: get_training_augmentation_fisheye_heavy()")
        print("3. Per training standard: get_training_augmentation_improved()")
        print("4. Per validation: get_validation_augmentation_improved()")
        
    except Exception as e:
        print(f"\n❌ ERRORE: {e}")
        import traceback
        traceback.print_exc()
