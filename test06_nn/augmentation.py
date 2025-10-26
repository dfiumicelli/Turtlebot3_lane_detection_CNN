# augmentation_DEFINITIVA.py - BASATA DIRETTAMENTE SULLA DOCUMENTAZIONE UFFICIALE
# Tutti i parametri estratti e verificati dalla documentazione ufficiale di Albumentations
# Nessun parametro inventato o non verificato

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


def get_training_augmentation_improved():
    """
    ✅ VERSIONE DEFINITIVA - PARAMETRI DALLA DOCUMENTAZIONE UFFICIALE
    Data augmentation per lane detection
    """

    train_transform = [
        # Resize base
        A.Resize(512, 512),

        # RandomResizedCrop - parametri corretti
        A.RandomResizedCrop(
            size=(512, 512),
            scale=(0.8, 1.0),
            ratio=(0.9, 1.1),
            interpolation=cv2.INTER_LINEAR,
            p=0.5
        ),

        # Flip orizzontale
        A.HorizontalFlip(p=0.5),

        # ShiftScaleRotate - parametri verificati
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.15,
            rotate_limit=15,
            interpolation=cv2.INTER_LINEAR,
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
            fill_mask=0,
            p=0.7
        ),

        # Perspective - parametri dalla documentazione ufficiale
        # scale: distortion scale (0.05, 0.1)
        # keep_size: resize back to original (True)
        # fit_output: adjust plane size (False)
        # border_mode: padding mode (cv2.BORDER_CONSTANT)
        # fill: padding value (0)
        # fill_mask: padding for mask (0)
        A.Perspective(
            scale=(0.05, 0.1),
            keep_size=True,
            fit_output=False,
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
            fill_mask=0,
            interpolation=cv2.INTER_LINEAR,
            p=0.3
        ),

        # RandomBrightnessContrast - parametri verificati
        A.RandomBrightnessContrast(
            brightness_limit=0.3,
            contrast_limit=0.3,
            p=0.5
        ),

        # CLAHE - parametri verificati
        A.CLAHE(
            clip_limit=2.0,
            tile_grid_size=(8, 8),
            p=0.4
        ),

        # GaussianBlur - parametri verificati
        A.GaussianBlur(
            blur_limit=(3, 5),
            p=0.2
        ),

        # CoarseDropout - parametri dalla documentazione ufficiale
        # num_holes_range: tuple (min, max) number of holes
        # hole_height_range: tuple (min, max) height range
        # hole_width_range: tuple (min, max) width range
        # fill: fill value ('random_uniform' per colore casuale)
        # fill_mask: mask fill value (None per non cambiare mask)
        A.CoarseDropout(
            num_holes_range=(1, 8),
            hole_height_range=(8, 32),
            hole_width_range=(8, 32),
            fill="random_uniform",
            fill_mask=0,
            p=0.3
        ),

        # RandomRain - parametri dalla documentazione ufficiale
        # slant_range: tuple (min, max) for slant angle
        # drop_length: int or None - length of drops (None = auto)
        # drop_width: int - width of drops
        # drop_color: tuple RGB
        # blur_value: int - blur amount
        # brightness_coefficient: float (0, 1] - brightness multiplier
        # rain_type: "drizzle", "heavy", "torrential", or "default"
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

        # RandomFog - parametri dalla documentazione ufficiale
        # alpha_coef: float (0, 1] - transparency of fog
        # fog_coef_range: tuple (min, max) for fog intensity
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


def get_validation_augmentation_improved():
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


def get_moderate_augmentation():
    """
    ✅ AUGMENTATION MODERATA
    Versione più leggera se quella aggressiva causa problemi
    """

    train_transform = [
        A.Resize(512, 512),

        A.HorizontalFlip(p=0.5),

        A.ShiftScaleRotate(
            shift_limit=0.0625,
            scale_limit=0.1,
            rotate_limit=10,
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
            fill_mask=0,
            p=0.6
        ),

        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        ),

        A.CLAHE(
            clip_limit=2.0,
            tile_grid_size=(8, 8),
            p=0.3
        ),

        A.GaussianBlur(blur_limit=3, p=0.2),

        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            max_pixel_value=255.0,
        ),

        ToTensorV2(),
    ]

    return A.Compose(train_transform)


def get_light_augmentation():
    """
    ✅ AUGMENTATION LEGGERA
    Solo trasformazioni essenziali
    """

    train_transform = [
        A.Resize(512, 512),

        A.HorizontalFlip(p=0.5),

        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.1,
            rotate_limit=10,
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
            fill_mask=0,
            p=0.5
        ),

        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        ),

        A.CLAHE(
            clip_limit=2.0,
            tile_grid_size=(8, 8),
            p=0.3
        ),

        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            max_pixel_value=255.0,
        ),

        ToTensorV2(),
    ]

    return A.Compose(train_transform)


def get_test_time_augmentation():
    """
    ✅ Test Time Augmentation (TTA)
    Solo flip orizzontale - il più sicuro e efficace
    """

    tta_transforms = []

    # Original
    tta_transforms.append(A.Compose([
        A.Resize(512, 512),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ]))

    # Horizontal flip
    tta_transforms.append(A.Compose([
        A.Resize(512, 512),
        A.HorizontalFlip(p=1.0),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ]))

    return tta_transforms


# ==================== TESTING ====================
if __name__ == '__main__':
    import numpy as np

    print("✅ Test augmentation definitiva (parametri ufficiali)...\n")

    # Immagine e mask dummy
    dummy_image = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    dummy_mask = np.random.randint(0, 2, (720, 1280), dtype=np.float32)

    try:
        # Test training augmentation
        print("📊 Testing Training Augmentation (Full)...")
        train_aug = get_training_augmentation_improved()
        augmented = train_aug(image=dummy_image, mask=dummy_mask)

        print(f"   Input shape: {dummy_image.shape}")
        print(f"   Output image shape: {augmented['image'].shape}")
        print(f"   Output mask shape: {augmented['mask'].shape}")
        print(f"   Image dtype: {augmented['image'].dtype}")
        print(f"   Mask dtype: {augmented['mask'].dtype}")
        print(f"   ✅ Training augmentation OK!")

        # Test validation augmentation
        print("\n📊 Testing Validation Augmentation...")
        val_aug = get_validation_augmentation_improved()
        val_augmented = val_aug(image=dummy_image, mask=dummy_mask)
        print(f"   Output image shape: {val_augmented['image'].shape}")
        print(f"   Output mask shape: {val_augmented['mask'].shape}")
        print(f"   ✅ Validation augmentation OK!")

        # Test moderate augmentation
        print("\n📊 Testing Moderate Augmentation...")
        mod_aug = get_moderate_augmentation()
        mod_augmented = mod_aug(image=dummy_image, mask=dummy_mask)
        print(f"   ✅ Moderate augmentation OK!")

        # Test light augmentation
        print("\n📊 Testing Light Augmentation...")
        light_aug = get_light_augmentation()
        light_augmented = light_aug(image=dummy_image, mask=dummy_mask)
        print(f"   ✅ Light augmentation OK!")

        # Test TTA
        print("\n📊 Testing TTA...")
        tta_transforms = get_test_time_augmentation()
        print(f"   TTA transforms: {len(tta_transforms)}")
        print(f"   ✅ TTA OK!")

        print("\n" + "=" * 70)
        print("✅ TUTTI I TEST COMPLETATI CON SUCCESSO!")
        print("=" * 70)
        print("\n📝 COME USARE:")
        print("1. Per training: get_training_augmentation_improved()")
        print("2. Se overfitting: get_moderate_augmentation()")
        print("3. Se dataset piccolo: get_light_augmentation()")
        print("4. Per validation: get_validation_augmentation_improved()")
        print("5. Per TTA inference: get_test_time_augmentation()")
        print("\n✅ File pronto per essere usato in train.py!")

    except Exception as e:
        print(f"\n❌ ERRORE: {e}")
        import traceback

        traceback.print_exc()
        print("\nVerifica che albumentations sia aggiornato:")
        print("pip install -U albumentations")

