# augmentation.py
# Script per data augmentation e preprocessing

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_training_augmentation():
    """
    Data augmentation per training set.
    Include trasformazioni geometriche e di colore per aumentare la robustezza.
    """
    train_transform = [
        # Resize a dimensioni fisse
        A.Resize(256, 256),
        
        # Trasformazioni geometriche
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.0625, 
            scale_limit=0.1, 
            rotate_limit=10, 
            p=0.5
        ),
        
        # Trasformazioni di colore (simula variazioni di illuminazione)
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
        
        # Blur e rumore (simula condizioni difficili)
        A.OneOf([
            A.GaussianBlur(blur_limit=3, p=1.0),
            A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
        ], p=0.3),
        
        # Dropout casuale (simula occlusioni)
        A.CoarseDropout(
            max_holes=8,
            max_height=32,
            max_width=32,
            fill_value=0,
            p=0.2
        ),
    ]
    return A.Compose(train_transform)


def get_validation_augmentation():
    """
    Augmentation per validation set.
    Solo resize, nessuna randomizzazione.
    """
    val_transform = [
        A.Resize(256, 256),
    ]
    return A.Compose(val_transform)


def get_preprocessing(preprocessing_fn=None):
    """
    Preprocessing specifico dell'encoder.
    
    Args:
        preprocessing_fn: funzione di preprocessing da segmentation_models_pytorch
    
    Returns:
        Compose di trasformazioni Albumentations
    """
    _transform = []
    
    if preprocessing_fn:
        _transform.append(A.Lambda(image=preprocessing_fn))
    
    # Converti a PyTorch tensor
    _transform.append(ToTensorV2())
    
    return A.Compose(_transform)


def get_heavy_augmentation():
    """
    Augmentation più aggressiva per dataset piccoli (< 500 immagini).
    Aumenta la variabilità per evitare overfitting.
    """
    heavy_transform = [
        A.Resize(256, 256),
        
        # Trasformazioni geometriche più aggressive
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.7),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.2,
            rotate_limit=15,
            p=0.7
        ),
        
        # Variazioni di colore e illuminazione
        A.RandomBrightnessContrast(
            brightness_limit=0.3,
            contrast_limit=0.3,
            p=0.6
        ),
        A.HueSaturationValue(
            hue_shift_limit=15,
            sat_shift_limit=30,
            val_shift_limit=30,
            p=0.5
        ),
        
        # Effetti atmosferici
        A.OneOf([
            A.GaussianBlur(blur_limit=5, p=1.0),
            A.MotionBlur(blur_limit=5, p=1.0),
            A.GaussNoise(var_limit=(10.0, 80.0), p=1.0),
        ], p=0.4),
        
        # Dropout e occlusioni
        A.CoarseDropout(
            max_holes=12,
            max_height=48,
            max_width=48,
            fill_value=0,
            p=0.3
        ),
        
        # Compressione JPEG (simula immagini di qualità ridotta)
        A.ImageCompression(quality_lower=80, quality_upper=100, p=0.2),
    ]
    return A.Compose(heavy_transform)
