# metrics_CORRECTED.py - METRICHE CORRETTE PER LANE DETECTION

import torch
import numpy as np

class LaneMetrics:
    """Metriche specializzate per lane detection - VERSIONE CORRETTA"""
    
    # ==================== 1️⃣ DICE COEFFICIENT ====================
    @staticmethod
    def dice_coefficient(pred, target, smooth=1.0):
        """
        ✅ CORRETTO
        Formula: Dice = (2 * TP) / (2 * TP + FP + FN)
        """
        # Converti a binario (pred potrebbe già essere binario o probabilità)
        if pred.max() > 1.0:
            pred_binary = (pred > 0.5).float()
        else:
            pred_binary = pred
        
        target = target.float()
        
        intersection = (pred_binary * target).sum()
        dice = (2.0 * intersection + smooth) / (pred_binary.sum() + target.sum() + smooth)
        return dice.item()
    
    
    # ==================== 2️⃣ SENSITIVITY (RECALL) ====================
    @staticmethod
    def sensitivity(pred, target, smooth=1e-6):
        """
        ✅ CORRETTO
        Formula: Sensitivity = TP / (TP + FN)
        
        ⚠️ BUG ORIGINALE: Non convertiva pred a binario!
        """
        # ✅ FIX: Converti a binario
        if pred.max() > 1.0:
            pred_binary = (pred > 0.5).float()
        else:
            pred_binary = pred
        
        target = target.float()
        
        TP = (pred_binary * target).sum()
        FN = ((1 - pred_binary) * target).sum()
        
        sensitivity = TP / (TP + FN + smooth)
        return sensitivity.item()
    
    
    # ==================== 3️⃣ SPECIFICITY ====================
    @staticmethod
    def specificity(pred, target, smooth=1e-6):
        """
        ✅ CORRETTO
        Formula: Specificity = TN / (TN + FP)
        
        ⚠️ BUG ORIGINALE: Non convertiva pred a binario!
        """
        # ✅ FIX: Converti a binario
        if pred.max() > 1.0:
            pred_binary = (pred > 0.5).float()
        else:
            pred_binary = pred
        
        target = target.float()
        
        TN = ((1 - pred_binary) * (1 - target)).sum()
        FP = (pred_binary * (1 - target)).sum()
        
        specificity = TN / (TN + FP + smooth)
        return specificity.item()
    
    
    # ==================== 4️⃣ F1 SCORE ====================
    @staticmethod
    def f1_score(pred, target, smooth=1e-6):
        """
        ✅ CORRETTO
        Formula: F1 = 2 * (Precision * Recall) / (Precision + Recall)
        """
        # ✅ FIX: Converti a binario
        if pred.max() > 1.0:
            pred_binary = (pred > 0.5).float()
        else:
            pred_binary = pred
        
        target = target.float()
        
        TP = (pred_binary * target).sum()
        FP = (pred_binary * (1 - target)).sum()
        FN = ((1 - pred_binary) * target).sum()
        
        precision = TP / (TP + FP + smooth)
        recall = TP / (TP + FN + smooth)
        f1 = 2 * (precision * recall) / (precision + recall + smooth)
        
        return f1.item()
    
    
    # ==================== 5️⃣ MATTHEWS CORRELATION COEFFICIENT ====================
    @staticmethod
    def mcc(pred, target, smooth=1e-6):
        """
        ✅ CORRETTO
        Formula: MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
        
        ⚠️ BUG ORIGINALE: Non gestiva correttamente i tensori
        """
        # ✅ FIX: Converti a binario
        if pred.max() > 1.0:
            pred_binary = (pred > 0.5).float()
        else:
            pred_binary = pred
        
        target = target.float()
        
        TP = (pred_binary * target).sum()
        TN = ((1 - pred_binary) * (1 - target)).sum()
        FP = (pred_binary * (1 - target)).sum()
        FN = ((1 - pred_binary) * target).sum()
        
        numerator = TP * TN - FP * FN
        denominator = torch.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN) + smooth)
        
        # ✅ FIX: Gestisci il caso di denominatore zero
        if denominator == 0:
            return 0.0
        
        mcc = numerator / denominator
        return mcc.item()
    
    
    # ==================== 6️⃣ PIXEL ACCURACY ====================
    @staticmethod
    def pixel_accuracy(pred, target, smooth=1e-6):
        """
        ✅ CORRETTO
        Formula: Accuracy = (TP + TN) / Total
        """
        # ✅ FIX: Converti a binario
        if pred.max() > 1.0:
            pred_binary = (pred > 0.5).float()
        else:
            pred_binary = pred
        
        target = target.float()
        
        correct = (pred_binary == target).float().sum()
        total = target.numel()
        
        # ✅ FIX: Gestisci il caso di total zero
        if total == 0:
            return 0.0
        
        return (correct / total).item()
    
    
    # ==================== 7️⃣ IoU (Intersection over Union) ====================
    @staticmethod
    def iou(pred, target, smooth=1e-6):
        """
        ✅ AGGIUNTO - Lo standard per segmentazione
        Formula: IoU = TP / (TP + FP + FN)
        """
        # ✅ FIX: Converti a binario
        if pred.max() > 1.0:
            pred_binary = (pred > 0.5).float()
        else:
            pred_binary = pred
        
        target = target.float()
        
        intersection = (pred_binary * target).sum()
        union = pred_binary.sum() + target.sum() - intersection
        
        iou_score = (intersection + smooth) / (union + smooth)
        return iou_score.item()


# ==================== FUNZIONE DI UTILITÀ ====================

def calculate_all_metrics(pred_batch, target_batch):
    """
    ✅ CORRETTA - Calcola tutte le metriche per un batch
    
    Input:
        pred_batch: tensor [B, H, W] con probabilità [0, 1] o valori > 1
        target_batch: tensor [B, H, W] con valori binari {0, 1}
    
    Output:
        dict con tutte le metriche
    """
    metrics = {
        'iou': [],
        'dice': [],
        'sensitivity': [],
        'specificity': [],
        'f1': [],
        'mcc': [],
        'accuracy': [],
    }
    
    # ✅ FIX: Itera correttamente su ogni elemento del batch
    for batch_idx in range(pred_batch.shape[0]):
        pred_single = pred_batch[batch_idx]    # [H, W]
        target_single = target_batch[batch_idx] # [H, W]
        
        # Calcola tutte le metriche per questa immagine
        metrics['iou'].append(LaneMetrics.iou(pred_single, target_single))
        metrics['dice'].append(LaneMetrics.dice_coefficient(pred_single, target_single))
        metrics['sensitivity'].append(LaneMetrics.sensitivity(pred_single, target_single))
        metrics['specificity'].append(LaneMetrics.specificity(pred_single, target_single))
        metrics['f1'].append(LaneMetrics.f1_score(pred_single, target_single))
        metrics['mcc'].append(LaneMetrics.mcc(pred_single, target_single))
        metrics['accuracy'].append(LaneMetrics.pixel_accuracy(pred_single, target_single))
    
    # ✅ Restituisci media di tutte le metriche
    return {k: np.mean(v) if v else 0.0 for k, v in metrics.items()}


# ==================== DEBUG ====================

if __name__ == '__main__':
    print("✅ Test metriche corrette...\n")
    
    # Crea dati fittizi
    pred = torch.rand(256, 256)           # Probabilità [0, 1]
    target = torch.randint(0, 2, (256, 256)).float()  # Binario
    
    print("📊 Metriche per Lane Detection:\n")
    print(f"  IoU:              {LaneMetrics.iou(pred, target):.4f}")
    print(f"  Dice:             {LaneMetrics.dice_coefficient(pred, target):.4f}")
    print(f"  Sensitivity:      {LaneMetrics.sensitivity(pred, target):.4f} ← Corsie trovate")
    print(f"  Specificity:      {LaneMetrics.specificity(pred, target):.4f} ← Falsi positivi")
    print(f"  F1 Score:         {LaneMetrics.f1_score(pred, target):.4f}")
    print(f"  MCC:              {LaneMetrics.mcc(pred, target):.4f}")
    print(f"  Accuracy:         {LaneMetrics.pixel_accuracy(pred, target):.4f}")
    
    # Test batch
    print("\n\n📊 Test Batch:\n")
    pred_batch = torch.rand(4, 256, 256)
    target_batch = torch.randint(0, 2, (4, 256, 256)).float()
    
    all_metrics = calculate_all_metrics(pred_batch, target_batch)
    for k, v in all_metrics.items():
        print(f"  {k.upper():12s}: {v:.4f}")
