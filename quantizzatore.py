import torch
import torch.nn as nn
import onnxruntime as ort
import numpy as np
from pathlib import Path
import segmentation_models_pytorch as smp
import time

# ============================================================================
# CONFIGURAZIONE
# ============================================================================

MODEL_PATH = "test08_nn/checkpoints_and_models_v2/last_unet_finetuned.pth"
OUTPUT_ONNX = "unet_mobilenet.onnx"

INPUT_HEIGHT = 240
INPUT_WIDTH = 320
INPUT_CHANNELS = 3

# ============================================================================
# PASSO 1: CARICA MODELLO PYTORCH
# ============================================================================

print("=" * 70)
print("ESPORTAZIONE ONNX - SOLUZIONE DEFINITIVA (con torch.jit.trace)")
print("=" * 70 + "\n")

print(f"[STEP 1] Caricamento modello PyTorch...")
print(f"File: {MODEL_PATH}\n")

# Carica il modello
loaded = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)

if isinstance(loaded, torch.nn.Module):
    model = loaded
    print("✓ Full model caricato\n")
else:
    # Istanzia e carica pesi
    model = smp.Unet(
        encoder_name='mobilenet_v2',
        encoder_weights=None,
        classes=1,
        activation=None,
        decoder_dropout=0.2
    )

    if isinstance(loaded, dict) and 'model_state_dict' in loaded:
        state = loaded['model_state_dict']
    else:
        state = loaded

    model.load_state_dict(state, strict=False)
    print("✓ State dict caricato\n")

model.eval()

# Verifica pesi
param_count = sum(p.numel() for p in model.parameters())
param_size_mb = param_count * 4 / (1024 ** 2)
print(f"Parametri modello: {param_count:,}")
print(f"Dimensione (float32): ~{param_size_mb:.1f} MB\n")

# ============================================================================
# PASSO 2: ESPORTA A ONNX usando torch.jit.trace (METODO AFFIDABILE)
# ============================================================================

print(f"[STEP 2] Esportazione a ONNX usando torch.jit.trace...\n")

# Crea dummy input per il trace
dummy_input = torch.randn(1, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH)

try:
    # Metodo 1: torch.jit.trace (più affidabile per segmentation_models_pytorch)
    print("  Metodo: torch.jit.trace (più affidabile)")

    # Testa prima che il modello funziona
    with torch.no_grad():
        test_output = model(dummy_input)
    print(f"  Output test: {test_output.shape}")

    # Esegui il trace
    traced_model = torch.jit.trace(model, dummy_input)

    # Esporta da traced model
    torch.onnx.export(
        traced_model,
        dummy_input,
        OUTPUT_ONNX,
        input_names=['input'],
        output_names=['output'],
        opset_version=18,
        verbose=False,
    )

    print(f"✓ Esportazione completata\n")

except Exception as e:
    print(f"  ⚠ Errore con jit.trace: {e}")
    print(f"  Provo metodo alternativo...\n")

    try:
        # Metodo 2: export diretto con opzioni minimali
        print("  Metodo: export diretto con do_constant_folding=False")

        torch.onnx.export(
            model,
            dummy_input,
            OUTPUT_ONNX,
            input_names=['input'],
            output_names=['output'],
            opset_version=18,
            do_constant_folding=False,  # ← CRITICO
            export_params=True,  # ← Esporta pesi
            verbose=False,
        )

        print(f"✓ Esportazione con metodo alternativo completata\n")

    except Exception as e2:
        print(f"✗ Errore: {e2}")
        exit(1)

# ============================================================================
# PASSO 3: VERIFICA FILE ONNX
# ============================================================================

print(f"[STEP 3] Verifica file ONNX...\n")

if not Path(OUTPUT_ONNX).exists():
    print(f"✗ ERRORE: File non creato!")
    exit(1)

file_size_mb = Path(OUTPUT_ONNX).stat().st_size / (1024 ** 2)
print(f"File creato: {OUTPUT_ONNX}")
print(f"Dimensione: {file_size_mb:.1f} MB")

if file_size_mb < 1:
    print(f"\n⚠ WARNING: File troppo piccolo ({file_size_mb:.1f} MB)")
    print(f"  Atteso: > 20 MB")
    print(f"  I pesi NON sono stati salvati!")
    print(f"\n  SOLUZIONE: Salva il modello PyTorch in modo diverso")
    print(f"  Nel tuo train.py, usa:")
    print(f"""
    # ✓ Modo corretto per salvare:
    torch.save(model, 'full_model.pth')  # Salva tutto (modello + pesi)

    # ✗ Non usare:
    torch.save(model.state_dict(), 'model.pth')  # Solo pesi, senza modello
    """)
    exit(1)
else:
    print(f"✓ Dimensione corretta!\n")

# ============================================================================
# PASSO 4: CARICA E TESTA ONNX
# ============================================================================

print(f"[STEP 4] Test del modello ONNX...\n")

try:
    # Carica il modello ONNX
    session = ort.InferenceSession(OUTPUT_ONNX)

    # Mostra info
    inputs = session.get_inputs()
    outputs = session.get_outputs()

    print(f"Inputs:")
    for inp in inputs:
        print(f"  - {inp.name}: {inp.shape}")

    print(f"\nOutputs:")
    for out in outputs:
        print(f"  - {out.name}: {out.shape}")

    # Test inference
    print(f"\nTest inference...")
    test_input = np.random.randn(1, 3, 240, 320).astype(np.float32)

    start = time.time()
    output = session.run(None, {'input': test_input})
    elapsed = time.time() - start

    print(f"  Input shape: {test_input.shape}")

    # Gestisci sia list che numpy array
    if isinstance(output, list):
        output_array = output
    else:
        output_array = output

    print(f"  Output shape: {output_array.shape}")
    print(f"  Tempo: {elapsed * 1000:.2f} ms")

    fps = 1 / elapsed
    print(f"  FPS: {fps:.2f}")

    print(f"\n✓ Test completato - Modello funziona!\n")

    test_success = True

except Exception as e:
    print(f"\n✗ Errore nel test: {type(e).__name__}")
    print(f"  {e}\n")
    test_success = False

# ============================================================================
# PASSO 5: BENCHMARK
# ============================================================================

if test_success:
    print(f"[STEP 5] Benchmark performance...\n")

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.intra_op_num_threads = 4

    session = ort.InferenceSession(OUTPUT_ONNX, sess_options)

    times = []
    num_runs = 100

    print(f"Esecuzione {num_runs} inference...")

    for i in range(num_runs):
        test_input = np.random.randn(1, 3, 240, 320).astype(np.float32)
        start = time.perf_counter()
        session.run(None, {'input': test_input})
        times.append(time.perf_counter() - start)

        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{num_runs}")

    times = np.array(times)
    mean_time = np.mean(times)
    fps_mean = 1 / mean_time

    print(f"\nRisultati benchmark:")
    print(f"  Tempo medio: {mean_time * 1000:.2f} ms")
    print(f"  FPS: {fps_mean:.2f}")
    print(f"  Expected su Latitude 5300 (i5-8265U): 15-30 FPS\n")

    # ============================================================================
    # RISULTATI FINALI
    # ============================================================================

    print("=" * 70)
    print("✓ ESPORTAZIONE COMPLETATA CON SUCCESSO")
    print("=" * 70)
    print(f"\nFile ONNX pronto: {OUTPUT_ONNX}")
    print(f"Dimensione: {file_size_mb:.1f} MB")
    print(f"FPS atteso: {fps_mean:.1f}\n")

    print("Codice per il nodo ROS2 Humble:\n")

    print("""
import onnxruntime as ort
import numpy as np
import cv2

# Carica il modello
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess_options.intra_op_num_threads = 4

session = ort.InferenceSession('unet_mobilenet.onnx', sess_options)

# Preprocessing
image = cv2.imread('image.jpg')
image = cv2.resize(image, (320, 240))
image = image.astype(np.float32) / 255.0

# Normalize (ImageNet)
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
image = (image - mean) / std

image = np.transpose(image, (2, 0, 1))  # HWC -> CHW
image = np.expand_dims(image, 0)  # Add batch

# Inference
output = session.run(None, {'input': image.astype(np.float32)})

# Estrai la maschera
if isinstance(output, list):
    mask = output
else:
    mask = output

segmentation_mask = mask[0, 0, :, :]  # shape (240, 320)
binary_mask = (segmentation_mask > 0.5).astype(np.uint8)
    """)

    print("=" * 70 + "\n")
else:
    print(f"⚠ Test fallito - modello potrebbe non essere stato salvato correttamente")
    print(f"\n💡 Suggerimento: Nel tuo train.py, salva il modello così:")
    print("""
# ✓ Modo CORRETTO:
torch.save(model, 'full_model.pth')

# Nel nodo ROS2, carica così:
model = torch.load('full_model.pth', map_location='cpu')
model.eval()
    """)
