import onnx
from pathlib import Path
import shutil


# ============================================================================
# CONSOLIDARE FILE ONNX
# ============================================================================

def consolidate_onnx(onnx_path, output_path=None):
    """
    Consolida i file ONNX in uno solo.
    Se hai:
      - model.onnx (piccolo)
      - model.onnx.data (grande, con i pesi)

    Crea:
      - model_consolidated.onnx (completo)
    """

    if output_path is None:
        output_path = onnx_path.replace('.onnx', '_consolidated.onnx')

    print("=" * 70)
    print("CONSOLIDAMENTO FILE ONNX")
    print("=" * 70 + "\n")

    onnx_path = Path(onnx_path)
    data_path = Path(str(onnx_path) + '.data')

    print(f"Input ONNX: {onnx_path}")
    print(f"Input DATA: {data_path}")
    print(f"Output: {output_path}\n")

    # Verifica che i file esistano
    if not onnx_path.exists():
        print(f"✗ File non trovato: {onnx_path}")
        return False

    if not data_path.exists():
        print(f"⚠ File data non trovato: {data_path}")
        print(f"   Potrebbe non esserci bisogno di consolidamento")
        print(f"   Copio il file .onnx così com'è\n")
        shutil.copy(onnx_path, output_path)
        print(f"✓ Copiato in: {output_path}\n")
        return True

    # Mostra dimensioni
    onnx_size = onnx_path.stat().st_size / (1024 ** 2)
    data_size = data_path.stat().st_size / (1024 ** 2)
    total_size = onnx_size + data_size

    print(f"Dimensioni attuali:")
    print(f"  {onnx_path.name}: {onnx_size:.1f} MB")
    print(f"  {data_path.name}: {data_size:.1f} MB")
    print(f"  Totale: {total_size:.1f} MB\n")

    # Consolida i file
    print(f"Consolidamento in corso...\n")

    try:
        # Carica il modello ONNX
        model = onnx.load(str(onnx_path))

        # Salva senza il file esterno (.data)
        # Questo forza onnx a includere i pesi nel file principale
        onnx.save(
            model,
            output_path,
            save_as_external_data=False  # ← Chiave: no file esterno
        )

        output_size = Path(output_path).stat().st_size / (1024 ** 2)

        print(f"✓ Consolidamento completato!\n")
        print(f"File consolidato: {output_path}")
        print(f"Dimensione: {output_size:.1f} MB\n")

        # Mostra il risparmio
        saved = total_size - output_size
        if saved > 0:
            print(f"Risparmio spazio: {saved:.1f} MB")

        return True

    except Exception as e:
        print(f"✗ Errore consolidamento: {e}")
        return False


# ============================================================================
# TEST MODELLO CONSOLIDATO
# ============================================================================

def test_consolidated_onnx(onnx_path):
    """Testa il modello ONNX consolidato"""
    import onnxruntime as ort
    import numpy as np
    import time

    print(f"\n[TEST] Caricamento modello consolidato...\n")

    try:
        # Carica il modello
        session = ort.InferenceSession(onnx_path)

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
        print(f"\nEsecuzione test inference...")
        test_input = np.random.randn(1, 3, 240, 320).astype(np.float32)

        start = time.perf_counter()
        output = session.run(None, {'input': test_input})
        elapsed = time.perf_counter() - start

        # Gestisci output come list o array
        if isinstance(output, list):
            output_array = output
        else:
            output_array = output

        print(f"  Input shape: {test_input.shape}")
        print(f"  Output shape: {output_array.shape}")
        print(f"  Tempo: {elapsed * 1000:.2f} ms")
        print(f"  FPS: {1 / elapsed:.2f}")

        print(f"\n✓ Test completato - Modello funziona!\n")
        return True

    except Exception as e:
        print(f"\n✗ Errore test: {type(e).__name__}")
        print(f"  {e}\n")
        return False


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Consolida il file ONNX
    success = consolidate_onnx('unet_mobilenet_int8_working.onnx')

    if success:
        # Testa il modello consolidato
        test_consolidated_onnx('unet_mobilenet_consolidated_int8.onnx')

        print("=" * 70)
        print("✓ MODELLO ONNX PRONTO PER IL TURTLEBOT")
        print("=" * 70)
        print(f"\nUsare nel nodo ROS2: full_model_v3.onnx\n")
        print(f"Codice ROS2:\n")
        print("""
import onnxruntime as ort
import numpy as np

sess_options = ort.SessionOptions()
sess_options.intra_op_num_threads = 4
session = ort.InferenceSession('full_model_v3.onnx', sess_options)

# Inference
output = session.run(None, {'input': image_tensor})
        """)
        print("=" * 70 + "\n")
