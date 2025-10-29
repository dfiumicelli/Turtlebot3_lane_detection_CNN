# python

import torch
import os

print("\n" + "=" * 80)
print("🔧 CARICAMENTO MODELLO TORCHSCRIPT (GENERATO SU WINDOWS)")
print("=" * 80)

# Percorso al file TorchScript generato su Windows
torchscript_path = 'model_traced.pt'  # File generato da Windows

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"\n📦 Caricamento modello: {torchscript_path}")
print(f"   Device: {device}")

# ============================================================================
# CARICAMENTO DIRETTO DEL FILE TORCHSCRIPT
# ============================================================================

try:
    # Carica il file TorchScript (NON ha dipendenze su segmentation_models_pytorch)
    model = torch.jit.load(torchscript_path, map_location=device)
    model.eval()

    print(f"✓ Modello TorchScript caricato con successo!")

    # Test: prova un forward pass
    print(f"\n🧪 Test forward pass...")
    dummy_input = torch.randn(1, 3, 240, 320).to(device)

    with torch.no_grad():
        output = model(dummy_input)

    print(f"✓ Forward pass riuscito!")
    print(f"   Input shape:  {dummy_input.shape}")
    print(f"   Output shape: {output.shape}")

except RuntimeError as e:
    if "PytorchStreamReader" in str(e) or "central directory" in str(e):
        print(f"\n❌ ERRORE: Il file '{torchscript_path}' è corrotto o non valido")
        print(f"\n📋 SOLUZIONE:")
        print(f"   1. Su Windows (Python 3.12), esegui:")
        print(f"      python model_generator_torchscript.py")
        print(f"   2. Genera di nuovo il file 'model_traced.pt'")
        print(f"   3. Copia il file su Ubuntu")
        print(f"   4. Prova di nuovo questo script")
        raise
    else:
        raise

except FileNotFoundError:
    print(f"\n❌ ERRORE: File non trovato: {torchscript_path}")
    print(f"\n📋 SOLUZIONE:")
    print(f"   1. Assicurati di avere il file 'model_traced.pt' in questa directory")
    print(f"   2. Se non lo hai, generalo su Windows con: python model_generator_torchscript.py")
    print(f"   3. Copia il file qui e prova di nuovo")
    raise

except Exception as e:
    print(f"\n❌ ERRORE INASPETTATO: {e}")
    print(f"\n📋 Prova a:")
    print(f"   1. Rigenerare il file su Windows")
    print(f"   2. Verificare che il file non sia corrotto")
    print(f"   3. Trasferirlo di nuovo a Ubuntu")
    raise

print("\n" + "=" * 80)
print("✅ MODELLO PRONTO PER L'USO!")
print("=" * 80)
print(f"\n💾 Salvataggio modello su variabile globale...")

# Salva il modello globalmente per accesso successivo
LOADED_MODEL = model
DEVICE = device

print(f"✓ Puoi usare: torch.jit.load('{torchscript_path}', map_location='cpu')")
print("=" * 80 + "\n")
