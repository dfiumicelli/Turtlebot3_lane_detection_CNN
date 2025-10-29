# python

import torch
import segmentation_models_pytorch as smp

# Percorsi (esempio)
weights_path = 'best_unet_mobilenet1.pth'  # file contenente solo state_dict
full_ckpt_path = 'full_checkpoint.pth'  # dove salvare checkpoint completo
full_model_path = 'full_model.pth'  # opzionale: salva l'intero oggetto modello

# TorchScript paths
torchscript_path = 'model_traced.pt'  # modello tracciato (cross-Python compatible)
torchscript_scripted_path = 'model_scripted.pt'  # modello scriptato

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Parametri esatti usati in training (assicurati coincidano)
encoder = 'mobilenet_v2'
encoder_weights = None
classes = 1
activation = None

# Se hai usato parametri addizionali (es. decoder_dropout), aggiungili qui.

# Istanzia il modello con gli stessi argomenti
model = smp.Unet(
    encoder_name=encoder,
    encoder_weights=encoder_weights,  # o None se checkpoint contiene tutti i pesi
    classes=classes,
    activation=activation,
    # aggiungi altri argomenti usati in training se necessari
)

model.to(device)

# Carica i pesi (gestisce sia state_dict diretto che dict con chiave model_state_dict)
loaded = torch.load(weights_path, map_location=device)
state = loaded.get('model_state_dict', loaded) if isinstance(loaded, dict) else loaded

# Prova a caricare con strict=True, altrimenti fallback a strict=False e stampa differenze
try:
    res = model.load_state_dict(state, strict=True)
    print("Caricato con strict=True. Missing:", res.missing_keys, "Unexpected:", res.unexpected_keys)
except Exception as e:
    print("strict=True fallito:", e)
    res = model.load_state_dict(state, strict=False)
    print("Caricato con strict=False. Missing:", res.missing_keys, "Unexpected:", res.unexpected_keys)

model.eval()

# ============================================================================
# CONVERSIONE A TORCHSCRIPT - METODO 1: torch.jit.trace (CONSIGLIATO)
# ============================================================================
print("\n[1/3] Conversione a TorchScript (torch.jit.trace)...")

# Crea un input dummy con la forma corretta (batch_size, channels, height, width)
# Adatta le dimensioni in base al tuo modello
dummy_input = torch.randn(1, 3, 512, 512).to(device)

try:
    # Traccia il modello
    traced_model = torch.jit.trace(model, dummy_input)

    # Salva il modello tracciato
    torch.jit.save(traced_model, torchscript_path)
    print(f"✓ Modello tracciato salvato in: {torchscript_path}")

    # Test: carica e verifica
    loaded_traced = torch.jit.load(torchscript_path, map_location=device)
    print("✓ Verifica: modello tracciato caricato correttamente")

except Exception as e:
    print(f"✗ Errore nella traccia: {e}")

# ============================================================================
# CONVERSIONE A TORCHSCRIPT - METODO 2: torch.jit.script (alternativo)
# ============================================================================
print("\n[2/3] Conversione a TorchScript (torch.jit.script)...")

try:
    # Script il modello (richiede codice scripted)
    scripted_model = torch.jit.script(model)
    torch.jit.save(scripted_model, torchscript_scripted_path)
    print(f"✓ Modello scriptato salvato in: {torchscript_scripted_path}")
except Exception as e:
    print(f"✗ Script fallito (normale per modelli complessi): {e}")
    print("   Consigliato usare il metodo tracciato (trace)")

# ============================================================================
# SALVATAGGIO TRADIZIONALE
# ============================================================================
print("\n[3/3] Salvataggio checkpoint tradizionale...")

# Salva checkpoint completo con metadata (raccomandato)
torch.save({
    'encoder': encoder,
    'encoder_weights': encoder_weights,
    'classes': classes,
    'activation': activation,
    'model_state_dict': model.state_dict(),
}, full_ckpt_path)

print(f"✓ Checkpoint salvato in: {full_ckpt_path}")

# Opzionale: salva l'intero oggetto modello (meno portabile)
torch.save(model, full_model_path)
print(f"✓ Intero modello salvato in: {full_model_path}")

# ============================================================================
# RIEPILOGO FORMATI
# ============================================================================
print("\n" + "=" * 70)
print("RIEPILOGO CONVERSIONE")
print("=" * 70)
print(f"✓ TorchScript (Traced):  {torchscript_path}")
print(f"  → Usa su UBUNTU (Python 3.10): torch.jit.load('{torchscript_path}')")
print(f"\n✓ Checkpoint:            {full_ckpt_path}")
print(f"  → State dict tradizionale con metadata")
print(f"\n✓ Full Model:            {full_model_path}")
print(f"  → Intero oggetto modello")
print("=" * 70)

# ============================================================================
# ESEMPIO DI UTILIZZO SU UBUNTU
# ============================================================================
print("\nPer usare il modello su Ubuntu (Python 3.10):")
print("""
import torch

# Carica il modello TorchScript
model = torch.jit.load('model_traced.pt', map_location='cpu')
model.eval()

# Usa il modello
with torch.no_grad():
    output = model(input_tensor)
""")
