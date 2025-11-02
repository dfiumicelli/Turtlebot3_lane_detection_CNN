# python
import torch
import segmentation_models_pytorch as smp

# Percorsi (esempio)
weights_path = 'C:\\Users\\mfiumicelli\\PycharmProjects\\Turtlebot3_Perception\\test08_nn\\checkpoints_and_models_v2\\last_unet_finetuned.pth'          # file contenente solo state_dict
full_ckpt_path = 'full_checkpoint_v3.pth'    # dove salvare checkpoint completo
full_model_path = 'full_model_v3.pth'        # opzionale: salva l'intero oggetto modello

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

# Salva checkpoint completo con metadata (raccomandato)
torch.save({
    'encoder': encoder,
    'encoder_weights': encoder_weights,
    'classes': classes,
    'activation': activation,
    'model_state_dict': model.state_dict(),
}, full_ckpt_path)
print(f"Checkpoint salvato in: {full_ckpt_path}")

# Opzionale: salva l'intero oggetto modello (meno portabile)
torch.save(model, full_model_path)
print(f"Intero modello salvato in: {full_model_path}")
