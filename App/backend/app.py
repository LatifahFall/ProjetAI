import os
import torch
import torch.nn as nn
import pickle
import whisper
from pathlib import Path
from flask import Flask
from flask_cors import CORS
from transformers import AutoModel, BertTokenizer, AutoTokenizer

app = Flask(__name__)
CORS(app)

# --- Configuration des chemins ---
# Chemin relatif vers le dossier models à la racine du projet
BASE_DIR = Path(__file__).parent.parent.parent  # Remonte jusqu'à la racine du projet
BASE_MODEL_PATH = str(BASE_DIR / 'models')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ══════════════════════════════════════════════════════════════════════════════
# ARCHITECTURES DES MODÈLES (DOIVENT ÊTRE DÉFINIES AVANT LE CHARGEMENT DES POIDS)
# ══════════════════════════════════════════════════════════════════════════════

class WavLMRegressionHead(nn.Module):
    def __init__(self, model_name="microsoft/wavlm-base-plus", n_traits=5):
        super().__init__()
        self.wavlm = AutoModel.from_pretrained(model_name)
        hidden_size = self.wavlm.config.hidden_size
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size * 2, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, n_traits)
        )
    
    def forward(self, waveform):
        outputs = self.wavlm(waveform)
        hidden = outputs.last_hidden_state
        mean_pool = hidden.mean(dim=1)
        std_pool = hidden.std(dim=1)
        pooled = torch.cat([mean_pool, std_pool], dim=1)
        return pooled  # Retourne les features pour le scaler

class BERTRegressionHead(nn.Module):
    def __init__(self, model_name="bert-base-uncased", n_traits=5):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, n_traits)
        )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        token_emb = outputs.last_hidden_state
        mask_expanded = attention_mask.unsqueeze(-1).expand(token_emb.size()).float()
        sum_emb = torch.sum(token_emb * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        pooled = sum_emb / sum_mask
        return pooled  # Retourne les features pour le scaler

# ══════════════════════════════════════════════════════════════════════════════
# CHARGEMENT DES COMPOSANTS
# ══════════════════════════════════════════════════════════════════════════════

print("[*] Chargement des modeles et scalers en memoire...")

# Initialiser les modèles
app.scalers = None
app.whisper_model = None
app.bert_tokenizer = None
app.bert_model = None
app.wavlm_model = None
app.models_loaded = False

# Charger les modèles uniquement si les fichiers existent
scalers_path = os.path.join(BASE_MODEL_PATH, 'scalers_v7.pkl')
bert_weights = os.path.join(BASE_MODEL_PATH, 'bert_finetuned_v5.pt')
wavlm_weights = os.path.join(BASE_MODEL_PATH, 'wavlm_finetuned_v5.pt')

if os.path.exists(scalers_path) and os.path.exists(bert_weights) and os.path.exists(wavlm_weights):
    try:
        # 1. Scalers
        with open(scalers_path, 'rb') as f:
            app.scalers = pickle.load(f)
        print("[OK] Scalers charges.")

        # 2. Whisper
        app.whisper_model = whisper.load_model("base").to(app.device)
        print("[OK] Whisper charge.")

        # 3. BERT + Poids fine-tunes
        app.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        app.bert_model = BERTRegressionHead().to(app.device)
        app.bert_model.load_state_dict(torch.load(bert_weights, map_location=app.device))
        app.bert_model.eval()
        print("[OK] BERT fine-tune charge.")

        # 4. WavLM + Poids fine-tunes
        app.wavlm_model = WavLMRegressionHead().to(app.device)
        app.wavlm_model.load_state_dict(torch.load(wavlm_weights, map_location=app.device))
        app.wavlm_model.eval()
        print("[OK] WavLM fine-tune charge.")

        app.models_loaded = True
        print("[OK] Tous les modeles sont charges - route /predict disponible")
    except Exception as e:
        print(f"[ERROR] Erreur lors du chargement des modeles: {str(e)}")
        print("[INFO] Route /predict non disponible")
else:
    missing = []
    if not os.path.exists(scalers_path):
        missing.append("scalers_v7.pkl")
    if not os.path.exists(bert_weights):
        missing.append("bert_finetuned_v5.pt")
    if not os.path.exists(wavlm_weights):
        missing.append("wavlm_finetuned_v5.pt")
    print(f"[INFO] Fichiers de modeles manquants: {', '.join(missing)}")
    print("[INFO] Route /predict non disponible - routes /clients fonctionnent normalement")

# ══════════════════════════════════════════════════════════════════════════════
# LANCEMENT
# ══════════════════════════════════════════════════════════════════════════════

# Import de la route predict (nécessite les modèles et dépendances ML)
if app.models_loaded:
    try:
        from routes.predict import predict_bp
        app.register_blueprint(predict_bp)
        print("[OK] Route /predict enregistree")
    except ImportError as e:
        print(f"[ERROR] Impossible d'importer routes.predict: {str(e)}")
        print("[INFO] Installez les dependances: pip install -r App/requirements.txt")
    except Exception as e:
        print(f"[ERROR] Erreur lors de l'import de routes.predict: {str(e)}")
else:
    print("[INFO] Route /predict non disponible - modeles ML non charges")

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    print("\n[OK] Serveur Flask pret sur http://127.0.0.1:5000")
    app.run(debug=True, port=5000)