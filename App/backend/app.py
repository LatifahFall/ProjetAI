import os
import torch
import torch.nn as nn
import pickle
import whisper
from flask import Flask
from flask_cors import CORS
from transformers import AutoModel, BertTokenizer, AutoTokenizer

app = Flask(__name__)
CORS(app)

# --- Configuration des chemins ---
BASE_MODEL_PATH = 'E:\Projets\ProjetAI\models'
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

print("⏳ Chargement des modèles et scalers en mémoire...")

# 1. Scalers
with open(os.path.join(BASE_MODEL_PATH, 'scalers_v7.pkl'), 'rb') as f:
    app.scalers = pickle.load(f)
print("✅ Scalers chargés.")

# 2. Whisper
app.whisper_model = whisper.load_model("base").to(app.device)
print("✅ Whisper chargé.")

# 3. BERT + Poids fine-tunés
app.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
app.bert_model = BERTRegressionHead().to(app.device)
bert_weights = os.path.join(BASE_MODEL_PATH, 'bert_finetuned_v5.pt')
app.bert_model.load_state_dict(torch.load(bert_weights, map_location=app.device))
app.bert_model.eval()
print("✅ BERT fine-tuné chargé.")

# 4. WavLM + Poids fine-tunés
app.wavlm_model = WavLMRegressionHead().to(app.device)
wavlm_weights = os.path.join(BASE_MODEL_PATH, 'wavlm_finetuned_v5.pt')
app.wavlm_model.load_state_dict(torch.load(wavlm_weights, map_location=app.device))
app.wavlm_model.eval()
print("✅ WavLM fine-tuné chargé.")

# ══════════════════════════════════════════════════════════════════════════════
# LANCEMENT
# ══════════════════════════════════════════════════════════════════════════════

# Import de la route (après avoir défini les classes pour éviter les erreurs)
from routes.predict import predict_bp
app.register_blueprint(predict_bp)

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    print("\n🚀 Serveur Flask prêt sur http://127.0.0.1:5000")
    app.run(debug=True, port=5000)