import os
import torch
import torch.nn as nn
import pickle
from pathlib import Path
from flask import Flask
from flask_cors import CORS
import numpy

# Compatibilité Torch
try:
    from numpy.dtypes import Float32DType
except ImportError:
    Float32DType = None

from transformers import (
    AutoModel, 
    WhisperProcessor, 
    WhisperForConditionalGeneration,
    BertTokenizer
)

safe_globals = [numpy._core.multiarray.scalar, numpy.dtype]
if Float32DType: safe_globals.append(Float32DType)
torch.serialization.add_safe_globals(safe_globals)

app = Flask(__name__)
CORS(app)


BASE_DIR = Path(__file__).parent.parent.parent
BASE_MODEL_PATH = str(BASE_DIR / 'models')

app.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Architectures ---

class InterTraitAttention(nn.Module):
    def __init__(self, hidden_dim, n_traits=5, n_heads=8):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=n_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
    def forward(self, x):
        attn_out, _ = self.attention(x, x, x)
        return self.norm(x + attn_out)

class MTLModelV7(nn.Module):
    def __init__(self, input_dim=2392, hidden_dims=[896, 512, 256], n_traits=5, dropout=0.30, n_heads=8):
        super().__init__()
        self.n_traits = n_traits
        self.shared_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.GELU(),
            nn.Dropout(dropout * 0.7)
        )
        self.task_encoders = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dims[0], hidden_dims[0]), nn.BatchNorm1d(hidden_dims[0]), nn.GELU(), nn.Dropout(dropout * 0.8)) 
            for _ in range(n_traits)
        ])
        self.inter_trait_attn = InterTraitAttention(hidden_dims[0], n_traits, n_heads)
        self.shared_layers = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dims[i], hidden_dims[i+1]), nn.BatchNorm1d(hidden_dims[i+1]), nn.GELU(), nn.Dropout(dropout * (0.9 - i*0.15))) 
            for i in range(len(hidden_dims)-1)
        ])
        final_dim = hidden_dims[-1]
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(final_dim, final_dim // 2), nn.LayerNorm(final_dim // 2), nn.GELU(), nn.Dropout(dropout * 0.5), nn.Linear(final_dim // 2, final_dim // 4), nn.GELU(), nn.Linear(final_dim // 4, 1)) 
            for _ in range(n_traits)
        ])
    def forward(self, x):
        shared_feat = self.shared_encoder(x)
        task_feats = torch.stack([encoder(shared_feat) for encoder in self.task_encoders], dim=1)
        task_feats = self.inter_trait_attn(task_feats)
        for layer in self.shared_layers:
            task_feats = torch.stack([layer(task_feats[:, i, :]) for i in range(self.n_traits)], dim=1)
        preds = [torch.sigmoid(self.heads[i](task_feats[:, i, :])) for i in range(self.n_traits)]
        return torch.cat(preds, dim=1)

class WavLMWrapper(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.wavlm = base_model
    def forward(self, waveform):
        outputs = self.wavlm(waveform)
        hidden = outputs.last_hidden_state
        return torch.cat([hidden.mean(dim=1), hidden.std(dim=1)], dim=1)

class BERTWrapper(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.bert = base_model
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        token_emb = outputs.last_hidden_state
        mask_expanded = attention_mask.unsqueeze(-1).expand(token_emb.size()).float()
        return torch.sum(token_emb * mask_expanded, 1) / torch.clamp(mask_expanded.sum(1), min=1e-9)

# --- Initialisation ---

print("[*] Démarrage du serveur AI...")

try:
    app.whisper_processor = WhisperProcessor.from_pretrained("distil-whisper/distil-medium.en")
    app.whisper_model = WhisperForConditionalGeneration.from_pretrained("distil-whisper/distil-medium.en").to(app.device)
    
    wavlm_base = AutoModel.from_pretrained("microsoft/wavlm-base-plus")
    app.wavlm_model = WavLMWrapper(wavlm_base).to(app.device)
    app.wavlm_model.load_state_dict(torch.load(os.path.join(BASE_MODEL_PATH, 'wavlm_finetuned_v5.pt'), map_location=app.device), strict=False)
    
    bert_base = AutoModel.from_pretrained("bert-base-uncased")
    app.bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    app.bert_model = BERTWrapper(bert_base).to(app.device)
    app.bert_model.load_state_dict(torch.load(os.path.join(BASE_MODEL_PATH, 'bert_finetuned_v5.pt'), map_location=app.device), strict=False)

    app.mtl_model = MTLModelV7(input_dim=2392).to(app.device)
    checkpoint = torch.load(os.path.join(BASE_MODEL_PATH, 'model_0_best-v7.pt'), map_location=app.device)
    app.mtl_model.load_state_dict(checkpoint.get('model_state_dict', checkpoint), strict=False)
    app.mtl_model.eval()
    
    with open(os.path.join(BASE_MODEL_PATH, 'scalers_v7.pkl'), 'rb') as f:
        app.scalers = pickle.load(f)

    print("[OK] Modèles chargés.")
except Exception as e:
    print(f"[ERROR] Échec chargement: {e}")

from routes.predict import predict_bp
app.register_blueprint(predict_bp)

if __name__ == '__main__':
    app.run(debug=True, port=5000)

