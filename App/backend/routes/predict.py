import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import numpy as np
import opensmile
import subprocess

from flask import Blueprint, request, jsonify, current_app
from transformers import AutoModel, AutoTokenizer

predict_bp = Blueprint('predict', __name__)

# ══════════════════════════════════════════════════════════════════════════════
# ARCHITECTURES EXACTES (Copie conforme du code d'entraînement)
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
        # Note: Dans le code d'entraînement, ils utilisent le regressor pour la prédiction
        # Mais pour l'extraction de features pure, on s'arrête souvent avant.
        # Ici, on va retourner les features concaténées pour le scaler.
        return pooled

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
        return pooled

# Initialisation OpenSMILE
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals
)

# ══════════════════════════════════════════════════════════════════════════════
# ROUTE DE PRÉDICTION
# ══════════════════════════════════════════════════════════════════════════════

@predict_bp.route('/predict', methods=['POST'])
def predict():
    if 'audio_file' not in request.files:
        return jsonify({"error": "Aucun fichier audio reçu"}), 400
    
    audio_file = request.files['audio_file']
    upload_dir = current_app.config['UPLOAD_FOLDER']

    raw_path = os.path.join(upload_dir, "raw_audio")
    wav_path = os.path.join(upload_dir, "current_capture.wav")

    # 1. Sauvegarde brute (format navigateur)
    audio_file.save(raw_path)

    # 2. Conversion forcée en WAV PCM 16kHz
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", raw_path,
            "-ac", "1",
            "-ar", "16000",
            "-f", "wav",
            wav_path
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True
    )

    temp_path = wav_path  # IMPORTANT


    print("\n[*] ANALYSE MULTIMODALE V5 EN COURS...")

    try:
        # 1. WHISPER (Transcription)
        print("[*] Step 1: Whisper Transcription...")
        transcription = current_app.whisper_model.transcribe(temp_path)["text"]
        print(f"   > '{transcription[:50]}...'")

        # 2. OPENSMILE (88 features)
        print("[*] Step 2: OpenSMILE Extraction...")
        smile_feats = smile.process_file(temp_path).values
        
        # 3. WAVLM (Pretraitement identique a WavLMFineTuneDataset)
        print("[*] Step 3: WavLM Processing...")
        waveform, sr = torchaudio.load(temp_path)
        if sr != 16000:
            waveform = T.Resample(sr, 16000)(waveform)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Troncature/Padding 15s
        max_samples = 16000 * 15
        if waveform.shape[1] > max_samples:
            waveform = waveform[:, :max_samples]
        else:
            waveform = F.pad(waveform, (0, max_samples - waveform.shape[1]))
        
        # Normalisation signal
        waveform = (waveform - waveform.mean()) / (waveform.std() + 1e-8)
        
        with torch.no_grad():
            wavlm_input = waveform.to(current_app.device)
            # On appelle l'architecture custom
            wavlm_raw = current_app.wavlm_model(wavlm_input).cpu().numpy()

        # 4. BERT (Pretraitement identique a BERTFineTuneDataset)
        print("[*] Step 4: BERT Processing...")
        inputs = current_app.bert_tokenizer(
            transcription, 
            max_length=512, 
            padding='max_length', 
            truncation=True, 
            return_tensors='pt'
        ).to(current_app.device)
        
        with torch.no_grad():
            bert_raw = current_app.bert_model(
                input_ids=inputs['input_ids'], 
                attention_mask=inputs['attention_mask']
            ).cpu().numpy()

        # 5. NORMALISATION (SCALERS)
        print("[*] Step 5: Scaling Features...")
        wav_scaled = current_app.scalers['wavlm'].transform(wavlm_raw)
        smile_scaled = current_app.scalers['opensmile'].transform(smile_feats)
        bert_scaled = current_app.scalers['bert'].transform(bert_raw)

        # 6. CONCATENATION
        X_combined = np.concatenate([wav_scaled, smile_scaled, bert_scaled], axis=1)
        print(f"[OK] Pipeline Termine. Vecteur final: {X_combined.shape}")

        # Pour l'instant, on renvoie les infos de succès. 
        # (Si vous avez le classifieur final, on l'ajoutera ici)
        return jsonify({
            "status": "success",
            "transcription": transcription,
            "traits": {
                "Extraversion": 0.0, 
                "Agreeableness": 0.0, 
                "Conscientiousness": 0.0, 
                "Neuroticism": 0.0, 
                "Openness": 0.0
            },
            "debug": {"shape": str(X_combined.shape)}
        })

    except Exception as e:
        print(f"[ERROR] Erreur: {str(e)}")
        return jsonify({"error": str(e)}), 500
    finally:
        for f in [raw_path, wav_path]:
            if os.path.exists(f):
                os.remove(f)
