from flask import Blueprint, request, jsonify, current_app
import torch
import librosa
import numpy as np
import os
import opensmile
from pydub import AudioSegment

predict_bp = Blueprint('predict', __name__)

smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
)

@predict_bp.route('/predict', methods=['POST'])
def predict():
    if 'audio_file' not in request.files:
        return jsonify({"error": "Fichier manquant"}), 400

    audio_file = request.files['audio_file']
    raw_path = "temp_raw_upload"
    temp_wav_path = "temp_converted.wav"
    
    audio_file.save(raw_path)

    try:
        # --- CONVERSION AUDIO (Fix: Format not recognised) ---
        # On force la conversion en WAV 16kHz Mono
        audio = AudioSegment.from_file(raw_path)
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(temp_wav_path, format="wav")

        # 1. Chargement Audio Standardisé
        speech, sr = librosa.load(temp_wav_path, sr=16000)

        # 2. Transcription (Whisper)
        input_features = current_app.whisper_processor(speech, sampling_rate=sr, return_tensors="pt").input_features.to(current_app.device)
        predicted_ids = current_app.whisper_model.generate(input_features)
        transcription = current_app.whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        if not transcription.strip(): transcription = "speech sample"

        # 3. Extraction + Normalisation par modalité
        # A. Audio (WavLM - 1536)
        with torch.no_grad():
            wav_feat = current_app.wavlm_model(torch.FloatTensor(speech).unsqueeze(0).to(current_app.device)).cpu().numpy()
            wav_scaled = current_app.scalers['wavlm'].transform(wav_feat)

        # B. Texte (BERT - 768)
        inputs = current_app.bert_tokenizer(transcription, return_tensors="pt", padding='max_length', truncation=True, max_length=512).to(current_app.device)
        with torch.no_grad():
            txt_feat = current_app.bert_model(inputs['input_ids'], inputs['attention_mask']).cpu().numpy()
            txt_scaled = current_app.scalers['bert'].transform(txt_feat)

        # C. Acoustique (OpenSmile - 88)
        smile_feat = smile.process_file(temp_wav_path).values
        smile_scaled = current_app.scalers['opensmile'].transform(smile_feat)

        # 4. Fusion des vecteurs (2392)
        combined_scaled = np.concatenate([wav_scaled, txt_scaled, smile_scaled], axis=1)
        final_input = torch.FloatTensor(combined_scaled).to(current_app.device)

        # 5. Prédiction Finale
        with torch.no_grad():
            preds = current_app.mtl_model(final_input).cpu().numpy()[0]

        # Nettoyage
        for f in [raw_path, temp_wav_path]:
            if os.path.exists(f): os.remove(f)

        return jsonify({
            "transcription": transcription,
            "traits": {
                "Extraversion": float(preds[0]),
                "Agreeableness": float(preds[1]),
                "Conscientiousness": float(preds[2]),
                "Neuroticism": float(preds[3]),
                "Openness": float(preds[4])
            }
        })

    except Exception as e:
        for f in [raw_path, temp_wav_path]:
            if os.path.exists(f): os.remove(f)
        print(f"Erreur Predict: {e}")
        return jsonify({"error": str(e)}), 500