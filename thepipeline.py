"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ PIPELINE BIG FIVE - VERSION CORRIGÉE                                         ║
║                                                                              ║
║ Fonctionnalités:                                                            ║
║ ✅ Audio <15s  → Duplication intelligente (pas de zero-padding)             ║
║ ✅ Audio =15s  → Traitement direct                                          ║
║ ✅ Audio >15s  → Découpage en segments + moyenne des prédictions           ║
║ ✅ Comparaison avec vraies valeurs du CSV                                   ║
║ ✅ Focus sur UN audio à la fois                                             ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
import librosa
import soundfile as sf
import opensmile
import pickle

from transformers import (
    AutoModel,
    AutoTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration
)

import matplotlib.pyplot as plt
from math import pi

warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PipelineConfig:
    """Configuration du pipeline"""
    # Chemins des modèles
    wavlm_checkpoint: str = "/kaggle/input/models/pytorch/default/5/wavlm_finetuned_v5.pt"
    bert_checkpoint: str = "/kaggle/input/models/pytorch/default/5/bert_finetuned_v5.pt"
    mtl_checkpoint: str = "/kaggle/input/models/pytorch/default/5/model_0_best-v7.pt"
    scalers_path: str = "/kaggle/input/models/pytorch/default/5/scalers_v7.pkl"
    
    # Chemin des vraies valeurs
    ground_truth_csv: str = "/kaggle/input/organized-dataset/test_labels.csv"
    
    # Paramètres audio
    sample_rate: int = 16000
    segment_duration_sec: float = 15.0
    max_samples_per_segment: int = 16000 * 15  # 240000 samples
    
    # Paramètres texte
    max_text_length: int = 512
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Traits de personnalité (ordre important!)
    traits: List[str] = None
    
    def __post_init__(self):
        if self.traits is None:
            self.traits = [
                'Extraversion',
                'Agreeableness', 
                'Conscientiousness',
                'Neuroticism',
                'Openness'
            ]

# ══════════════════════════════════════════════════════════════════════════════
# WRAPPERS POUR MODÈLES
# ══════════════════════════════════════════════════════════════════════════════

class WavLMWrapper(nn.Module):
    """Wrapper pour WavLM fine-tuné"""
    def __init__(self, base_model):
        super().__init__()
        self.wavlm = base_model
        hidden_size = base_model.config.hidden_size
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size * 2, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 5)
        )
    
    def forward(self, waveform):
        outputs = self.wavlm(waveform)
        hidden = outputs.last_hidden_state
        mean_pool = hidden.mean(dim=1)
        std_pool = hidden.std(dim=1)
        return torch.cat([mean_pool, std_pool], dim=1)


class BERTWrapper(nn.Module):
    """Wrapper pour BERT fine-tuné"""
    def __init__(self, base_model):
        super().__init__()
        self.bert = base_model
        hidden_size = base_model.config.hidden_size
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 5)
        )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        token_emb = outputs.last_hidden_state
        mask_expanded = attention_mask.unsqueeze(-1).expand(token_emb.size()).float()
        sum_emb = torch.sum(token_emb * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        return sum_emb / sum_mask

# ══════════════════════════════════════════════════════════════════════════════
# ARCHITECTURE MTL V7
# ══════════════════════════════════════════════════════════════════════════════

class InterTraitAttention(nn.Module):
    def __init__(self, hidden_dim, n_traits=5, n_heads=8):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x):
        attn_out, _ = self.attention(x, x, x)
        return self.norm(x + attn_out)


class MTLModelV7(nn.Module):
    def __init__(self, input_dim, hidden_dims=[896, 512, 256], n_traits=5, dropout=0.30, n_heads=8):
        super().__init__()
        self.n_traits = n_traits
        
        self.shared_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.GELU(),
            nn.Dropout(dropout * 0.7)
        )
        
        self.task_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dims[0], hidden_dims[0]),
                nn.BatchNorm1d(hidden_dims[0]),
                nn.GELU(),
                nn.Dropout(dropout * 0.8)
            ) for _ in range(n_traits)
        ])
        
        self.inter_trait_attn = InterTraitAttention(hidden_dims[0], n_traits, n_heads)
        
        self.shared_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                nn.BatchNorm1d(hidden_dims[i+1]),
                nn.GELU(),
                nn.Dropout(dropout * (0.9 - i*0.15))
            ) for i in range(len(hidden_dims)-1)
        ])
        
        final_dim = hidden_dims[-1]
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(final_dim, final_dim // 2),
                nn.LayerNorm(final_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(final_dim // 2, final_dim // 4),
                nn.GELU(),
                nn.Linear(final_dim // 4, 1)
            ) for _ in range(n_traits)
        ])
    
    def forward(self, x):
        shared_feat = self.shared_encoder(x)
        task_feats = torch.stack([encoder(shared_feat) for encoder in self.task_encoders], dim=1)
        task_feats = self.inter_trait_attn(task_feats)
        
        for layer in self.shared_layers:
            task_feats = torch.stack([layer(task_feats[:, i, :]) for i in range(self.n_traits)], dim=1)
        
        preds = [torch.sigmoid(self.heads[i](task_feats[:, i, :])) for i in range(self.n_traits)]
        return torch.cat(preds, dim=1)

# ══════════════════════════════════════════════════════════════════════════════
# PREPROCESSING AUDIO
# ══════════════════════════════════════════════════════════════════════════════

class AudioPreprocessor:
    """Prétraitement audio avec gestion intelligente de la durée"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
    
    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """Charge un fichier audio"""
        try:
            waveform, sr = torchaudio.load(str(audio_path))
            waveform = waveform.numpy()
        except:
            waveform, sr = librosa.load(str(audio_path), sr=None, mono=False)
            if waveform.ndim == 1:
                waveform = waveform[np.newaxis, :]
        
        return waveform, sr
    
    def duplicate_to_target_length(self, waveform: torch.Tensor, target_length: int) -> torch.Tensor:
        """
        Duplique intelligemment le signal pour atteindre la longueur cible
        SANS utiliser de zero-padding
        """
        current_length = waveform.shape[0]
        
        if current_length >= target_length:
            return waveform[:target_length]
        
        # Calculer le nombre de répétitions nécessaires
        n_repeats = int(np.ceil(target_length / current_length))
        
        # Répéter le signal et tronquer à la longueur exacte
        duplicated = waveform.repeat(n_repeats)
        return duplicated[:target_length]
    
    def split_long_audio(self, waveform: torch.Tensor, audio_path: str) -> List[Tuple[torch.Tensor, str]]:
        """
        Découpe un audio long en segments de 15s
        Retourne liste de (segment_waveform, segment_name)
        """
        total_samples = waveform.shape[0]
        segment_samples = self.config.max_samples_per_segment
        
        # Cas 1: Audio <= 15s
        if total_samples <= segment_samples:
            return [(waveform, Path(audio_path).stem)]
        
        # Cas 2: Audio > 15s → Découpage
        n_segments = int(np.ceil(total_samples / segment_samples))
        segments = []
        base_name = Path(audio_path).stem
        
        for i in range(n_segments):
            start = i * segment_samples
            end = min((i + 1) * segment_samples, total_samples)
            segment = waveform[start:end]
            
            # Si dernier segment < 15s, on duplique intelligemment
            if segment.shape[0] < segment_samples:
                segment = self.duplicate_to_target_length(segment, segment_samples)
            
            segment_name = f"{base_name}_part{i+1:02d}"
            segments.append((segment, segment_name))
        
        return segments
    
    def preprocess(self, audio_path: str) -> List[Tuple[torch.Tensor, str, float]]:
        """
        Pipeline complet de preprocessing
        
        Returns:
            Liste de (waveform_preprocessed, segment_name, duration_sec)
        """
        # Chargement
        waveform, sr = self.load_audio(audio_path)
        waveform = torch.from_numpy(waveform).float()
        
        # Conversion mono
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Resampling si nécessaire
        if sr != self.config.sample_rate:
            resampler = T.Resample(sr, self.config.sample_rate)
            waveform = resampler(waveform)
        
        waveform = waveform.squeeze(0)
        
        # Durée originale
        original_duration = waveform.shape[0] / self.config.sample_rate
        
        # Traitement selon durée
        if original_duration < self.config.segment_duration_sec:
            # Cas 1: Audio court → Duplication
            print(f"   ⚙️  Audio court ({original_duration:.1f}s) → Duplication à 15s")
            waveform_padded = self.duplicate_to_target_length(waveform, self.config.max_samples_per_segment)
            segments = [(waveform_padded, Path(audio_path).stem, original_duration)]
        
        elif abs(original_duration - self.config.segment_duration_sec) < 0.5:
            # Cas 2: Audio ~15s → Traitement direct
            print(f"   ⚙️  Audio ~15s ({original_duration:.1f}s) → Traitement direct")
            segments = [(waveform[:self.config.max_samples_per_segment], Path(audio_path).stem, original_duration)]
        
        else:
            # Cas 3: Audio long → Découpage
            segments_raw = self.split_long_audio(waveform, audio_path)
            print(f"   ⚙️  Audio long ({original_duration:.1f}s) → {len(segments_raw)} segments de 15s")
            segments = [(seg, name, self.config.segment_duration_sec) for seg, name in segments_raw]
        
        # Normalisation de chaque segment
        normalized_segments = []
        for segment, name, duration in segments:
            segment_norm = (segment - segment.mean()) / (segment.std() + 1e-8)
            normalized_segments.append((segment_norm, name, duration))
        
        return normalized_segments

# ══════════════════════════════════════════════════════════════════════════════
# EXTRACTION FEATURES
# ══════════════════════════════════════════════════════════════════════════════

class FeatureExtractor:
    """Extraction de features multimodales"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        print("🔧 Initialisation des extracteurs...")
        
        # WavLM Fine-Tuné
        print("   • WavLM Fine-Tuned...", end=" ")
        wavlm_base = AutoModel.from_pretrained("microsoft/wavlm-base-plus")
        self.wavlm = WavLMWrapper(wavlm_base).to(self.device)
        self.wavlm.load_state_dict(torch.load(config.wavlm_checkpoint, map_location=self.device))
        self.wavlm.eval()
        print("✅")
        
        # BERT Fine-Tuné
        print("   • BERT Fine-Tuned...", end=" ")
        bert_base = AutoModel.from_pretrained("bert-base-uncased")
        self.bert = BERTWrapper(bert_base).to(self.device)
        self.bert.load_state_dict(torch.load(config.bert_checkpoint, map_location=self.device))
        self.bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.bert.eval()
        print("✅")
        
        # Distil-Whisper
        print("   • Distil-Whisper...", end=" ")
        self.whisper_processor = WhisperProcessor.from_pretrained("distil-whisper/distil-medium.en")
        self.whisper_model = WhisperForConditionalGeneration.from_pretrained("distil-whisper/distil-medium.en").to(self.device)
        self.whisper_model.eval()
        print("✅")
        
        # OpenSMILE
        print("   • OpenSMILE...", end=" ")
        self.smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals
        )
        print("✅\n")
    
    def transcribe(self, waveform: torch.Tensor) -> str:
        """Transcription avec Distil-Whisper"""
        try:
            audio = waveform.cpu().numpy()
            inputs = self.whisper_processor(
                audio,
                sampling_rate=16000,
                return_tensors="pt"
            ).input_features.to(self.device)
            
            with torch.no_grad():
                pred_ids = self.whisper_model.generate(inputs)
            
            text = self.whisper_processor.batch_decode(pred_ids, skip_special_tokens=True)[0]
            return text.strip()
        except Exception as e:
            print(f"⚠️ Erreur transcription: {e}")
            return ""
    
    def extract_wavlm(self, waveform: torch.Tensor) -> np.ndarray:
        """Features WavLM (1536D)"""
        try:
            with torch.no_grad():
                waveform_gpu = waveform.unsqueeze(0).to(self.device)
                features = self.wavlm(waveform_gpu)
            return features.squeeze(0).cpu().numpy()
        except:
            return np.zeros(1536, dtype=np.float32)
    
    def extract_bert(self, text: str) -> np.ndarray:
        """Features BERT (768D)"""
        try:
            if len(text.strip()) < 3:
                text = "Audio speech sample"
            
            inputs = self.bert_tokenizer(
                text,
                max_length=self.config.max_text_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            ).to(self.device)
            
            with torch.no_grad():
                features = self.bert(inputs['input_ids'], inputs['attention_mask'])
            
            return features.squeeze(0).cpu().numpy()
        except:
            return np.zeros(768, dtype=np.float32)
    
    def extract_opensmile(self, waveform: torch.Tensor) -> np.ndarray:
        """Features OpenSMILE (88D)"""
        try:
            import tempfile
            audio_np = waveform.cpu().numpy()
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                sf.write(tmp.name, audio_np, self.config.sample_rate)
                features = self.smile.process_file(tmp.name).values.flatten()
                os.unlink(tmp.name)
            
            return features
        except:
            return np.zeros(88, dtype=np.float32)
    
    def extract(self, waveform: torch.Tensor) -> Dict[str, np.ndarray]:
        """Extraction complète"""
        transcription = self.transcribe(waveform)
        
        wavlm_features = self.extract_wavlm(waveform)
        bert_features = self.extract_bert(transcription)
        opensmile_features = self.extract_opensmile(waveform)
        
        return {
            'wavlm': wavlm_features,
            'opensmile': opensmile_features,
            'bert': bert_features,
            'transcription': transcription
        }

# ══════════════════════════════════════════════════════════════════════════════
# PRÉDICTION MTL
# ══════════════════════════════════════════════════════════════════════════════

class MTLPredictor:
    """Prédiction Big Five avec MTL V7"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        print("🔧 Chargement MTL V7...")
        
        # Scalers
        print("   • Scalers...", end=" ")
        with open(config.scalers_path, 'rb') as f:
            self.scalers = pickle.load(f)
        print("✅")
        
        # Modèle MTL
        print("   • Modèle MTL...", end=" ")
        self.model = MTLModelV7(
            input_dim=2392,  # 1536 + 88 + 768
            hidden_dims=[896, 512, 256],
            dropout=0.30,
            n_heads=8
        ).to(self.device)
        
        checkpoint = torch.load(config.mtl_checkpoint, map_location=self.device, weights_only=False)
        state = checkpoint.get('model_state_dict', checkpoint)
        self.model.load_state_dict(state, strict=False)
        self.model.eval()
        print("✅\n")
    
    def predict(self, features: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Prédiction à partir des features"""
        # Standardisation
        wavlm_scaled = self.scalers['wavlm'].transform(features['wavlm'].reshape(1, -1))
        opensmile_scaled = self.scalers['opensmile'].transform(features['opensmile'].reshape(1, -1))
        bert_scaled = self.scalers['bert'].transform(features['bert'].reshape(1, -1))
        
        # Fusion
        X = np.concatenate([wavlm_scaled, opensmile_scaled, bert_scaled], axis=1)
        
        # Prédiction
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            preds = self.model(X_tensor).cpu().numpy()[0]
        
        # Format résultat
        results = {trait: float(pred) for trait, pred in zip(self.config.traits, preds)}
        return results

# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

class BigFivePipeline:
    """Pipeline complet de prédiction Big Five"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        
        print("╔" + "═" * 78 + "╗")
        print("║" + " " * 20 + "🚀 BIG FIVE PIPELINE V7 CORRIGÉ" + " " * 26 + "║")
        print("╚" + "═" * 78 + "╝\n")
        
        self.preprocessor = AudioPreprocessor(config)
        self.extractor = FeatureExtractor(config)
        self.predictor = MTLPredictor(config)
        
        # Chargement des vraies valeurs
        self.ground_truth = self._load_ground_truth()
        
        print("✅ Pipeline prêt!\n")
    
    def _load_ground_truth(self) -> pd.DataFrame:
        """Charge le CSV des vraies valeurs"""
        try:
            df = pd.read_csv(self.config.ground_truth_csv)
            # Renommer les colonnes pour correspondre aux traits
            rename_map = {
                'ValueExtraversion': 'Extraversion',
                'ValueAgreeableness': 'Agreeableness',
                'ValueConscientiousness': 'Conscientiousness',
                'ValueNeurotisicm': 'Neuroticism',
                'ValueOpenness': 'Openness'
            }
            df = df.rename(columns=rename_map)
            print(f"✅ Chargé {len(df)} vraies valeurs depuis {Path(self.config.ground_truth_csv).name}\n")
            return df
        except Exception as e:
            print(f"⚠️ Impossible de charger les vraies valeurs: {e}\n")
            return None
    
    def get_ground_truth(self, filename: str) -> Optional[Dict[str, float]]:
        """Récupère les vraies valeurs pour un fichier"""
        if self.ground_truth is None:
            return None
        
        row = self.ground_truth[self.ground_truth['Videoname'] == filename]
        if row.empty:
            return None
        
        return {
            trait: float(row[trait].values[0])
            for trait in self.config.traits
        }
    
    def predict(self, audio_path: str) -> Dict:
        """
        Prédiction sur UN audio
        
        Returns:
            {
                'filename': "audio.wav",
                'original_duration': 8.5,
                'n_segments': 1 ou >1,
                'processing_method': "duplication" | "direct" | "split",
                'predictions': {'Extraversion': 0.72, ...},
                'ground_truth': {'Extraversion': 0.65, ...} ou None,
                'errors': {'Extraversion': 0.07, ...} ou None,
                'mae': 0.05,
                'transcription': "...",
                'segments_details': [...] si multi-segments,
                'timestamp': "..."
            }
        """
        filename = Path(audio_path).name
        print(f"🎵 Traitement: {filename}")
        
        # 1. Preprocessing
        print("   📊 Preprocessing audio...")
        segments = self.preprocessor.preprocess(audio_path)
        n_segments = len(segments)
        original_duration = segments[0][2]  # Durée originale
        
        # Déterminer la méthode de traitement
        if original_duration < self.config.segment_duration_sec:
            processing_method = "duplication"
        elif abs(original_duration - self.config.segment_duration_sec) < 0.5:
            processing_method = "direct"
        else:
            processing_method = "split"
        
        # 2. Extraction + Prédiction pour chaque segment
        segments_results = []
        all_predictions = {trait: [] for trait in self.config.traits}
        all_transcriptions = []
        
        for i, (waveform, segment_name, duration) in enumerate(segments):
            if n_segments > 1:
                print(f"   • Segment {i+1}/{n_segments} ({segment_name})...", end=" ")
            else:
                print(f"   • Extraction features...", end=" ")
            
            # Extraction
            features = self.extractor.extract(waveform)
            
            # Prédiction
            predictions = self.predictor.predict(features)
            
            # Stockage
            for trait in self.config.traits:
                all_predictions[trait].append(predictions[trait])
            
            all_transcriptions.append(features['transcription'])
            
            segments_results.append({
                'segment_name': segment_name,
                'predictions': predictions,
                'transcription': features['transcription'],
                'duration': duration
            })
            
            print("✅")
        
        # 3. Agrégation si multi-segments
        if n_segments > 1:
            print(f"   • Agrégation {n_segments} segments...", end=" ")
            final_predictions = {
                trait: float(np.mean(all_predictions[trait]))
                for trait in self.config.traits
            }
            print("✅")
        else:
            final_predictions = segments_results[0]['predictions']
        
        full_transcription = " ".join(all_transcriptions)
        
        # 4. Récupération des vraies valeurs
        ground_truth = self.get_ground_truth(filename)
        
        # 5. Calcul des erreurs
        errors = None
        mae = None
        if ground_truth is not None:
            errors = {
                trait: abs(final_predictions[trait] - ground_truth[trait])
                for trait in self.config.traits
            }
            mae = np.mean(list(errors.values()))
        
        print()
        
        # 6. Construction du résultat
        result = {
            'filename': filename,
            'original_duration': original_duration,
            'n_segments': n_segments,
            'processing_method': processing_method,
            'predictions': final_predictions,
            'ground_truth': ground_truth,
            'errors': errors,
            'mae': mae,
            'transcription': full_transcription,
            'timestamp': datetime.now().isoformat()
        }
        
        if n_segments > 1:
            result['segments_details'] = segments_results
        
        return result
    
    def print_results(self, result: Dict):
        """Affiche les résultats de manière lisible"""
        print("╔" + "═" * 78 + "╗")
        print("║" + " " * 30 + "📊 RÉSULTATS" + " " * 35 + "║")
        print("╚" + "═" * 78 + "╝\n")
        
        print(f"📁 Fichier: {result['filename']}")
        print(f"⏱️  Durée originale: {result['original_duration']:.1f}s")
        print(f"🔧 Méthode: {result['processing_method']}")
        print(f"📦 Segments traités: {result['n_segments']}\n")
        
        # Transcription
        print("📝 Transcription:")
        trans = result['transcription']
        print(f"   {trans[:150]}{'...' if len(trans) > 150 else ''}\n")
        
        # Tableau des prédictions
        print("🎯 Prédictions Big Five:\n")
        print("   " + "─" * 70)
        print(f"   {'Trait':<20} {'Prédiction':<15} {'Vraie Valeur':<15} {'Erreur':<15}")
        print("   " + "─" * 70)
        
        for trait in self.config.traits:
            pred = result['predictions'][trait]
            
            if result['ground_truth'] is not None:
                true_val = result['ground_truth'][trait]
                error = result['errors'][trait]
                print(f"   {trait:<20} {pred:>6.4f}          {true_val:>6.4f}          {error:>6.4f}")
            else:
                print(f"   {trait:<20} {pred:>6.4f}          {'N/A':<15} {'N/A':<15}")
        
        print("   " + "─" * 70)
        
        if result['mae'] is not None:
            print(f"\n   📊 MAE (Mean Absolute Error): {result['mae']:.4f}")
        
        print()
        
        # Détails des segments si multi-segments
        if result['n_segments'] > 1 and 'segments_details' in result:
            print("📦 Détails par segment:\n")
            for i, seg in enumerate(result['segments_details'], 1):
                print(f"   Segment {i}: {seg['segment_name']}")
                for trait in self.config.traits:
                    print(f"      • {trait:<20}: {seg['predictions'][trait]:.4f}")
                print()
    
    def visualize_comparison(self, result: Dict, save_path: Optional[str] = None):
        """Visualisation radar avec comparaison prédictions vs vraies valeurs"""
        
        categories = self.config.traits
        predictions = [result['predictions'][t] for t in categories]
        
        N = len(categories)
        angles = [n / float(N) * 2 * pi for n in range(N)]
        predictions += predictions[:1]
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        # Tracer les prédictions
        ax.plot(angles, predictions, 'o-', linewidth=2, color='steelblue', label='Prédictions')
        ax.fill(angles, predictions, alpha=0.25, color='steelblue')
        
        # Tracer les vraies valeurs si disponibles
        if result['ground_truth'] is not None:
            ground_truth = [result['ground_truth'][t] for t in categories]
            ground_truth += ground_truth[:1]
            ax.plot(angles, ground_truth, 'o-', linewidth=2, color='crimson', label='Vraies Valeurs')
            ax.fill(angles, ground_truth, alpha=0.15, color='crimson')
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, size=11)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.grid(True)
        
        title = f"Profil Big Five - {result['filename']}"
        if result['mae'] is not None:
            title += f"\nMAE: {result['mae']:.4f}"
        
        plt.title(title, size=14, pad=20)
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def visualize_segments(self, result: Dict, save_path: Optional[str] = None):
        """Visualisation comparative des segments (si multi-segments)"""
        
        if result['n_segments'] == 1:
            print("⚠️ Un seul segment, utiliser visualize_comparison()")
            return
        
        n_segments = result['n_segments']
        segments_details = result['segments_details']
        
        fig, axes = plt.subplots(1, n_segments + 1, 
                                figsize=(5 * (n_segments + 1), 5),
                                subplot_kw=dict(polar=True))
        
        if n_segments == 1:
            axes = [axes]
        
        categories = self.config.traits
        N = len(categories)
        angles = [n / float(N) * 2 * pi for n in range(N)]
        
        # Tracer chaque segment
        for i, seg in enumerate(segments_details):
            values = [seg['predictions'][t] for t in categories]
            values += values[:1]
            
            axes[i].plot(angles + angles[:1], values, 'o-', linewidth=2, color='steelblue')
            axes[i].fill(angles + angles[:1], values, alpha=0.25, color='steelblue')
            axes[i].set_xticks(angles)
            axes[i].set_xticklabels(categories)
            axes[i].set_ylim(0, 1)
            axes[i].set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
            axes[i].grid(True)
            axes[i].set_title(f"Segment {i+1}", size=12)
        
        # Tracer la moyenne
        avg_values = [result['predictions'][t] for t in categories]
        avg_values += avg_values[:1]
        
        axes[-1].plot(angles + angles[:1], avg_values, 'o-', linewidth=3, color='darkred')
        axes[-1].fill(angles + angles[:1], avg_values, alpha=0.3, color='darkred')
        axes[-1].set_xticks(angles)
        axes[-1].set_xticklabels(categories)
        axes[-1].set_ylim(0, 1)
        axes[-1].set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        axes[-1].grid(True)
        axes[-1].set_title("MOYENNE", size=14, weight='bold')
        
        plt.suptitle(f"Comparaison Segments - {result['filename']}", size=16, y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# EXEMPLE D'UTILISATION
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    
    # Configuration
    config = PipelineConfig(
        wavlm_checkpoint="/kaggle/input/models/pytorch/default/5/wavlm_finetuned_v5.pt",
        bert_checkpoint="/kaggle/input/models/pytorch/default/5/bert_finetuned_v5.pt",
        mtl_checkpoint="/kaggle/input/models/pytorch/default/5/model_0_best-v7.pt",
        scalers_path="/kaggle/input/models/pytorch/default/5/scalers_v7.pkl",
        ground_truth_csv="/kaggle/input/organized-dataset/test_labels.csv"
    )
    
    # Initialisation du pipeline
    pipeline = BigFivePipeline(config)
    
    # ══════════════════════════════════════════════════════════════════════════
    # TEST 1: Audio court (<15s) - Duplication automatique
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "═" * 80)
    print("TEST 1: AUDIO COURT (Duplication)")
    print("═" * 80 + "\n")
    
    audio_short = "/kaggle/input/organized-dataset/test/aud/-Gl98Jn45Fs.004.wav"
    result_short = pipeline.predict(audio_short)
    pipeline.print_results(result_short)
    
    # Visualisation
    pipeline.visualize_comparison(
        result_short,
        save_path='/kaggle/working/profil_short.png'
    )
    
    # ══════════════════════════════════════════════════════════════════════════
    # TEST 2: Audio ~15s - Traitement direct
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "═" * 80)
    print("TEST 2: AUDIO ~15s (Traitement direct)")
    print("═" * 80 + "\n")
    
    # Remplacez par un fichier ~15s de votre dataset
    # audio_normal = "/kaggle/input/organized-dataset/test/aud/FICHIER_15s.wav"
    # result_normal = pipeline.predict(audio_normal)
    # pipeline.print_results(result_normal)
    # pipeline.visualize_comparison(result_normal, save_path='/kaggle/working/profil_normal.png')
    
    # ══════════════════════════════════════════════════════════════════════════
    # TEST 3: Audio long (>15s) - Découpage + Moyenne
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "═" * 80)
    print("TEST 3: AUDIO LONG (Découpage en segments)")
    print("═" * 80 + "\n")
    
    # Remplacez par un fichier >15s de votre dataset
    # audio_long = "/kaggle/input/organized-dataset/test/aud/FICHIER_LONG.wav"
    # result_long = pipeline.predict(audio_long)
    # pipeline.print_results(result_long)
    # 
    # # Visualisation comparative des segments
    # pipeline.visualize_segments(result_long, save_path='/kaggle/working/profil_long_segments.png')
    # 
    # # Visualisation avec moyenne vs vraies valeurs
    # pipeline.visualize_comparison(result_long, save_path='/kaggle/working/profil_long_comparison.png')
    
    # ══════════════════════════════════════════════════════════════════════════
    # Sauvegarde des résultats
    # ══════════════════════════════════════════════════════════════════════════
    
    # JSON détaillé
    with open('/kaggle/working/result_detailed.json', 'w', encoding='utf-8') as f:
        json.dump(result_short, f, indent=2, ensure_ascii=False)
    
    print("\n✅ Résultats sauvegardés:")
    print("   • /kaggle/working/result_detailed.json")
    print("   • /kaggle/working/profil_short.png")
    
    print("\n" + "═" * 80)
    print("✨ Pipeline terminé avec succès!")
    print("═" * 80)