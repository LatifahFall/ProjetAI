"""
Configuration centralisée pour le projet MLOps
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

# Chemins de base
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
MLRUNS_DIR = BASE_DIR / "mlruns"
LOGS_DIR = BASE_DIR / "logs"
CONFIG_DIR = BASE_DIR / "config"

# Créer les dossiers s'ils n'existent pas
MODELS_DIR.mkdir(exist_ok=True)
MLRUNS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)
CONFIG_DIR.mkdir(exist_ok=True)

# Environnement (dev, staging, prod)
ENVIRONMENT = os.getenv("ENVIRONMENT", "dev")

# MLflow Configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"file://{MLRUNS_DIR.absolute()}")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "personality_prediction")

# Chemins des modèles
MODEL_STORAGE_PATH = os.getenv("MODEL_STORAGE_PATH", str(MODELS_DIR))
MODEL_VERSION_FORMAT = os.getenv("MODEL_VERSION_FORMAT", "v{timestamp}")

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = LOGS_DIR / f"app_{ENVIRONMENT}.log"

# API Configuration (pour Phase 4)
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# Secrets (à définir dans .env)
# Exemple: API_KEY, DATABASE_URL, etc.

