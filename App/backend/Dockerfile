FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Installer dépendances système nécessaires (ffmpeg, libsndfile pour torchaudio, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ffmpeg libsndfile1 git curl && \
    rm -rf /var/lib/apt/lists/*

# Copier et installer les dépendances Python
COPY App/requirements.txt /app/requirements.txt
RUN pip install --upgrade pip
RUN pip install -r /app/requirements.txt

# Copier le projet
COPY . /app

WORKDIR /app/App/backend

# Variables d'environnement utiles
ENV BASE_MODEL_PATH=/app/models
ENV UPLOAD_FOLDER=/app/uploads

EXPOSE 5000

CMD ["python", "app.py"]
