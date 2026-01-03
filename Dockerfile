FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system deps needed by audio libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 git curl && rm -rf /var/lib/apt/lists/*

# Python deps
COPY App/requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

# Copy backend code
COPY App/backend /app/backend

# Copy models (if present via Git LFS or included in repo)
COPY models /app/models

ENV BASE_MODEL_PATH=/app/models
# Default PORT expected by Hugging Face Spaces. The Spaces runner will override $PORT when set.
ENV PORT=7860

WORKDIR /app/backend
# EXPOSE is ignored by Hugging Face but kept for clarity; use 7860 (Space default)
EXPOSE 7860

CMD ["python", "app.py"]
