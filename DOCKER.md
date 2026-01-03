# Docker quickstart

Small instructions to build and run the frontend + backend using Docker Compose.

Prerequisites
- Docker and Docker Compose installed.
- The `models/` folder (containing `scalers_v7.pkl`, `bert_finetuned_v5.pt`, `wavlm_finetuned_v5.pt`, ...) must exist at the project root.

Build and run (project root):
```bash
docker-compose build
docker-compose up -d
```

View logs:
```bash
docker-compose logs -f backend
```

Stopping & removing containers:
```bash
docker-compose down
```

Notes
- The Compose file mounts `./models` into the backend container at `/app/models` (read-only). Ensure model files are present before starting.
- To override the model path or port, set environment variables in `docker-compose.yml` or pass them before `docker-compose`:
  - Linux/macOS: `BASE_MODEL_PATH=/path/to/models PORT=5000 docker-compose up --build`
  - Windows (PowerShell): `$env:BASE_MODEL_PATH='C:\path\to\models'; docker-compose up --build`
- The backend container installs system packages (ffmpeg, libsndfile) and Python dependencies from `App/requirements.txt`. The image can be large because of PyTorch/Whisper.
- GPU support: the current backend Dockerfile uses the CPU image. If you need CUDA acceleration, I can add a GPU-enabled Dockerfile using an NVIDIA CUDA base image plus `nvidia-docker2`/`--gpus` usage.

Frontend
- The frontend is built in a Node stage and served by nginx on container port `80`, mapped to host port `3000` by default.

If you want, I can:
- Add a `.env` file to control `BASE_MODEL_PATH` and `PORT`.
- Provide a GPU-enabled backend Dockerfile.
