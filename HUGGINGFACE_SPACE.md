# Deploy backend to Hugging Face Spaces (Docker)

Quick steps to deploy your Flask backend using the Docker option on Hugging Face Spaces.

1. Create a new Space on https://huggingface.co/spaces — choose "Docker" as SDK.

2. Push this repository to the new Space remote (either via web UI or using `huggingface-cli`).
   - Example using git remote (replace `<HF_REMOTE>` with the URL from the Space page):
     ```bash
     git remote add hf <HF_REMOTE>
     git push hf main
     ```

3. In the Space settings you can enable a GPU if needed (recommended for Whisper / large Torch workloads).

Notes / Caveats
- Models in `models/` are tracked with Git LFS here — Spaces supports LFS but large models may slow the build or exceed quotas. If that happens, host models externally (S3) and set `BASE_MODEL_PATH` to the download path.
- The Dockerfile at the repository root starts the Flask app on port `5000` (it is compatible with Spaces Docker). If HF requires a specific port in your environment, update `app.py` to use `PORT` env var (already supported).
- Builds that include large Torch/Whisper installations can be slow; consider using a GPU-enabled Space and a lightweight base image if you need acceleration.

If you want, I can:
- Create a minimal `start.sh` or `Dockerfile` variant optimized for HF build cache.
- Provide exact `huggingface-cli` commands to create the Space and push (you'll need to `huggingface-cli login`).
