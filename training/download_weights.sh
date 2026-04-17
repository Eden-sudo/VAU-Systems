#!/bin/bash
# download_weights.sh

echo "[*] Descargando modelo VLM base (Moondream2)..."

export HF_HUB_ENABLE_HF_TRANSFER=1

huggingface-cli download vikhyatk/moondream2 \
    --local-dir ./models/base_model \
    --local-dir-use-symlinks False

echo "[*] Descarga completada en ./models/base_model"
