#!/bin/bash
# setup_server.sh

echo "[*] Instalando Miniconda..."
wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda
source "$HOME/miniconda/etc/profile.d/conda.sh"

echo "[*] Creando entorno citep_vau..."
conda create -n citep_vau python=3.10 -y
conda activate citep_vau

echo "[*] Instalando dependencias base..."
# CUDA 12.1 es universal para gráficas modernas (RTX 30XX, 40XX, 50XX)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

echo "[*] Entorno listo. Ejecuta: conda activate citep_vau"
