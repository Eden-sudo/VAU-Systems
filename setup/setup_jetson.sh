#!/bin/bash

# Abortar si ocurre algún error crítico
set -e

echo "[VAU-SYSTEM] INICIANDO CONFIGURACION EXCLUSIVA PARA JETSON ORIN NANO"
echo "------------------------------------------------------------------"

VENV_DIR="vau_env"

echo "[1/5] Instalando dependencias del sistema operativo..."
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv python3-dev \
    libopenblas-dev libopenmpi-dev libjpeg-dev zlib1g-dev \
    cmake git build-essential

echo "[2/5] Inicializando entorno virtual aislado..."
if [ ! -d "$VENV_DIR" ]; then
    # El flag system-site-packages es OBLIGATORIO en Jetson para leer TensorRT
    python3 -m venv $VENV_DIR --system-site-packages
    echo "[INFO] Entorno $VENV_DIR creado."
else
    echo "[SKIP] El entorno $VENV_DIR ya existe."
fi

PIP="$VENV_DIR/bin/pip"
$PIP install --upgrade pip

echo "[3/5] Inyectando PyTorch y Torchvision (Optimizados para ARM64/JetPack)..."
# Descarga los binarios nativos de NVIDIA. (Ajustado a JetPack 5.1.2 / 6.0)
$PIP install torch torchvision torchaudio --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v512

echo "[4/5] Instalando ecosistema de inferencia (YOLO y Qwen2-VL)..."
$PIP install ultralytics opencv-python pillow
$PIP install "transformers>=4.45.0" huggingface_hub accelerate qwen-vl-utils av decord

echo "[5/5] Compilando motor nativo NVIDIA (jetson-utils para Zero-Copy)..."
if [ ! -d "jetson-inference" ]; then
    echo "[INFO] Descargando y compilando código fuente C++ de NVIDIA..."
    git clone --recursive https://github.com/dusty-nv/jetson-inference
    cd jetson-inference
    mkdir build
    cd build
    cmake ../
    make -j$(nproc)
    sudo make install
    sudo ldconfig
    cd ../..
    echo "[INFO] Libreria jetson-utils compilada e instalada."
else
    echo "[SKIP] Repositorio jetson-inference ya clonado."
fi

echo "------------------------------------------------------------------"
echo "[EXITO] Entorno Jetson configurado correctamente."
echo "[INFO] Activa tu entorno ejecutando: source $VENV_DIR/bin/activate"
