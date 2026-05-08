#!/bin/bash

echo "=========================================================="
echo "[VAU-SYSTEM] INSTALADOR DE ESPACIO DE TRABAJO (HAL-Aware)"
echo "=========================================================="

VENV_DIR="vau_env"

# 1. Autocreación del Entorno Virtual
if [ ! -d "$VENV_DIR" ]; then
    echo "[VAU] Creando entorno virtual aislado en '$VENV_DIR'..."
    python3 -m venv $VENV_DIR
else
    echo "[VAU] Entorno virtual '$VENV_DIR' detectado. Usando existente."
fi

# Referencia directa al pip del entorno (Evita tener que hacer source para instalar)
PIP="$VENV_DIR/bin/pip"

echo "[VAU] 1. Actualizando herramientas base de compilación..."
$PIP install --upgrade pip wheel setuptools cython ninja

# Detección de Arquitectura
ARCH=$(uname -m)

echo "[VAU] 2. Instalando Core ML Stack para la arquitectura: $ARCH"
if [[ "$ARCH" == "aarch64" ]]; then
    echo "[!] Jetson Nano detectada. Evitando el PyTorch genérico de CPU..."
    # Instala utilidades necesarias para Jetson
    sudo apt-get update && sudo apt-get install -y libopenblas-dev libopenmpi-dev
    
    # Instala el PyTorch optimizado directamente de los servidores de NVIDIA
    $PIP install torch torchvision torchaudio --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v512
else
    echo "[!] PC/Laptop detectada. Instalando PyTorch estándar con soporte CUDA/CPU..."
    $PIP install torch torchvision
fi

echo "[VAU] 3. Instalando Stack de Visión Periférica (YOLO-World)..."
$PIP install ultralytics opencv-python pillow

echo "[VAU] 4. Configurando Cerebro Multimodal (Qwen2-VL)..."
# 'av' y 'decord' previene fugas de memoria al leer videos continuos
$PIP install "transformers>=4.45.0" huggingface_hub urllib3 accelerate qwen-vl-utils av decord

echo "[VAU] 5. Instalando Motor de Precisión (Grounding DINO)..."
$PIP install --no-build-isolation git+https://github.com/IDEA-Research/GroundingDINO.git

echo "----------------------------------------------------------"
echo "[OK] Instalación completada con éxito."
echo "[INFO] Tu ecosistema está listo y blindado."
echo "[INFO] Para empezar a programar, activa tu entorno con:"
echo "       source $VENV_DIR/bin/activate"
echo "=========================================================="
