#!/bin/bash

# Abortar en caso de error
set -e

echo "=========================================================="
echo "[VAU-EXPORT] COMPILADOR GGUF PARA JETSON ORIN NANO"
echo "=========================================================="

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Apuntamos directamente al modelo que extrajiste del .tar.gz
MERGED_DIR="$BASE_DIR/models/Qwen-VAU-Global-Optimizado"
LLAMA_CPP_DIR="$BASE_DIR/export/llama.cpp"

# Nombres de los archivos finales
TEMP_F16="$BASE_DIR/export/temp_texto_f16.gguf"
OUTPUT_TEXT="$BASE_DIR/models/qwen2-vl-vau-int4.gguf"
OUTPUT_VISION="$BASE_DIR/models/mmproj-qwen-vau-f16.gguf"

if [ ! -d "$MERGED_DIR" ]; then
    echo "[-] Error: No se encontró el modelo en $MERGED_DIR"
    exit 1
fi

echo "[VAU] 1. Preparando motor de compilación llama.cpp..."
if [ ! -d "$LLAMA_CPP_DIR" ]; then
    git clone https://github.com/ggerganov/llama.cpp.git "$LLAMA_CPP_DIR"
    pip install -r "$LLAMA_CPP_DIR/requirements.txt"
else
    cd "$LLAMA_CPP_DIR" && git pull && cd "$BASE_DIR"
fi

echo "[VAU] 2.1 Extrayendo el Cerebro de Texto (LLM)..."
python3 "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" "$MERGED_DIR" \
    --outfile "$TEMP_F16" \
    --outtype f16

echo "[VAU] 2.2 Extrayendo el Ojo Multimodal (Vision Projector)..."
python3 "$LLAMA_CPP_DIR/examples/llava/convert_image_encoder_to_gguf.py" \
    -m "$MERGED_DIR" \
    --llava-projector "$MERGED_DIR" \
    --output-dir "$BASE_DIR/models" \
    --projector-type qwen2vl

# Renombramos el proyector generado para mantener el orden
mv "$BASE_DIR/models/llava.projector" "$OUTPUT_VISION" 2>/dev/null || true

echo "[VAU] 3. Compilando matriz de texto a 4-bits (Q4_K_M)..."
cd "$LLAMA_CPP_DIR" && make llama-quantize
cd "$BASE_DIR"

"$LLAMA_CPP_DIR/llama-quantize" "$TEMP_F16" "$OUTPUT_TEXT" q4_k_m

echo "[VAU] 4. Limpiando residuos temporales..."
rm -f "$TEMP_F16"

echo "----------------------------------------------------------"
echo "[EXITO] Transmutación completada."
echo "[INFO] Cerebro (Texto): $OUTPUT_TEXT"
echo "[INFO] Ojo (Visión): $OUTPUT_VISION"
echo "=========================================================="
