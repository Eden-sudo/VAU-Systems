#!/bin/bash
set -e

echo "=========================================================="
echo "[VAU-EXPORT] COMPILADOR GGUF UNIFICADO (CMAKE)"
echo "=========================================================="

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MERGED_DIR="$BASE_DIR/models/Qwen-VAU-Global-Optimizado"
LLAMA_CPP_DIR="$BASE_DIR/export/llama.cpp"

TEMP_F16="$BASE_DIR/export/temp_modelo_f16.gguf"
OUTPUT_FINAL="$BASE_DIR/models/qwen2-vl-vau-int4.gguf"

echo "[VAU] 1. Convirtiendo Modelo Unificado (Visión + Texto)..."
python3 "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" "$MERGED_DIR" \
    --outfile "$TEMP_F16" \
    --outtype f16

echo "[VAU] 2. Compilando cuantizador con CMake..."
cd "$LLAMA_CPP_DIR"
mkdir -p build
cd build
cmake ..
cmake --build . --config Release --target llama-quantize -j 4
cd "$BASE_DIR"

echo "[VAU] 3. Cuantizando matriz a 4-bits (Q4_K_M)..."
"$LLAMA_CPP_DIR/build/bin/llama-quantize" "$TEMP_F16" "$OUTPUT_FINAL" q4_k_m

echo "[VAU] 4. Limpiando residuos temporales..."
rm -f "$TEMP_F16"

echo "----------------------------------------------------------"
echo "[EXITO] Transmutación completada."
echo "[INFO] Modelo Unificado: $OUTPUT_FINAL"
echo "=========================================================="
