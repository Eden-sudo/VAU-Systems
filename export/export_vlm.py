import os
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import PeftModel

print("[VAU-EXPORT] Iniciando forja de modelo Qwen2-VL...")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BASE_MODEL = os.path.join(BASE_DIR, "models", "Qwen2-VL-2B-Instruct")
# FIX: Ruta correcta donde train_qlora.py guardará el conocimiento
LORA_DIR = os.path.join(BASE_DIR, "models", "Qwen-VAU-Seguridad-Adapter")
OUTPUT_DIR = os.path.join(BASE_DIR, "export", "qwen_merged_fp16")

if not os.path.exists(LORA_DIR):
    print(f"[!] Error: No se encontró el adaptador LoRA en {LORA_DIR}")
    print("[!] Entrena el modelo primero antes de intentar fusionarlo.")
    exit(1)

print("[VAU] 1. Cargando arquitectura multimodal base (Qwen2-VL)...")
base_model = Qwen2VLForConditionalGeneration.from_pretrained(
    BASE_MODEL,  
    device_map="cpu",  
    torch_dtype=torch.bfloat16
)

print("[VAU] 2. Inyectando pesos de entrenamiento (LoRA)...")
model = PeftModel.from_pretrained(base_model, LORA_DIR)

print("[VAU] 3. Colapsando matriz de pesos (Merge & Unload)...")
model = model.merge_and_unload()

print(f"[VAU] 4. Escribiendo binarios unificados en {OUTPUT_DIR}...")
os.makedirs(OUTPUT_DIR, exist_ok=True)
model.save_pretrained(OUTPUT_DIR, safe_serialization=True)

processor = AutoProcessor.from_pretrained(BASE_MODEL)
processor.save_pretrained(OUTPUT_DIR)

print("[EXITO] Fase 1 completada. Modelo VLM fusionado correctamente.")
