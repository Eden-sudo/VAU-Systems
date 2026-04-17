import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from peft import PeftModel

print("[*] Iniciando exportación para Edge...")

# Rutas de origen (se debe ajustar directorios)
BASE_MODEL = "../traning/models/base_model"
LORA_DIR = "../traning/lora_adapters"
OUTPUT_DIR = "./merged_model"

print("[*] Cargando modelo base en memoria...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, 
    device_map="cpu", # Usamos CPU para la exportación para no pelear con la VRAM
    torch_dtype=torch.float16
)

print("[*] Fusionando conocimiento nuevo (LoRA)...")
model = PeftModel.from_pretrained(base_model, LORA_DIR)
model = model.merge_and_unload()

print(f"[*] Guardando modelo unificado en {OUTPUT_DIR}...")
model.save_pretrained(OUTPUT_DIR)

# Guardar también el procesador de imágenes
processor = AutoProcessor.from_pretrained(BASE_MODEL)
processor.save_pretrained(OUTPUT_DIR)

print("[*] ¡Listo! Copia la carpeta 'merged_model' a la Jetson Nano.")
