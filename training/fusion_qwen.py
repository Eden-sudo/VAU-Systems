"""
Script para la fusion de adaptadores LoRA con el modelo base Qwen2-VL.
Genera un modelo consolidado para optimizar tiempos de carga e inferencia.
"""

import os
import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from peft import PeftModel

# Resolucion dinamica de rutas
# Se asume que la carpeta 'models' se encuentra en la raiz del proyecto
BASE_DIR = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
RUTA_BASE = os.path.join(BASE_DIR, "models", "Qwen2-VL-2B-Instruct")
RUTA_ADAPTADOR = os.path.join(BASE_DIR, "models", "Qwen-VAU-Seguridad-Adapter")
RUTA_SALIDA = os.path.join(BASE_DIR, "models", "Qwen-VAU-Global-Optimizado")

def fundir_modelo():
    """Ejecuta el merge de pesos del adaptador en la estructura del modelo base."""
    print("[INFO] Iniciando proceso de fusion de pesos.")

    # 1. Carga de modelo base
    # Se utiliza CPU para preservar memoria de video durante la operacion de merge
    print("[INFO] Paso 1/4: Cargando modelo base en memoria RAM.")
    modelo_base = Qwen2VLForConditionalGeneration.from_pretrained(
        RUTA_BASE,
        torch_dtype=torch.float16,
        device_map="cpu",  
        low_cpu_mem_usage=True
    )

    # 2. Aplicacion del adaptador Peft
    print("[INFO] Paso 2/4: Acoplando adaptador LoRA.")
    modelo_peft = PeftModel.from_pretrained(
        modelo_base,  
        RUTA_ADAPTADOR,
        device_map="cpu"
    )

    # 3. Fusion de matrices (Merge and Unload)
    print("[INFO] Paso 3/4: Fusionando matrices de pesos.")
    modelo_global = modelo_peft.merge_and_unload()

    # 4. Exportacion de artefactos
    print(f"[INFO] Paso 4/4: Guardando modelo unificado en {RUTA_SALIDA}.")
    modelo_global.save_pretrained(RUTA_SALIDA)
    
    # El procesador debe persistirse junto al modelo para mantener la compatibilidad
    procesador = AutoProcessor.from_pretrained(RUTA_BASE)
    procesador.save_pretrained(RUTA_SALIDA)

    print("[INFO] Fusion completada. Modelo listo para despliegue.")

if __name__ == "__main__":
    fundir_modelo()
