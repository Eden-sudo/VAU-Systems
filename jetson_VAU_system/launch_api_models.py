"""
Lanzador Maestro de Inferencia VAU-Systems.
Orden de arranque estricto para evitar OOM en Jetson Nano.
"""
import os
import yaml
import asyncio

# 1. Ajustes del Allocator de PyTorch (Para YOLO)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:32"

import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.join(BASE_DIR, "inference") not in sys.path:
    sys.path.append(os.path.join(BASE_DIR, "inference"))

from inference.nodes.cognition import MotorCognicion
from inference.core.memory_hal import GestorMemoriaHAL
from inference.nodes.perception import MotorPercepcion
from inference.core.orchestrator import OrchestratorVAU

async def iniciar_motores():
    print("[INIT] Cargando configuración global...")
    with open(os.path.join(BASE_DIR, "config", "settings.yaml"), 'r') as f:
        config = yaml.safe_load(f)
        
    with open(os.path.join(BASE_DIR, "config", "prompts", "juez_seguridad.txt"), 'r') as f:
        prompt_maestro = f.read()

    # =======================================================
    # SECUENCIA DE ARRANQUE ESTRICTA (Protección de Memoria)
    # =======================================================
    
    print("[INIT] Paso 1: Bloqueando VRAM para el Juez...")
    nodo_cognicion = MotorCognicion(config, prompt_maestro)
    
    print("[INIT] Paso 2: Reservando Ring Buffer Zero-Copy...")
    hal = GestorMemoriaHAL(config)
    
    print("[INIT] Paso 3: Inicializando Sabueso (Ultralytics)...")
    nodo_percepcion = MotorPercepcion(config)
    
    print("[INIT] Paso 4: Enlazando Orquestador y ZeroMQ...")
    motor_central = OrchestratorVAU(config, nodo_percepcion, nodo_cognicion, hal)

    try:
        await motor_central.start()
    except KeyboardInterrupt:
        print("\n[INFO] Apagado manual del sistema.")
    except Exception as e:
        print(f"\n[!] Fallo crítico en inferencia: {e}")

if __name__ == "__main__":
    asyncio.run(iniciar_motores())
