"""
yolo.py
Script para el entrenamiento y ajuste fino de modelos YOLO-World.
Configurado para ejecucion en hardware con soporte CUDA y Tensor Cores.
"""

import os
from ultralytics import YOLO

# Resolucion dinamica de rutas
# Localiza la raiz del proyecto VAU-Systems partiendo de la ubicacion del script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)

# Definicion de rutas de recursos y configuracion
# Se asume la existencia de la subcarpeta 'vision' dentro de 'models'
RUTA_MODELO = os.path.join(BASE_DIR, "models", "vision", "yolov8m-worldv2.pt")
RUTA_YAML = os.path.join(BASE_DIR, "yolo_dataset", "dataset.yaml")
RUTA_SALIDA = os.path.join(BASE_DIR, "models")

def iniciar_entrenamiento():
    """Configura e inicia el proceso de entrenamiento del modelo de vision."""
    print("[INFO] Inicializando entorno de entrenamiento YOLO.")

    # Carga de arquitectura base
    try:
        model = YOLO(RUTA_MODELO)
    except Exception as e:
        print(f"[ERROR] No se pudo cargar el modelo base en {RUTA_MODELO}: {e}")
        return

    # Parametros de entrenamiento optimizados para arquitectura NVIDIA RTX
    # Se utiliza precision mixta (AMP) para aprovechar los Tensor Cores
    model.train(
        data=RUTA_YAML,
        epochs=50,                # Ciclos completos sobre el dataset
        batch=16,                # Ajustar segun disponibilidad de VRAM (rango 16-32)
        imgsz=640,               # Resolucion CCTV estandar
        device=0,                # ID del dispositivo CUDA primario
        amp=True,                # Mixed precision (FP16) habilitada
        workers=8,               # Hilos de carga de datos en CPU
        project=RUTA_SALIDA,     # Directorio de exportacion
        name="YOLO-VAU-Fighter"  # Identificador de la version del modelo
    )

    print(f"[INFO] Entrenamiento finalizado. Modelo exportado a: {RUTA_SALIDA}/YOLO-VAU-Fighter")

if __name__ == "__main__":
    iniciar_entrenamiento()
