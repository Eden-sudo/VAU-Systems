"""
Script para la generacion de datasets multimodales en formato JSONL.
Prepara la estructura de entrenamiento para el modelo Qwen2-VL.
"""

import os
import json
import mimetypes
from pathlib import Path

# Configuracion dinamica de rutas
# Se asume que el script reside en 'training/' y 'datasets/' en la raiz
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
CARPETA_DATASETS = BASE_DIR / "datasets"
ARCHIVO_SALIDA = CARPETA_DATASETS / "dataset_seguridad_vau.jsonl"

# Escaneo de archivos en directorios especificos
archivos_anomalos_raw = (CARPETA_DATASETS / "anomalias").rglob("*")
archivos_normales_raw = (CARPETA_DATASETS / "normales").rglob("*")

# Filtrado para asegurar solo archivos validos
ARCHIVOS_ANOMALOS = [v for v in archivos_anomalos_raw if v.is_file()]
ARCHIVOS_NORMALES = [v for v in archivos_normales_raw if v.is_file()]

PROMPT_SISTEMA = (
    "Eres un analista de seguridad experto en circuito cerrado. "
    "Tu tarea es analizar la evidencia visual proporcionada y detectar comportamientos anomalos. "
    "Sigue esta escala de riesgo estrictamente:\n"
    "- Riesgo 1-3: Actividades cotidianas.\n"
    "- Riesgo 4-6: Comportamiento sospechoso.\n"
    "- Riesgo 7-10: Amenaza inminente.\n\n"
    "REGLA CRITICA: Escribe tu razonamiento de la situacion, seguido ESTRICTAMENTE por un objeto JSON valido: "
    '{"descripcion": "...", "riesgo": [1-10], "estado": "NORMAL" o "ALERTA"}'
)

def detectar_tipo(ruta):
    """
    Identifica si el archivo es video o imagen mediante MIME types.
    Incluye fallback para extensiones comunes si falla la deteccion.
    """
    tipo_mime, _ = mimetypes.guess_type(str(ruta))
    
    if tipo_mime:
        if tipo_mime.startswith('video'):
            return "video"
        elif tipo_mime.startswith('image'):
            return "image"
            
    # Fallback por extension para archivos sin metadata MIME
    extensiones_video = ('.avi', '.mp4', '.mov', '.mkv')
    if str(ruta).lower().endswith(extensiones_video):
        return "video"
    return "image"

def crear_registro(ruta_archivo, es_anomalia):
    """Estructura un ejemplo de entrenamiento siguiendo el esquema de mensajes de Qwen."""
    tipo_media = detectar_tipo(ruta_archivo)
    
    if es_anomalia:
        respuesta_ideal = (
            "Analisis: Se detecta un altercado fisico violento o acto delictivo en progreso. "
            "Hay posturas agresivas evidentes que requieren intervencion inmediata.\n"
            '{"descripcion": "Altercado fisico en progreso", "riesgo": 9, "estado": "ALERTA"}'
        )
    else:
        respuesta_ideal = (
            "Analisis: Las personas en la zona transitan o interactuan de forma pacifica. "
            "No se detectan movimientos erraticos, agresiones ni armas.\n"
            '{"descripcion": "Actividad cotidiana pacifica", "riesgo": 1, "estado": "NORMAL"}'
        )

    # El contenido multimedia varia su llave segun el tipo detectado
    media_content = {
        "type": tipo_media,
        tipo_media: str(ruta_archivo)
    }

    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    media_content,
                    {"type": "text", "text": PROMPT_SISTEMA}
                ]
            },
            {
                "role": "assistant",
                "content": respuesta_ideal
            }
        ]
    }

def compilar_dataset():
    """Genera el archivo JSONL final iterando sobre los archivos encontrados."""
    # Asegurar que el directorio de salida existe
    CARPETA_DATASETS.mkdir(parents=True, exist_ok=True)
    
    registros_totales = 0
    with open(ARCHIVO_SALIDA, 'w', encoding='utf-8') as f:
        # Procesamiento de casos positivos (anomalias)
        for archivo in ARCHIVOS_ANOMALOS:
            f.write(json.dumps(crear_registro(archivo, True)) + '\n')
            registros_totales += 1
            
        # Procesamiento de casos negativos (normales)
        for archivo in ARCHIVOS_NORMALES:
            f.write(json.dumps(crear_registro(archivo, False)) + '\n')
            registros_totales += 1

    print(f"[INFO] Dataset compilado. Total registros: {registros_totales}.")
    print(f"[INFO] Ruta de salida: {ARCHIVO_SALIDA}")

if __name__ == "__main__":
    mimetypes.init()
    compilar_dataset()
