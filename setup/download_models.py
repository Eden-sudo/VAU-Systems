import os
import sys
import subprocess
import requests
from huggingface_hub import snapshot_download

# Localización dinámica del proyecto
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(BASE_DIR, "models")

def setup_env():
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
        print(f"[+] Carpeta de modelos creada: {MODELS_DIR}")

def download_file(url, filename):
    """Descarga genérica directa para forzar archivos en la carpeta models"""
    target = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(target):
        print(f"[run] Descargando {filename}...")
        try:
            response = requests.get(url, stream=True, headers={'User-Agent': 'Mozilla/5.0'})
            response.raise_for_status()
            with open(target, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"[+] Guardado en: {target}")
        except Exception as e:
            print(f"[!] Error descargando {filename}: {e}")
    else:
        print(f"[skip] {filename} ya existe.")

def clone_repo(repo_url, folder_name):
    """Clona un repositorio de GitHub si no existe."""
    target_path = os.path.join(MODELS_DIR, folder_name)
    if not os.path.exists(target_path):
        print(f"[run] Clonando código fuente de {folder_name}...")
        subprocess.run(["git", "clone", repo_url, target_path], check=True)
        print(f"[+] Repositorio {folder_name} clonado exitosamente.")
    else:
        print(f"[skip] El código fuente de {folder_name} ya existe.")

def get_models():
    # 1. Qwen2-VL-2B-Instruct
    print("\n--- Model Semántico (Qwen2-VL) ---")
    snapshot_download(
        repo_id="Qwen/Qwen2-VL-2B-Instruct",
        local_dir=os.path.join(MODELS_DIR, "Qwen2-VL-2B-Instruct"),
        local_dir_use_symlinks=False,
        ignore_patterns=["*.msgpack", "*.h5", "*.ot"]
    )

    # 2. YOLOv8m-World (Descarga manual blindada, adiós a los fallos de la librería)
    print("\n--- Model de Detección (YOLO-World v8) ---")
    yolo_url = "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m-worldv2.pt"
    download_file(yolo_url, "yolov8m-worldv2.pt")

    # 3. Grounding DINO
    print("\n--- Model de Precisión (Grounding DINO) ---")
    dino_weights = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
    download_file(dino_weights, "groundingdino_swint_ogc.pth")
    dino_config = "https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    download_file(dino_config, "GroundingDINO_SwinT_OGC.py")

    # 4. STVG: TubeDETR (Arquitectura + Pesos)
    print("\n--- Model de Comportamiento Temporal (TubeDETR) ---")
    # A. Clonar la arquitectura (El código Python)
    clone_repo("https://github.com/jialianwu/TubeDETR.git", "TubeDETR_source")
    # B. Descargar los pesos (Los números .pth)
    tubedetr_weights = "https://huggingface.co/jialianwu/TubeDETR/resolve/main/tubedetr_resnet50.pth"
    download_file(tubedetr_weights, "tubedetr_resnet50.pth")

if __name__ == "__main__":
    setup_env()
    get_models()
    print("\n" + "="*40)
    print(" Instalación completada con éxito.")
    print("="*40)
