import subprocess
import sys
import time
import os

def fiumba_launch():
    # Carpeta para esconder los logs gigantes del modelo
    os.makedirs("logs", exist_ok=True)
    log_file = open("logs/model_engine.log", "w")

    print("==================================================")
    print(" 🚀 VAU-SYSTEMS ORCHESTRATOR INICIADO 🚀")
    print("==================================================")

    # 1. Lanzar el motor pesado (Juez + Sabueso)
    print("[1/3] Despertando IA (Logs redirigidos a logs/model_engine.log)...")
    motor_process = subprocess.Popen(
        [sys.executable, "launch_api_models.py"], 
        stdout=log_file, 
        stderr=subprocess.STDOUT
    )

    # Damos 5 segundos para que la Jetson reserve la RAM y abra ZeroMQ
    time.sleep(5)

    # 2. Lanzar el Servidor FastAPI (Visible en terminal)
    print("[2/3] Levantando Servidor FastAPI...")
    server_process = subprocess.Popen([sys.executable, "launch_api_servidor.py"])

    # 3. Lanzar Cloudflare (Para darte el Link público)
    print("[3/3] Cavando túnel seguro con Cloudflare...")
    cf_process = subprocess.Popen(["cloudflared", "tunnel", "--url", "http://localhost:8000"])

    try:
        # Mantener todo vivo
        server_process.wait()
    except KeyboardInterrupt:
        print("\n[!] Señal de apagado detectada. Cerrando VAU-Systems...")
        motor_process.terminate()
        server_process.terminate()
        cf_process.terminate()
        log_file.close()
        print("[!] Apagado completo. Fiumba out.")

if __name__ == "__main__":
    fiumba_launch()
