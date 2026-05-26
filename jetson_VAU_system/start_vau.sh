#!/bin/bash
echo "========================================="
echo " INICIANDO VAU-SYSTEM (CLOUDFLARE MODE) "
echo "========================================="

# 1. Activar el entorno
source ~/VAU_systems/vau_env/bin/activate

# 2. Iniciar FastAPI en segundo plano
echo "[+] Levantando Servidor Web FastAPI..."
python launch_api_servidor.py &
PID_API=$!
sleep 3 # Damos tiempo a que el puerto se abra

# 3. Iniciar Túnel Cloudflare (Usando HTTP2 para evitar cortes de red)
echo "[+] Estableciendo Túnel de Cloudflare..."
cloudflared tunnel --protocol http2 --url http://127.0.0.1:8000 &
PID_TUNNEL=$!

# 4. Iniciar Modelos de IA
echo "[+] Despertando Redes Neuronales (YOLO + Qwen)..."
python launch_api_models.py &
PID_MODELS=$!

# Función para apagar todo al presionar Ctrl+C
cleanup() {
    echo -e "\n[!] Apagando VAU-System..."
    kill $PID_MODELS $PID_API $PID_TUNNEL 2>/dev/null
    echo "Sistema fuera de línea. ¡Buen trabajo!"
    exit 0
}

# Capturar señal de interrupción
trap cleanup SIGINT SIGTERM

# Mantener el script vivo
wait
