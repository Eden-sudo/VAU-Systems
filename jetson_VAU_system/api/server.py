import os
import asyncio
import zmq
import zmq.asyncio
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="VAU Systems API - Puente ZeroMQ")

# --- POLÍTICA CORS ---
# Permite que el frontend (Vue) consulte la API sin bloqueos de seguridad del navegador
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # En producción estricta, cambia "*" por la IP de tu Jetson
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- GESTIÓN DE CONEXIONES WEBSOCKET ---
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"[API] Cliente UI Conectado. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        print(f"[API] Cliente UI Desconectado. Total: {len(self.active_connections)}")

manager = ConnectionManager()

# --- PUENTE ZEROMQ (Motor de IA -> FastAPI) ---
zmq_context = zmq.asyncio.Context()

# Socket para Telemetría (JSON) - Se conecta al PUB del Orchestrator
sub_json = zmq_context.socket(zmq.SUB)
sub_json.connect("tcp://127.0.0.1:5555")
sub_json.setsockopt_string(zmq.SUBSCRIBE, "")

# Socket para Video (Bytes) - Se conecta al PUB del Orchestrator
sub_video = zmq_context.socket(zmq.SUB)
sub_video.connect("tcp://127.0.0.1:5556")
sub_video.setsockopt_string(zmq.SUBSCRIBE, "")

# --- ENDPOINTS ---
@app.get("/video_feed")
async def video_feed():
    """Ruta para la etiqueta <img src='/video_feed'> del frontend Vue"""
    async def generate():
        while True:
            frame_bytes = await sub_video.recv()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.websocket("/ws/telemetria")
async def websocket_endpoint(websocket: WebSocket):
    """Canal bidireccional de datos en tiempo real para Vue"""
    await manager.connect(websocket)
    try:
        while True:
            data = await sub_json.recv_json()
            await websocket.send_json(data)
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"[!] Error en WebSocket: {e}")
        manager.disconnect(websocket)

# --- MONITOREO DE HARDWARE JETSON ---
jetson_stats = {}

async def monitor_jetson():
    """Lee sensores de la Jetson usando jtop de forma segura"""
    try:
        from jtop import jtop
        with jtop() as jetson:
            while jetson.ok():
                global jetson_stats
                jetson_stats = {
                    "gpu_util": jetson.gpu['val'],
                    "cpu_util": jetson.cpu['total']['user'],
                    "temp_gpu": jetson.temperature['GPU'],
                    "ram_uso": f"{jetson.memory['RAM']['used'] / (1024**3):.1f}GB / {(jetson.memory['RAM']['tot'] / (1024**3)):.1f}GB"
                }
                # Podemos emitir esto por el WS también si lo deseas en el futuro
                await asyncio.sleep(1.0)
    except ImportError:
        print("[!] Advertencia: Librería 'jtop' no encontrada. Monitoreo de hardware desactivado.")
    except Exception as e:
        print(f"[!] Error leyendo sensores Jetson: {e}")

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(monitor_jetson())

# --- SERVIR VUE.JS (FRONTEND) ---
# Resolución de ruta absoluta para compatibilidad con systemctl
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DIST_DIR = os.path.join(BASE_DIR, "dist")

if os.path.exists(DIST_DIR):
    print(f"[API] Carpeta frontend encontrada en {DIST_DIR}. Sirviendo UI Web...")
    app.mount("/", StaticFiles(directory=DIST_DIR, html=True), name="static")
else:
    print(f"[!] Advertencia crítica: Carpeta 'dist' no encontrada en {DIST_DIR}.")
    print("[!] Ejecuta 'npm run build' en tu proyecto Vue y copia la carpeta aquí.")

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
