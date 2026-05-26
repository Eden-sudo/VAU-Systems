"""
Servidor API Gateway para VAU-Systems con Depuración de ZMQ y Heartbeats.
"""
import os
import asyncio
import zmq
import zmq.asyncio
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DIST_DIR = os.path.join(BASE_DIR, "api", "dist")

zmq_context = zmq.asyncio.Context()
latest_frame = b""

# --- GESTOR DE WEBSOCKETS ---
class GestorConexiones:
    def __init__(self):
        self.conexiones_activas: list[WebSocket] = []

    async def conectar(self, websocket: WebSocket):
        await websocket.accept()
        self.conexiones_activas.append(websocket)
        print(f"[API] 🟢 Cliente Web Vue conectado. Total: {len(self.conexiones_activas)}")

    def desconectar(self, websocket: WebSocket):
        if websocket in self.conexiones_activas:
            self.conexiones_activas.remove(websocket)
            print(f"[API] 🔴 Cliente Web Vue desconectado. Total: {len(self.conexiones_activas)}")

    async def emitir_telemetria(self, paquete: dict):
        for conexion in self.conexiones_activas.copy():
            try:
                await conexion.send_json(paquete)
            except Exception:
                self.desconectar(conexion)

gestor_ws = GestorConexiones()

# --- HILOS DE LECTURA ZEROMQ ---
async def escuchar_telemetria_zmq():
    socket_json = zmq_context.socket(zmq.SUB)
    socket_json.connect("tcp://127.0.0.1:5555")
    socket_json.setsockopt_string(zmq.SUBSCRIBE, "")
    print("[API] Hilo de Telemetría (5555) esperando a la IA...")
    try:
        while True:
            # wait_for nos devuelve el control si la IA tarda más de 1 segundo
            paquete = await asyncio.wait_for(socket_json.recv_json(), timeout=1.0)
            print(f"[ZMQ] ⚡ JSON recibido de la IA. Emitiendo a Vue...")
            await gestor_ws.emitir_telemetria(paquete)
            
    except asyncio.TimeoutError:
        # HEARTBEAT: Si la IA está callada, le decimos a Vue que no se desconecte
        if len(gestor_ws.conexiones_activas) > 0:
            await gestor_ws.emitir_telemetria({"tipo": "heartbeat", "estado": "Esperando visión..."})
        # Seguimos iterando
        asyncio.create_task(escuchar_telemetria_zmq_loop(socket_json)) # fallback para evitar cortar el hilo

async def escuchar_telemetria_zmq_loop(socket_json):
    while True:
        try:
            paquete = await asyncio.wait_for(socket_json.recv_json(), timeout=1.0)
            print(f"[ZMQ] ⚡ JSON recibido de la IA. Emitiendo a Vue...")
            await gestor_ws.emitir_telemetria(paquete)
        except asyncio.TimeoutError:
            if len(gestor_ws.conexiones_activas) > 0:
                await gestor_ws.emitir_telemetria({"tipo": "heartbeat", "estado": "Esperando visión..."})

async def escuchar_video_zmq():
    global latest_frame
    contador_frames = 0
    socket_video = zmq_context.socket(zmq.SUB)
    socket_video.setsockopt(zmq.CONFLATE, 1) 
    socket_video.connect("tcp://127.0.0.1:5556")
    socket_video.setsockopt_string(zmq.SUBSCRIBE, "")
    print("[API] Hilo de Video (5556) esperando a la IA...")
    
    while True:
        try:
            latest_frame = await asyncio.wait_for(socket_video.recv(), timeout=1.0)
            contador_frames += 1
            if contador_frames % 60 == 0: # Imprime 1 vez cada par de segundos
                print(f"[ZMQ] 🎥 Video fluyendo (Frame recibido, {len(latest_frame)} bytes)")
        except asyncio.TimeoutError:
            pass

# --- LIFESPAN ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    t1 = asyncio.create_task(escuchar_telemetria_zmq())
    t2 = asyncio.create_task(escuchar_video_zmq())
    yield
    t1.cancel()
    t2.cancel()
    zmq_context.term()

app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.websocket("/ws/telemetria")
async def ui_websocket(websocket: WebSocket):
    await gestor_ws.conectar(websocket)
    try:
        while True:
            await websocket.receive_text() 
    except WebSocketDisconnect:
        gestor_ws.desconectar(websocket)

@app.get("/video_feed")
async def video_feed():
    async def generate():
        global latest_frame
        while True:
            if latest_frame:
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + latest_frame + b'\r\n')
            await asyncio.sleep(0.03) 
    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")

if os.path.exists(DIST_DIR):
    app.mount("/", StaticFiles(directory=DIST_DIR, html=True), name="frontend")

if __name__ == "__main__":
    nombre_modulo = os.path.splitext(os.path.basename(__file__))[0]
    uvicorn.run(f"{nombre_modulo}:app", host="0.0.0.0", port=8000, reload=False)
