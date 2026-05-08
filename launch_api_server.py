"""
Servidor API Gateway para VAU-Systems.
Gestiona la comunicacion entre el motor de inferencia (ZMQ) y la interfaz web (WebSockets).
"""

import os
import asyncio
import zmq
import zmq.asyncio
import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager

# Configuracion de rutas dinamicas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "api", "static")

# Buffer de eventos para distribucion asincrona
cola_eventos = asyncio.Queue()

async def escuchar_zero_mq():
    """Hilo de subscripcion para mensajes provenientes del motor de inferencia."""
    context = zmq.asyncio.Context()
    socket_sub = context.socket(zmq.SUB)
    socket_sub.connect("tcp://127.0.0.1:5555")
    socket_sub.setsockopt_string(zmq.SUBSCRIBE, "")
    
    print("[INFO] Enlace ZeroMQ activo en puerto 5555.")
    
    try:
        while True:
            paquete = await socket_sub.recv_json()
            await cola_eventos.put(paquete)
    except asyncio.CancelledError:
        pass
    finally:
        socket_sub.close()
        context.term()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manejo de inicio y cierre de servicios de fondo."""
    tarea_zmq = asyncio.create_task(escuchar_zero_mq())
    yield
    tarea_zmq.cancel()
    try:
        await tarea_zmq
    except asyncio.CancelledError:
        pass

app = FastAPI(lifespan=lifespan)

# Montaje de archivos estaticos para la interfaz de usuario
if os.path.exists(STATIC_DIR):
    app.mount("/dashboard", StaticFiles(directory=STATIC_DIR, html=True), name="static")
else:
    print(f"[ERROR] No se encontro el directorio estatico en: {STATIC_DIR}")

@app.websocket("/ws/ui")
async def ui_websocket(websocket: WebSocket):
    """Punto de enlace para transmision de datos en tiempo real a la UI."""
    await websocket.accept()
    print("[INFO] Conexion WebSocket establecida con el cliente.")
    try:
        while True:
            paquete = await cola_eventos.get()
            await websocket.send_json(paquete)
    except Exception:
        print("[INFO] Conexion WebSocket cerrada.")

if __name__ == "__main__":
    # Resolucion dinamica del nombre de modulo para soporte de hot-reload
    nombre_modulo = os.path.splitext(os.path.basename(__file__))[0]
    
    print(f"[INFO] Servidor disponible en http://localhost:8000/dashboard")
    uvicorn.run(f"{nombre_modulo}:app", host="0.0.0.0", port=8000, reload=True)
