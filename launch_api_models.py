"""
Punto de entrada para el subsistema de inferencia VAU-Systems.
Gestiona la ejecucion de modelos y publicacion de resultados via ZeroMQ.
"""

import os
import sys
import asyncio
import zmq
import zmq.asyncio

# Configuracion de directorios base
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INFERENCE_DIR = os.path.join(BASE_DIR, "inference")

# Inyeccion de ruta para resolucion de modulos internos
if INFERENCE_DIR not in sys.path:
    sys.path.append(INFERENCE_DIR)

from core.orquestador import OrquestadorVAU
from nodes.percepcion import MotorPercepcion
from nodes.cognicion import MotorCognicion

async def iniciar_motores():
    # Inicializacion de contexto y socket ZeroMQ (Publisher)
    context = zmq.asyncio.Context()
    socket_pub = context.socket(zmq.PUB)
    socket_pub.bind("tcp://127.0.0.1:5555")

    # Definicion de origen de datos y orquestador
    video_path = os.path.join(BASE_DIR, "datasets", "videoplayback.mp4")
    motor_central = OrquestadorVAU(origen_video=video_path)
    
    # Inyeccion de dependencias de hardware e IA
    motor_central.nodo_percepcion = MotorPercepcion()
    motor_central.nodo_cognicion = MotorCognicion()
    motor_central.socket_salida = socket_pub

    try:
        print("[INFO] Motor de inferencia iniciado.")
        await motor_central.iniciar_sistema()
    except KeyboardInterrupt:
        print("[INFO] Detencion solicitada por el usuario.")
    except Exception as e:
        print(f"[ERROR] Fallo critico en inferencia: {e}")
    finally:
        # Limpieza de recursos de red y memoria
        socket_pub.close()
        context.term()
        print("[INFO] Recursos liberados. Sistema offline.")

if __name__ == "__main__":
    asyncio.run(iniciar_motores())
