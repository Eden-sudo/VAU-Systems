import asyncio
from core.orquestador import OrquestadorVAU
from nodes.percepcion import MotorPercepcion
from nodes.cognicion import MotorCognicion

def desplegar_sistema():
    print("[INIT] Conectando nodos al orquestador principal...")
    motor_central = OrquestadorVAU(origen_video="datasets/videoplayback.mp4")
    
    # Inyectamos las dependencias
    motor_central.nodo_percepcion = MotorPercepcion()
    motor_central.nodo_cognicion = MotorCognicion()
    
    # Encendemos la maquina
    asyncio.run(motor_central.iniciar_sistema())

if __name__ == "__main__":
    desplegar_sistema()
