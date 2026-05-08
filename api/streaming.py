from fastapi import WebSocket

class GestorConexiones:
    def __init__(self):
        # Lista de clientes conectados a la UI Web
        self.conexiones_activas: list[WebSocket] = []

    async def conectar(self, websocket: WebSocket):
        await websocket.accept()
        self.conexiones_activas.append(websocket)
        print(f"[STREAMING] Nuevo cliente conectado. Total: {len(self.conexiones_activas)}")

    def desconectar(self, websocket: WebSocket):
        self.conexiones_activas.remove(websocket)
        print(f"[STREAMING] Cliente desconectado. Total: {len(self.conexiones_activas)}")

    async def emitir_evento(self, paquete_json: dict):
        """
        Envía la alerta o telemetría a todos los navegadores abiertos.
        """
        for conexion in self.conexiones_activas:
            try:
                await conexion.send_json(paquete_json)
            except Exception as e:
                print(f"[!] Error enviando a cliente WS: {e}")
                self.desconectar(conexion)

# Instancia global para importar en main.py
gestor_ws = GestorConexiones()
