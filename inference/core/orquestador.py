import asyncio
import time
import cv2
import os

from core.memoria_hal import GestorMemoriaHAL
from utils.visualizador import HUDCognitivo

class OrquestadorVAU:
    def __init__(self, origen_video="datasets/videoplayback.mp4"):
        # Resolucion dinamica absoluta: Fuerza la busqueda desde la raiz del proyecto
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        self.origen_video = os.path.join(base_dir, origen_video)
        
        self.hal = GestorMemoriaHAL(max_frames=60)
        self.hud = HUDCognitivo()
        
        self.cola_clips_sospechosos = asyncio.Queue(maxsize=3)
        
        # Diccionario compartido para que el Juez y el HUD se comuniquen
        # Formato: { track_id: {"texto": "ESTADO", "color": (B, G, R)} }
        self.memoria_eventos = {}

        self.nodo_percepcion = None
        self.nodo_cognicion = None
        self.sistema_activo = True
        
        # Puerto de salida IPC (ZeroMQ). Será inyectado desde launch_api_models.py
        self.socket_salida = None

    # --- NUEVA FUNCIÓN: PARCHE DE CONTEXTO ---
    def _expandir_caja(self, x1, y1, x2, y2, frame_ancho, frame_alto, padding=0.4):
        """Expande la caja delimitadora un porcentaje para dar contexto periférico a Qwen."""
        ancho = x2 - x1
        alto = y2 - y1
        
        pad_x = int(ancho * padding)
        pad_y = int(alto * padding)
        
        nx1 = max(0, x1 - pad_x)
        ny1 = max(0, y1 - pad_y)
        nx2 = min(frame_ancho, x2 + pad_x)
        ny2 = min(frame_alto, y2 + pad_y)
        
        return [nx1, ny1, nx2, ny2]
    # -----------------------------------------

    async def _trabajador_camara_y_percepcion(self):
        print("[ORQUESTADOR] Levantando hilo de Percepcion (Sabueso)...")
        cap = cv2.VideoCapture(self.origen_video)
        
        if not cap.isOpened():
            print(f"[!] Error: No se pudo abrir {self.origen_video}")
            self.sistema_activo = False
            return

        # Ajuste de FPS para que corra a velocidad humana (aprox 30 FPS)
        tiempo_por_frame = 1.0 / 30.0

        # Obtenemos dimensiones del video para no salirnos de los bordes al expandir cajas
        frame_ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_alto = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        while self.sistema_activo and cap.isOpened():
            t_inicio_ciclo = time.time()
            
            ret, frame = await asyncio.to_thread(cap.read)
            if not ret:
                print("[ORQUESTADOR] Fin del flujo de video.")
                break
                
            self.hal.agregar_frame(frame)
            
            # 1. Inferencia del Sabueso (Retorna multiples pistas con ByteTrack)
            pistas_activas = await asyncio.to_thread(self.nodo_percepcion.detectar, frame)
            
            # 2. Logica de Enrutamiento
            if pistas_activas:
                # Tomamos el primer sujeto detectado para no saturar al Juez en esta fase de prueba
                pista_principal = pistas_activas[0]
                x1, y1, x2, y2, track_id = pista_principal
                
                # Si es un sujeto nuevo, lo registramos como "Analizando" (Color Naranja)
                if track_id not in self.memoria_eventos:
                    self.memoria_eventos[track_id] = {"texto": "ANALIZANDO...", "color": (0, 165, 255)}
                    
                    # --- APLICAMOS EL PARCHE DE CONTEXTO ---
                    bbox_expandido = self._expandir_caja(x1, y1, x2, y2, frame_ancho, frame_alto, padding=0.4)
                    tensor_4d = self.hal.extraer_clip_recortado(bbox_expandido)
                    # ---------------------------------------
                    
                    if tensor_4d is not None and not self.cola_clips_sospechosos.full():
                        self.hud.exportar_tubo_debug(tensor_4d, track_id)

                        # Empaquetamos el tensor junto con su ID para que el Juez sepa a quien evalua
                        await self.cola_clips_sospechosos.put((track_id, tensor_4d))
            
            # 3. Renderizado Inteligente
            frame_display = self.hud.renderizar_analiticas(frame, pistas_activas, self.memoria_eventos)
            
            if self.hud.motor == "OPENCV_CPU":
                cv2.imshow("VAU System - HUD Cognitivo", frame_display)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.sistema_activo = False
                    break
            
            # Control de velocidad estricto
            tiempo_procesado = time.time() - t_inicio_ciclo
            tiempo_espera = max(0.001, tiempo_por_frame - tiempo_procesado)
            await asyncio.sleep(tiempo_espera)
            
        cap.release()
        cv2.destroyAllWindows()
        await self.cola_clips_sospechosos.put(None)

    async def _trabajador_cognicion(self):
        print("[ORQUESTADOR] Levantando hilo de Cognicion (Juez)...")
        
        while self.sistema_activo:
            paquete = await self.cola_clips_sospechosos.get()
            if paquete is None:
                self.cola_clips_sospechosos.task_done()
                break
                
            track_id, tensor_4d = paquete
            
            print(f"[JUEZ] Evaluando Tubo Temporal del Sujeto #{track_id}...")
            
            # Invocamos a Qwen2-VL (Caera a CPU si no hay grafica NVIDIA en la PC)
            veredicto = await asyncio.to_thread(self.nodo_cognicion.razonar, tensor_4d)
            
            # Actualizamos el estado visual en base a la respuesta de Qwen
            es_alerta = veredicto.get("estado") == "ALERTA"
            
            if es_alerta:
                self.memoria_eventos[track_id] = {"texto": veredicto.get("descripcion", "AMENAZA"), "color": (0, 0, 255)} # Rojo
            else:
                self.memoria_eventos[track_id] = {"texto": "DESPEJADO", "color": (0, 255, 0)} # Verde
                
            print(f"[JUEZ] Veredicto Sujeto #{track_id} -> {veredicto}")
            
            # --- NUEVA LÓGICA DE TRANSMISIÓN IPC (ZeroMQ) ---
            if self.socket_salida is not None:
                alerta_empaquetada = {
                    "tipo": "alerta_seguridad" if es_alerta else "telemetria",
                    "timestamp": time.time(),
                    "track_id": track_id,
                    "datos": veredicto
                }
                try:
                    await self.socket_salida.send_json(alerta_empaquetada)
                except Exception as e:
                    print(f"[!] Error inyectando datos en el puente ZeroMQ: {e}")
            # ------------------------------------------------

            self.cola_clips_sospechosos.task_done()

    async def iniciar_sistema(self):
        print("==========================================================")
        print("INICIANDO VAU SYSTEM (NUCLEO ASINCRONO ZERO-COPY)")
        print("==========================================================")
        tarea_percepcion = asyncio.create_task(self._trabajador_camara_y_percepcion())
        tarea_cognicion = asyncio.create_task(self._trabajador_cognicion())
        await asyncio.gather(tarea_percepcion, tarea_cognicion)
        print("[ORQUESTADOR] Apagado de forma segura.")
