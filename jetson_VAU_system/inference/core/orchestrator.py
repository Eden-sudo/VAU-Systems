import asyncio
import time
import cv2
import zmq
import zmq.asyncio
import glob
import os

class OrchestratorVAU:
    def __init__(self, config, node_perception, node_cognition, hal):
        print("[ORCHESTRATOR] Inicializando Nucleo de Enrutamiento Asincrono...")
        self.config = config
        
        # Extraemos variables desde el diccionario de configuracion
        camera_conf = config.get("camera", {})
        self.video_source = camera_conf.get("source", 0) 
        self.target_fps = camera_conf.get("fps", 30)
        
        self.hal = hal
        self.node_perception = node_perception
        self.node_cognition = node_cognition
        
        # Cola de tubos 4D limitados para no ahogar al VLM
        self.suspicious_clips_queue = asyncio.Queue(maxsize=3)
        self.event_memory = {}
        self.system_active = True
        
        # --- INFRAESTRUCTURA ZEROMQ (PUBLISHER) ---
        self.zmq_context = zmq.asyncio.Context()
        
        self.socket_json = self.zmq_context.socket(zmq.PUB)
        self.socket_json.bind("tcp://*:5555")
        
        self.socket_video = self.zmq_context.socket(zmq.PUB)
        self.socket_video.bind("tcp://*:5556")
        
        print("[ORCHESTRATOR] Puertos IPC ZeroMQ expuestos [5555: JSON | 5556: VIDEO]")

    def _expand_box(self, x1, y1, x2, y2, f_width, f_height, padding=0.4):
        """Expande la caja delimitadora para dar contexto periferico a Qwen."""
        ancho = x2 - x1
        alto = y2 - y1
        
        pad_x = int(ancho * padding)
        pad_y = int(alto * padding)
        
        nx1 = max(0, x1 - pad_x)
        ny1 = max(0, y1 - pad_y)
        nx2 = min(f_width, x2 + pad_x)
        ny2 = min(f_height, y2 + pad_y)
        
        return [nx1, ny1, nx2, ny2]

    async def _camera_and_perception_worker(self):
        print("[ORCHESTRATOR] Hilo de Percepcion (Sabueso) en linea.")

        # Detectar si la fuente es un directorio de pruebas o un archivo unico
        if isinstance(self.video_source, str) and os.path.isdir(self.video_source):
            lista_videos = sorted(glob.glob(os.path.join(self.video_source, "*.mp4")))
            print(f"[ORCHESTRATOR] Directorio detectado. Se procesaran {len(lista_videos)} videos secuencialmente.")
        else:
            lista_videos = [self.video_source]

        # Bucle maestro de reproduccion (Playlist)
        for video in lista_videos:
            if not self.system_active:
                break
                
            print(f"\n[ORCHESTRATOR] Reproduciendo video: {os.path.basename(str(video))}")
            cap = cv2.VideoCapture(video)
            
            if not cap.isOpened():
                print(f"[!] Advertencia: No se pudo abrir {video}, saltando al siguiente...")
                continue

            tiempo_por_frame = 1.0 / self.target_fps
            f_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            f_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            while self.system_active and cap.isOpened():
                t_inicio = time.time()
                
                ret, frame = await asyncio.to_thread(cap.read)
                if not ret:
                    print(f"[ORCHESTRATOR] Fin del video: {os.path.basename(str(video))}")
                    break
                    
                self.hal.agregar_frame(frame)
                
                # 1. Inferencia de Percepcion (YOLO-World + ByteTrack)
                pistas_activas = await asyncio.to_thread(self.node_perception.detectar, frame)
                
                if pistas_activas:
                    ids_rastreados = [p[4] for p in pistas_activas]
                    print(f"[YOLO] Detectado movimiento. IDs activos: {ids_rastreados}")
                
                # --- GARBAGE COLLECTION: LIMPIEZA ESTRICTA DE MEMORIA RAM ---
                ids_presentes = [p[4] for p in pistas_activas] if pistas_activas else []
                claves_a_borrar = [tid for tid in self.event_memory.keys() if tid not in ids_presentes]
                for tid in claves_a_borrar:
                    del self.event_memory[tid]
                # -------------------------------------------------------------

                # 2. Logica de Enrutamiento hacia Cognicion
                if pistas_activas:
                    pista_principal = pistas_activas[0]
                    x1, y1, x2, y2, track_id, clase_yolo = pista_principal
                    
                    if track_id not in self.event_memory:
                        self.event_memory[track_id] = {"estado": "ANALIZANDO...", "riesgo": 0}
                        
                        bbox_expandido = self._expand_box(x1, y1, x2, y2, f_width, f_height)
                        tensor_4d = self.hal.extraer_clip_recortado(bbox_expandido)
                        
                        if tensor_4d is not None and not self.suspicious_clips_queue.full():
                            # Enviamos track_id, el tensor de video y la etiqueta de YOLO al Juez
                            await self.suspicious_clips_queue.put((track_id, tensor_4d, clase_yolo))
                    
                    # Dibujado basico de HUD de seguridad para la transmision web
                    color = (0, 165, 255) if self.event_memory[track_id]["riesgo"] == 0 else (
                            (0, 0, 255) if self.event_memory[track_id]["estado"] in ["ALERTA", "CRITICO"] else (0, 255, 0))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"ID:{track_id} [{clase_yolo}] {self.event_memory[track_id]['estado']}",  
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # 3. Transmision de Video MJPEG por Red a FastAPI
                _, buffer_img = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                try:
                    await self.socket_video.send(buffer_img.tobytes())
                except Exception:
                    pass 
                
                tiempo_proc = time.time() - t_inicio
                await asyncio.sleep(max(0.001, tiempo_por_frame - tiempo_proc))
                
            cap.release()
            
        print("[ORCHESTRATOR] Playlist finalizada. Todos los videos procesados.")
        await self.suspicious_clips_queue.put(None) 

    async def _cognition_worker(self):
        print("[ORCHESTRATOR] Hilo de Cognicion (Juez) en linea.")
        
        while self.system_active:
            paquete = await self.suspicious_clips_queue.get()
            if paquete is None:
                self.suspicious_clips_queue.task_done()
                break
                
            # CORREGIDO: Ahora desempaqueta las 3 variables de forma segura
            track_id, tensor_4d, clase_yolo = paquete
            
            # 1. Inferencia del VLM pasando la pista contextual de YOLO
            veredicto = await asyncio.to_thread(self.node_cognition.razonar, tensor_4d, clase_yolo)
            
            # Sincronizacion de alertas con la nueva escala del prompt
            es_alerta = veredicto.get("estado") in ["ALERTA", "CRITICO"]
            
            # DEPURACION LIMPIA DE COGNICION
            print(f"\n[QWEN-VL] Veredicto emitido para ID {track_id}:")
            print(f"   -> Riesgo: {veredicto.get('riesgo')}/10")
            print(f"   -> Estado: {veredicto.get('estado')}")
            print(f"   -> Justificacion: {veredicto.get('descripcion', 'Sin descripcion')}\n")
            
            # 2. Actualizacion de estado en tiempo real en la RAM
            self.event_memory[track_id] = {
                "estado": veredicto.get("estado", "DESCONOCIDO"),
                "riesgo": veredicto.get("riesgo", 0)
            }
            
            # 3. Transmision de Telemetria JSON consolidada a la API Web
            alerta_empaquetada = {
                "tipo": "alerta_seguridad" if es_alerta else "telemetria",
                "timestamp": time.time(),
                "track_id": track_id,
                "datos": veredicto
            }
            
            try:
                await self.socket_json.send_json(alerta_empaquetada)
            except Exception as e:
                print(f"[!] Error inyectando JSON en ZeroMQ: {e}")

            self.suspicious_clips_queue.task_done()

    async def _telemetry_worker(self):
        print("[ORCHESTRATOR] Hilo de Telemetria de Hardware en linea.")
        import psutil

        while self.system_active:
            try:
                # 1. Leer Memoria RAM Unificada real
                ram = psutil.virtual_memory()
                ram_uso = f"{(ram.used / (1024**3)):.1f}GB"
                
                # 2. Leer Temperatura Real de la Jetson (Sensor Térmico 0)
                temp = 0
                try:
                    with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
                        temp = int(f.read()) / 1000.0
                except:
                    temp = 45 # Valor de respaldo por si el archivo está bloqueado

                # 3. CPU / GPU Load
                carga = psutil.cpu_percent(interval=None)

                # 4. Empaquetar y enviar por ZeroMQ
                paquete_hw = {
                    "tipo": "hardware_stats",
                    "datos": {
                        "fps": self.target_fps,
                        "gpu_util": int(carga),
                        "temp_gpu": int(temp),
                        "ram_uso": ram_uso
                    }
                }
                
                await self.socket_json.send_json(paquete_hw)
                
            except Exception as e:
                pass
                
            # Enviar actualización cada 2 segundos para no saturar el WebSocket
            await asyncio.sleep(2)

    async def start(self):
        print("==========================================================")
        print(" VAU SYSTEM NODE INICIADO (IPC ROUTING ACTIVO) ")
        print("==========================================================")
        task_perception = asyncio.create_task(self._camera_and_perception_worker())
        task_cognition = asyncio.create_task(self._cognition_worker())
        task_telemetria = asyncio.create_task(self._telemetry_worker()) # <--- NUEVO HILO AÑADIDO
        
        # Juntamos los 3 hilos
        await asyncio.gather(task_perception, task_cognition, task_telemetria)
