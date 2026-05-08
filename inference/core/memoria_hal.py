import platform
import collections
import numpy as np
import torch
import cv2

class GestorMemoriaHAL:
    def __init__(self, max_frames=60, camera_width=1920, camera_height=1080):
        self.arch = platform.machine()
        self.is_jetson = 'aarch64' in self.arch.lower()
        self.engine = "DESCONOCIDO"
        
        self.max_frames = max_frames
        self.camera_width = camera_width
        self.camera_height = camera_height

        print("[HAL] Inicializando Capa de Abstracción de Hardware...")

        if self.is_jetson:
            try:
                import jetson_utils
                self.jetson_utils = jetson_utils
                self.engine = "CUDA_ZERO_COPY"
                print("[HAL] Jetson detectada. Configurando memoria unificada VRAM.")
                self._init_jetson_buffer()
            except ImportError:
                print("[!] Advertencia: jetson_utils no encontrado. Cayendo a simulación CPU.")
                self._fallback_to_cpu()
        else:
            self._fallback_to_cpu()

    def _fallback_to_cpu(self):
        self.engine = "CPU_NUMPY"
        print(f"[HAL] PC/Laptop detectada. Usando Búfer Deque en RAM estándar.")
        self.cpu_buffer = collections.deque(maxlen=self.max_frames)

    def _init_jetson_buffer(self, format='rgb8'):
        """Preasigna la memoria física del Ring Buffer en la GPU al arrancar el sistema"""
        print(f"[HAL] Preasignando {self.max_frames} frames en memoria Zero-Copy...")
        self.cuda_buffer = [
            self.jetson_utils.cudaAllocMapped(width=self.camera_width, height=self.camera_height, format=format)
            for _ in range(self.max_frames)
        ]
        self.ring_index = 0
        self.frames_actuales = 0

    def agregar_frame(self, frame_data):
        """Sobrescribe la memoria física continuamente sin generar basura (Garbage Collection)"""
        if self.engine == "CPU_NUMPY":
            self.cpu_buffer.append(frame_data)
        elif self.engine == "CUDA_ZERO_COPY":
            # Extraemos el puntero del búfer preasignado que toca en el turno
            frame_destino = self.cuda_buffer[self.ring_index]
            
            # Copiamos a bajo nivel los bytes entrantes al búfer seguro
            self.jetson_utils.cudaMemcpy(frame_destino, frame_data)
            
            self.ring_index = (self.ring_index + 1) % self.max_frames
            self.frames_actuales = min(self.frames_actuales + 1, self.max_frames)

    def extraer_clip_recortado(self, bbox):
        """Genera el Tubo 4D garantizando la geometria y color correctos"""
        print("[HAL] Extrayendo tubo de acción del búfer...")
        x1, y1, x2, y2 = [int(v) for v in bbox]

        if self.engine == "CPU_NUMPY":
            clip_recortado = []
            for frame in self.cpu_buffer:
                # 1. Protección de límites (si YOLO detecta algo en el borde de la pantalla)
                h_frame, w_frame = frame.shape[:2]
                x1_s, y1_s = max(0, x1), max(0, y1)
                x2_s, y2_s = min(w_frame, x2), min(h_frame, y2)
                
                recorte = frame[y1_s:y2_s, x1_s:x2_s]
                if recorte.size == 0: continue
                
                # 2. FIX: Escala real interpolada preservando la geometría
                recorte_resized = cv2.resize(recorte, (224, 224))
                
                # 3. FIX: Conversión de paleta de colores BGR (Cámara) a RGB (LLM)
                recorte_rgb = cv2.cvtColor(recorte_resized, cv2.COLOR_BGR2RGB)
                
                clip_recortado.append(recorte_rgb)
                
            if not clip_recortado: return None
            tensor_video = torch.tensor(np.array(clip_recortado)).permute(0, 3, 1, 2)
            return tensor_video.float()

        elif self.engine == "CUDA_ZERO_COPY":
            if self.frames_actuales == 0: return None
            tensores_recortados = []
            for i in range(self.frames_actuales):
                idx = (self.ring_index - self.frames_actuales + i) % self.max_frames
                frame_origen = self.cuda_buffer[idx]
                tensor_gpu = torch.as_tensor(frame_origen, device='cuda').permute(2, 0, 1)
                tensor_recorte = tensor_gpu[:, y1:y2, x1:x2]
                tensores_recortados.append(tensor_recorte)
                
            tensor_video = torch.stack(tensores_recortados)
            return tensor_video.float()
