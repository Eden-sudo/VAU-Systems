"""
Capa de Abstracción de Hardware (HAL) para Jetson Nano.
Gestiona el Ring Buffer de video en Memoria Unificada (UMA) usando NumPy puro.
"""
import numpy as np
import cv2

class GestorMemoriaHAL:
    def __init__(self, config):
        print("[HAL] Inicializando Capa de Abstracción de Hardware (NumPy UMA)...")
        
        # Extraemos la configuración dinámica del settings.yaml
        camera_conf = config.get("camera", {})
        self.max_frames = camera_conf.get("max_frames", 60)
        self.camera_width = camera_conf.get("width", 1920)
        self.camera_height = camera_conf.get("height", 1080)
        
        self.ring_index = 0
        self.cuda_buffer = []
        
        self._init_jetson_buffer()

    def _init_jetson_buffer(self):
        """Preasigna la memoria RAM física para evitar Garbage Collection"""
        print(f"[HAL] Reservando Ring Buffer ({self.max_frames} frames) en RAM Compartida...")
        try:
            for _ in range(self.max_frames):
                # np.zeros asegura que el bloque de memoria esté limpio y continuo
                frame_vacio = np.zeros((self.camera_height, self.camera_width, 3), dtype=np.uint8)
                self.cuda_buffer.append(frame_vacio)
            print("[HAL] Búfer Zero-Copy inicializado con éxito.")
        except Exception as e:
            print(f"[!] Error crítico al inicializar la memoria física: {e}")

    def agregar_frame(self, frame_data):
        """Sobrescribe la memoria física continuamente mediante copias de bajo nivel"""
        frame_destino = self.cuda_buffer[self.ring_index]
        
        # 1. Protección geométrica: Forzamos el tamaño si la cámara envía otra resolución
        if frame_data.shape[0] != self.camera_height or frame_data.shape[1] != self.camera_width:
            frame_data = cv2.resize(frame_data, (self.camera_width, self.camera_height))
        
        # 2. Copia hiperrápida bloque a bloque (Zero-Copy a nivel de CPU/UMA)
        np.copyto(frame_destino, frame_data)
        
        # 3. Rotamos el puntero
        self.ring_index = (self.ring_index + 1) % self.max_frames

    def extraer_clip_recortado(self, bbox, num_frames=30):
        """
        Extrae un tensor temporal del Ring Buffer sin copiar datos extra.
        Retorna una lista de frames (NumPy arrays) lista para el Juez GGUF.
        """
        # 1. Aseguramos que las coordenadas no rompan los límites de la pantalla
        x1, y1, x2, y2 = map(int, bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(self.camera_width, x2), min(self.camera_height, y2)

        clip_recortado = []
        frames_a_extraer = min(num_frames, self.max_frames)
        
        # 2. Recolectamos los frames en orden cronológico (del más viejo al más nuevo)
        for i in range(frames_a_extraer, 0, -1):
            idx = (self.ring_index - i) % self.max_frames
            frame_completo = self.cuda_buffer[idx]
            
            # 3. Magia de NumPy: Slicing instantáneo [Alto, Ancho]
            recorte = frame_completo[y1:y2, x1:x2]
            
            # Protección contra recortes vacíos (cajas anómalas de YOLO)
            if recorte.size == 0:
                continue
                
            clip_recortado.append(recorte)
            
        return clip_recortado
