import platform
import cv2

class HUDCognitivo:
    def __init__(self):
        self.arch = platform.machine()
        self.is_jetson = 'aarch64' in self.arch.lower()
        
        if self.is_jetson:
            print("[HUD] Arquitectura ARM64 detectada. Iniciando Motor Grafico CUDA...")
            try:
                import jetson_utils
                self.jetson_utils = jetson_utils
                # Cargamos la fuente directamente en la VRAM de la GPU
                self.font = jetson_utils.cudaFont()
                self.motor = "CUDA_NATIVO"
            except ImportError:
                print("[!] Advertencia: jetson_utils no encontrado. Cayendo a CPU (OpenCV).")
                self.motor = "OPENCV_CPU"
        else:
            print("[HUD] Arquitectura x86_64 (PC) detectada. Iniciando Motor OpenCV...")
            self.motor = "OPENCV_CPU"

    def renderizar_analiticas(self, frame_data, tracks, diccionario_eventos):
        """
        Enrutador dinámico. Dirige el dibujo al hardware correspondiente.
        """
        if self.motor == "CUDA_NATIVO":
            self._renderizar_jetson_gpu(frame_data, tracks, diccionario_eventos)
            return frame_data # Retorna el mismo objeto cudaImage modificado
        else:
            frame_modificado = self._renderizar_pc_cpu(frame_data, tracks, diccionario_eventos)
            return frame_modificado

    def _renderizar_pc_cpu(self, frame_np, tracks, diccionario_eventos):
        """Método generico para pruebas en Laptop/Estacion de trabajo"""
        frame_display = frame_np.copy()
        
        for track in tracks:
            # track_id simulado para la explicacion
            x1, y1, x2, y2, track_id = track
            
            estado = diccionario_eventos.get(track_id, {"texto": "RASTREANDO", "color": (0, 255, 0)})
            color = estado["color"]
            
            cv2.rectangle(frame_display, (x1, y1), (x2, y2), color, 2)
            
            etiqueta = f"ID:{track_id} | {estado['texto']}"
            cv2.putText(frame_display, etiqueta, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
        return frame_display

    def _renderizar_jetson_gpu(self, frame_cuda, tracks, diccionario_eventos):
        """Método especifico para Jetson. Cero intervencion de CPU."""
        for track in tracks:
            x1, y1, x2, y2, track_id = track
            
            estado = diccionario_eventos.get(track_id, {"texto": "RASTREANDO", "color": (0, 255, 0, 255)})
            # NVIDIA usa RGBA (Red, Green, Blue, Alpha/Transparencia)
            r, g, b, a = estado["color"]
            
            # Dibujo de primitivas directo en la VRAM
            self.jetson_utils.cudaDrawRect(frame_cuda, (x1, y1, x2, y2), (r, g, b, a))
            
            etiqueta = f"ID:{track_id} | {estado['texto']}"
            # Renderizado de texto acelerado por hardware
            self.font.OverlayText(frame_cuda, frame_cuda.width, frame_cuda.height, etiqueta, x1, y1 - 20, self.font.White, (r, g, b, a))

    def exportar_tubo_debug(self, tensor_4d, track_id):
            """Re-ensambla el Tensor 4D y lo guarda como MP4 para ver la perspectiva de Qwen"""
            import os
            import torch
            import numpy as np

            os.makedirs("debug_vau", exist_ok=True)
            ruta_salida = f"debug_vau/vision_qwen_id_{track_id}.mp4"

            # 1. Bajar el tensor de la GPU y volver a formato de imagen (T, H, W, C)
            if isinstance(tensor_4d, torch.Tensor):
                frames_np = tensor_4d.cpu().permute(0, 2, 3, 1).numpy().astype(np.uint8)
            else:
                frames_np = tensor_4d.astype(np.uint8)

            alto, ancho = frames_np.shape[1], frames_np.shape[2]
        
            # 2. Configurar el escritor de video de OpenCV
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(ruta_salida, fourcc, 30.0, (ancho, alto))

            for frame in frames_np:
                 # Revertir de RGB (como piensa Qwen) a BGR (como escribe OpenCV)
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)

            out.release()
            print(f"[DEBUG] Tubo del Sujeto #{track_id} exportado a: {ruta_salida}")
