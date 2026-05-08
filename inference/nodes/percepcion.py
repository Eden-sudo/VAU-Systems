import os
import yaml
from ultralytics import YOLOWorld

class MotorPercepcion:
    def __init__(self, clases_objetivo=None, conf_umbral=0.15):
        print("[SABUESO] Inicializando corteza visual (YOLO-World + BoT-SORT ReID)...")
        
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        carpeta_modelos = os.path.join(base_dir, "models")
        modelo_path = os.path.join(carpeta_modelos, "yolov8m-worldv2.pt")
        
        # --- FIX DEFINITIVO: INYECCIÓN DE RUTA ABSOLUTA ---
        tracker_base_path = os.path.join(base_dir, "config", "custom_tracker.yaml")
        self.tracker_activo_path = os.path.join(base_dir, "config", "_active_tracker.yaml")
        
        if not os.path.exists(tracker_base_path):
            raise FileNotFoundError(f"[!] Error: Archivo base no encontrado en {tracker_base_path}")

        # Leemos tu YAML original
        with open(tracker_base_path, 'r') as f:
            tracker_config = yaml.safe_load(f)

        # Forzamos la ruta absoluta hacia la carpeta 'models'
        ruta_reid_absoluta = os.path.join(carpeta_modelos, "yolov8n-cls.pt")
        tracker_config['model'] = ruta_reid_absoluta

        # Guardamos un YAML temporal que el rastreador usará
        with open(self.tracker_activo_path, 'w') as f:
            yaml.safe_dump(tracker_config, f)
        # --------------------------------------------------
        
        if not os.path.exists(modelo_path):
            raise FileNotFoundError(f"[!] Error: El motor neuronal no existe en {modelo_path}")
            
        self.model = YOLOWorld(modelo_path)
        self.conf_umbral = conf_umbral
        
        if clases_objetivo is None:
            # Aprovechamos el Zero-Shot del modelo base
            self.clases = ["person arguing", "person fighting", "aggressive person", "person"]
        else:
            self.clases = clases_objetivo
            
        print(f"[SABUESO] Inyectando directrices de busqueda: {self.clases}")
        self.model.set_classes(self.clases)
        
    def detectar(self, frame):
        """
        Escanea el fotograma aplicando seguimiento temporal avanzado.
        """
        # Inyectamos el archivo YAML personalizado en el argumento tracker
        resultados = self.model.track(
            frame, 
            conf=self.conf_umbral, 
            persist=True, 
            tracker=self.tracker_activo_path, 
            verbose=False
        )
        
        cajas = resultados[0].boxes
        
        if len(cajas) == 0 or cajas.id is None:
            return []

        pistas_activas = []
        
        for caja, track_id in zip(cajas.xyxy, cajas.id):
            x1, y1, x2, y2 = caja.cpu().numpy().tolist()
            id_entero = int(track_id.item())
            pistas_activas.append([int(x1), int(y1), int(x2), int(y2), id_entero])
            
        return pistas_activas
