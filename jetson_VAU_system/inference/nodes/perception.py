import os
from ultralytics import YOLOWorld

class MotorPercepcion:
    def __init__(self, config):
        print("[SABUESO] Inicializando corteza visual (YOLO-World + ByteTrack Nativo)...")
        
        base_dir = config.get("models", {}).get("base_dir", "./models")
        yolo_file = config.get("models", {}).get("yolo_model", "yolov8m-worldv2.pt")
        modelo_path = os.path.join(base_dir, yolo_file)
        
        percepcion_cfg = config.get("perception", {})
        self.conf_umbral = percepcion_cfg.get("confidence_threshold", 0.3)
        self.clases = percepcion_cfg.get("target_classes", ["person"])
            
        self.model = YOLOWorld(modelo_path)
        
        print(f"[SABUESO] Inyectando directrices de busqueda: {self.clases}")
        self.model.set_classes(self.clases)

    def detectar(self, frame):
        resultados = self.model.track(
            frame,  
            conf=self.conf_umbral,  
            persist=True,  
            tracker="bytetrack.yaml",  
            verbose=False
        )

        cajas = resultados[0].boxes

        if len(cajas) == 0 or cajas.id is None:
            return []

        pistas_activas = []

        for caja, track_id, clase_id in zip(cajas.xyxy, cajas.id, cajas.cls):
            x1, y1, x2, y2 = caja.cpu().numpy().tolist()
            id_entero = int(track_id.item())
            nombre_clase = self.model.names[int(clase_id.item())]
            pistas_activas.append([int(x1), int(y1), int(x2), int(y2), id_entero, nombre_clase])

        return pistas_activas
