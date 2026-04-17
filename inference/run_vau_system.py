import cv2
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
from ultralytics import YOLO
from business_logic import evaluar_anomalia, generar_prompt_ask_hint

print("[*] Iniciando motor de inferencia VAU...")

# 1. Cargar Percepción (YOLOv8 Nano - Rápido para la demo)
detector = YOLO('yolov8n.pt')

# 2. Cargar Razonamiento (VLM) desde la ruta local de tu script de descarga
MODEL_PATH = "../models/base_model"
print(f"[*] Cargando VLM desde {MODEL_PATH}...")

# Configuración optimizada para Moondream2 en la Jetson/GPU
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map={"": 0},
    torch_dtype=torch.float16,
    trust_remote_code=True
)

cap = cv2.VideoCapture(0)
PROMPT = generar_prompt_ask_hint()

print("[*] Sistema Online. Presiona 'q' para salir.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    alerta_activa = False
    mensaje_ui = "MONITOREO ACTIVO"
    respuesta_raw = ""

    # Detección de humanos (Clase 0 en COCO)
    resultados = detector(frame, classes=[0], verbose=False)

    for r in resultados:
        for caja in r.boxes:
            x1, y1, x2, y2 = map(int, caja.xyxy[0])
            w, h = x2 - x1, y2 - y1
            
            # Smart Crop: Expandir 1.5x para mantener contexto sin enviar toda la imagen
            cx, cy = x1 + w//2, y1 + h//2
            side = int(max(w, h) * 1.5)
            
            nx1, ny1 = max(0, cx - side//2), max(0, cy - side//2)
            nx2, ny2 = min(frame.shape[1], cx + side//2), min(frame.shape[0], cy + side//2)

            crop = frame[ny1:ny2, nx1:nx2]
            if crop.size == 0: continue

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # Inferencia VLM con el recorte contextual
            imagen_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            
            try:
                # Moondream2 API específica para mejor rendimiento
                enc_image = model.encode_image(imagen_pil)
                respuesta_ia = model.answer_question(enc_image, PROMPT, tokenizer)
                respuesta_raw = respuesta_ia
                
                # Evaluación
                alerta, mensaje = evaluar_anomalia(respuesta_ia)
                if alerta:
                    alerta_activa = True
                    mensaje_ui = mensaje
            except Exception as e:
                print(f"[!] Error en VLM: {e}")

    # UI Visual
    color = (0, 0, 255) if alerta_activa else (0, 255, 0)
    cv2.putText(frame, mensaje_ui, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    # Imprimir log crudo en pantalla para que el director vea que la IA "piensa"
    if respuesta_raw:
        cv2.putText(frame, "IA: " + respuesta_raw.split('\n')[-1][:50], (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow("VAU System - CITEP Demo", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("[*] Sistema desconectado.")
