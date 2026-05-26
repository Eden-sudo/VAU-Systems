import os
import base64
import cv2
import json
import re
import gc
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Qwen25VLChatHandler

class MotorCognicion:
    def __init__(self, config, prompt_maestro):
        print("[JUEZ] Cargando Inteligencia GGUF (Qwen2-VL Int4 + Vision F16)...")
        
        base_dir = config["models"]["base_dir"]
        self.modelo_path = os.path.join(base_dir, config["models"]["qwen_gguf"])
        self.proyector_path = os.path.join(base_dir, config["models"]["proyector_gguf"])
        self.prompt_sistema = prompt_maestro

        # Volvemos al handler que la libreria compilo perfectamente
        chat_handler = Qwen25VLChatHandler(clip_model_path=self.proyector_path)

        self.model = Llama(
            model_path=self.modelo_path,
            chat_handler=chat_handler,
            n_gpu_layers=-1,         
            n_ctx=3072,              # Espacio ideal para 4 frames optimizados
            n_batch=8,
            use_mmap=True,           # Paginacion hacia la microSD activa
            verbose=False            # Apagamos los logs de C++ para ver clara la UI
        )
        print("[JUEZ] Nodo Cognitivo Qwen cargado con exito.")

    def _frames_a_base64(self, clip_frames):
        """Extrae 4 fotogramas clave y los optimiza a 224x224."""
        frames_b64 = []
        l = len(clip_frames)
        
        indices = [0, l//3, 2*(l//3), l-1] if l >= 4 else [0, l-1]
        
        for idx in indices:
            frame = clip_frames[idx]
            # Resolucion optima para que los tokens encajen en el n_ctx
            img = cv2.resize(frame, (224, 224))
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            _, buffer = cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 60])
            frames_b64.append(base64.b64encode(buffer).decode('utf-8'))
            
        return frames_b64

    def razonar(self, clip_frames, clase_yolo="person"):
        # EL FIX MAESTRO: Vaciar la memoria KV para evitar tokens corruptos entre videos
        self.model.reset()
        
        contexto_radar = f"\n\n[DATOS DEL RADAR SENSORIAL: YOLO ha detectado un/una '{clase_yolo}'. Analiza la accion temporal.]\n\n"
        prompt_final = self.prompt_sistema + contexto_radar

        if not clip_frames or len(clip_frames) == 0:
            return {"estado": "ERROR", "riesgo": 0, "descripcion": "Sin video."}

        frames = self._frames_a_base64(clip_frames)
        
        # Envio puro de las imagenes sin etiquetas manuales <__media__>
        contenido_usuario = [{"type": "text", "text": prompt_final}]
        for b64 in frames:
            contenido_usuario.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})

        mensajes = [{"role": "user", "content": contenido_usuario}]

        try:
            res = self.model.create_chat_completion(
                messages=mensajes,  
                response_format={"type": "json_object"}
            )
            contenido = res["choices"][0]["message"]["content"]
            
            match = re.search(r'\{.*\}', contenido, re.DOTALL)
            if match:
                return json.loads(match.group())
                
            return {"estado": "DESPEJADO", "riesgo": 0, "descripcion": "Analisis completado sin amenazas."}
            
        except Exception as e:
            print(f"[!] Error en procesamiento neuronal: {e}")
            return {"estado": "ERROR", "riesgo": 0, "descripcion": "Fallo cognitivo interno."}
        finally:
            gc.collect()
