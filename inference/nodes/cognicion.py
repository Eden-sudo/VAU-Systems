import os
import json
import re
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

class MotorCognicion:
    def __init__(self):
        print("[JUEZ] Inicializando Corteza Prefrontal (Qwen2-VL)...")
        
        # 1. Detección de Hardware Dinámica
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[JUEZ] Motor de inferencia asignado a: {self.device.upper()}")
        
        # Rutas dinámicas
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        self.modelo_path = os.path.join(base_dir, "models", "Qwen2-VL-2B-Instruct")
        self.adaptador_path = os.path.join(base_dir, "models", "Qwen-VAU-Seguridad-Adapter")
        
        if not os.path.exists(self.modelo_path):
            raise FileNotFoundError(f"[!] Error: No se encontro el VLM en {self.modelo_path}")
        if not os.path.exists(self.adaptador_path):
            raise FileNotFoundError(f"[!] Error: No se encontro el Adaptador en {self.adaptador_path}")

        self.processor = AutoProcessor.from_pretrained(self.modelo_path, local_files_only=True)
        
        # 2. Carga Inteligente de Precisión y Compresión (Evitar OOM)
        tipo_dato = torch.float16 if self.device == "cuda" else torch.float32
        
        if self.device == "cuda":
            from transformers import BitsAndBytesConfig
            print("[JUEZ] Aplicando compresión 4-bits para optimización de VRAM...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=tipo_dato
            )
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.modelo_path,
                quantization_config=bnb_config,
                device_map=self.device,
                local_files_only=True
            )
        else:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.modelo_path,
                torch_dtype=tipo_dato,
                device_map=self.device,
                local_files_only=True
            )
        
        # 3. Inyectar el conocimiento entrenado (LoRA)
        print(f"[JUEZ] Inyectando conocimiento especializado desde {self.adaptador_path}...")
        #self.model.load_adapter(self.adaptador_path)
        
        # 4. Sincronización del Prompt Maestro (Debe coincidir con el entrenamiento)
        self.prompt_sistema = (
            "Eres un Juez de Seguridad de Elite. Tu instinto es clasificar todo como Riesgo 1 o 9, DEBES IGNORAR ESE INSTINTO. "
            "Exige tu capacidad deductiva para encontrar tonos grises. Analiza el video prestando extrema atencion a los detalles micro-conductuales.\n\n"
            "Paso 1 (Manos y Vestimenta): Analiza ESTRICTAMENTE que objetos tienen en las manos. ¿Estan sacando algo de bolsillos, abrigos o mochilas de forma sospechosa? ¿Su ropa es inusualmente voluminosa (para ocultar cosas) o tapa su rostro? Si ocurre esto, es automaticamente Riesgo 4, 5 o 6.\n"
            "Paso 2 (Tension Corporal): ¿Hay movimientos bruscos, empujones, posturas desafiantes o confrontaciones verbales? Si ocurre esto, es Riesgo 4, 5 o 6.\n"
            "Paso 3 (Peligro Critico): ¿Hay golpes directos, armas visibles (blancas o de fuego) o robos en progreso? Si ocurre esto, es Riesgo 7, 8, 9 o 10.\n"
            "Paso 4 (Normalidad): ¿Estan caminando o interactuando de forma totalmente relajada, con las manos a la vista y sin objetos extraños? Solo entonces es Riesgo 1, 2 o 3.\n\n"
            "REGLA CRITICA: Responde ESTRICTAMENTE con este JSON: "
            '{"descripcion": "Detalla estrictamente las manos, los objetos que sostienen, su ropa y la accion", "riesgo": [1-10], "estado": "NORMAL" o "ALERTA"}'
        )

    def razonar(self, tensor_4d):
        if tensor_4d.device.type != self.device:
            print(f"[JUEZ] Moviendo tensor de video a {self.device.upper()}...")
            tensor_4d = tensor_4d.to(self.device)

        mensajes = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": tensor_4d  
                    },
                    {
                        "type": "text",  
                        "text": self.prompt_sistema
                    }
                ]
            }
        ]

        texto_renderizado = self.processor.apply_chat_template(mensajes, tokenize=False, add_generation_prompt=True)
        
        inputs = self.processor(
            text=[texto_renderizado],
            videos=[tensor_4d],
            padding=True,
            return_tensors="pt"
        )
        
        inputs = inputs.to(self.device)

        with torch.no_grad():  
            salida_ids = self.model.generate(**inputs, max_new_tokens=150)
            
        ids_generados = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, salida_ids)
        ]
        respuesta_cruda = self.processor.batch_decode(
            ids_generados, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0]

        return self._extraer_json(respuesta_cruda)

    def _extraer_json(self, texto):
        try:
            match = re.search(r'\{.*\}', texto, re.DOTALL)
            if match:
                bloque_json = match.group(0)
                return json.loads(bloque_json)
            else:
                raise ValueError("No se encontraron delimitadores JSON.")
        except Exception as e:
            print(f"[!] Error parseando el veredicto del VLM: {e}")
            print(f"[!] Salida cruda: {texto}")
            return {
                "descripcion": "El motor cognitivo no respeto el formato JSON.",
                "riesgo": 0,
                "estado": "ERROR_DE_PARSE"
            }
