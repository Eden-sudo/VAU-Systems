"""
Script de entrenamiento QLoRA para Qwen2-VL.
Optimizado para el ajuste fino multimodal en tareas de seguridad VAU.
"""

import os
import torch
import av
import warnings

# Supresion de advertencias de librerias de terceros
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers.video_utils")

# Parche para dispatch de kernels en torchvision
_orig_has_kernel = torch._C._dispatch_has_kernel_for_dispatch_key
def _safe_has_kernel(qualname, dispatch_key):
    try:
        return _orig_has_kernel(qualname, dispatch_key)
    except RuntimeError as e:
        if "does not exist" in str(e):
            return False
        raise e
torch._C._dispatch_has_kernel_for_dispatch_key = _safe_has_kernel

# Parche de compatibilidad para PyAV
if not hasattr(av, 'AVError'):
    try:
        av.AVError = av.error.Error
    except AttributeError:
        av.AVError = Exception

from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, BitsAndBytesConfig, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from qwen_vl_utils import process_vision_info

# Resolucion dinamica de rutas basadas en la estructura del proyecto
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
RUTA_MODELO_LOCAL = os.path.join(BASE_DIR, "models", "Qwen2-VL-2B-Instruct")
RUTA_DATASET = os.path.join(BASE_DIR, "datasets", "dataset_seguridad_vau.jsonl")
RUTA_SALIDA = os.path.join(BASE_DIR, "models", "Qwen-VAU-Seguridad-Adapter")

def iniciar_entrenamiento():
    """Configura y ejecuta el entrenamiento k-bit con adaptadores LoRA."""
    
    # Configuracion de cuantizacion NF4 para ahorro de VRAM
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    # Carga de artefactos locales
    procesador = AutoProcessor.from_pretrained(RUTA_MODELO_LOCAL, local_files_only=True)
    modelo = Qwen2VLForConditionalGeneration.from_pretrained(
        RUTA_MODELO_LOCAL,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation="sdpa",
        local_files_only=True
    )
    
    # Preparacion para entrenamiento k-bit y ahorro de memoria via Checkpointing
    modelo.config.use_cache = False  
    modelo = prepare_model_for_kbit_training(modelo, use_gradient_checkpointing=True)

    # Configuracion de LoRA apuntando a todas las capas lineales
    lora_config = LoraConfig(
        r=32,  
        lora_alpha=64,
        target_modules="all-linear",  
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    modelo = get_peft_model(modelo, lora_config)
    modelo.enable_input_require_grads() 

    # Carga y pre-procesamiento del dataset
    dataset = load_dataset("json", data_files=RUTA_DATASET, split="train")

    def inyectar_limites_video(ejemplo):
        """Ajusta metadatos de video para normalizar la carga computacional."""
        for msg in ejemplo["messages"]:
            if msg["role"] == "user":
                for item in msg["content"]:
                    if item["type"] == "video":
                        item["fps"] = 1.0  
        return ejemplo
    
    dataset = dataset.map(inyectar_limites_video, desc="Normalizando FPS de video")

    def collate_fn(ejemplos):
        """Maneja el padding y enmascaramiento de labels para entrenamiento causal."""
        try:
            mensajes_batch = [e["messages"] for e in ejemplos]
            textos = [procesador.apply_chat_template(m, tokenize=False, add_generation_prompt=False) for m in mensajes_batch]
            
            image_inputs, video_inputs = process_vision_info(mensajes_batch)
            
            batch = procesador(
                text=textos,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            )
            
            labels = batch["input_ids"].clone()
            
            # Enmascaramiento de tokens de padding e identificadores de vision
            labels[labels == procesador.tokenizer.pad_token_id] = -100
            for vision_token_id in [151652, 151653, 151654, 151655]:
                labels[labels == vision_token_id] = -100
                
            # Enmascaramiento del prompt para entrenar solo sobre la respuesta del asistente
            im_start_id = procesador.tokenizer.convert_tokens_to_ids("<|im_start|>")
            assistant_tokens = procesador.tokenizer.encode("assistant\n", add_special_tokens=False)
            secuencia_magica = [im_start_id] + assistant_tokens
            largo_seq = len(secuencia_magica)
            
            for i in range(labels.shape[0]):  
                tokens = labels[i].tolist()
                for j in range(len(tokens) - largo_seq + 1):
                    if tokens[j:j+largo_seq] == secuencia_magica:
                        labels[i, :j+largo_seq] = -100  
                        break
            
            batch["labels"] = labels
            return batch
            
        except Exception as e:
            print(f"[ERROR] Fallo en colacion de datos: {e}")
            raise e

    # Configuracion del ciclo de entrenamiento
    argumentos_entrenamiento = TrainingArguments(
        output_dir=RUTA_SALIDA,
        per_device_train_batch_size=2,      
        gradient_accumulation_steps=2,      
        learning_rate=1e-4,  
        optim="adamw_torch_fused",           
        num_train_epochs=3,
        save_strategy="epoch",
        logging_steps=1,                    
        fp16=False,  
        bf16=True,  
        remove_unused_columns=False,
        dataloader_num_workers=8,           
        dataloader_prefetch_factor=2,       
        max_grad_norm=1.0,                  
        gradient_checkpointing=True,        
        report_to="none"
    )

    trainer = Trainer(
        model=modelo,
        train_dataset=dataset,
        data_collator=collate_fn,
        args=argumentos_entrenamiento,
    )

    print("[INFO] Iniciando ejecucion de Trainer.")
    trainer.train()

    # Persistencia de adaptador y procesador final
    trainer.model.save_pretrained(RUTA_SALIDA)
    procesador.save_pretrained(RUTA_SALIDA)
    print(f"[INFO] Entrenamiento completado. Resultados en: {RUTA_SALIDA}")

if __name__ == "__main__":
    iniciar_entrenamiento()
