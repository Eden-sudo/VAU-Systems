import torch
from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from trl import SFTTrainer

print("[*] Iniciando configuración QLoRA...")

MODEL_ID = "../models/base_model"
OUTPUT_DIR = "../lora_adapters"

# Cuantización INT4
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Cargar procesador y modelo (Forzando GPU 0)
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map={"": 0} 
)

# PEFT y LoRA
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules="all-linear", # Universal para arquitecturas VLM
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# Argumentos optimizados para evitar OOM (Out of Memory)
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    optim="paged_adamw_32bit",
    save_steps=50,
    logging_steps=10,
    learning_rate=2e-4,
    fp16=True,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="constant",
)

print("[*] Configurando Trainer...")

# trainer = SFTTrainer(
#     model=model,
#     train_dataset=dataset_citep, # Insertar dataset parseado aquí
#     peft_config=lora_config,
#     dataset_text_field="text", 
#     max_seq_length=512,
#     tokenizer=processor.tokenizer,
#     args=training_args,
# )

print("[*] Entrenando...")
# trainer.train() 

model.save_pretrained(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR) # Guardar procesador garantiza compatibilidad
print(f"[*] Adaptadores LoRA guardados en {OUTPUT_DIR}")
