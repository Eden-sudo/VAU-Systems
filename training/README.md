VAU-System: Suite de Entrenamiento
Este repositorio contiene los scripts diseñados para el ajuste fino (fine-tuning) y la optimización del modelo Qwen2-VL aplicado a tareas de seguridad.

Entorno de Ejecución
Estos scripts están configurados para operar exclusivamente en el servidor de IA del laboratorio (citep-ia-server). El uso de rutas absolutas y configuraciones de hardware (cuantización NF4, optimizadores fusionados y gestión de hilos de CPU) está optimizado para la arquitectura del servidor.

Descripción de Scripts
1. train_qlora.py
Script principal de entrenamiento mediante QLoRA (Quantized Low-Rank Adaptation).

Cuantización: Implementa NF4 de 4 bits para reducir el consumo de VRAM.

Parches: Incluye correcciones de compatibilidad para torchvision y PyAV.

Lógica de Máscara: Aplica un enmascaramiento de pérdida (loss masking) sobre el prompt y los tokens de visión para que el modelo aprenda únicamente a generar la respuesta del asistente.

Optimización: Configurado con adamw_torch_fused y gradient_checkpointing para maximizar el uso de los núcleos CUDA y minimizar el impacto en la memoria.

2. merge_weights.py
Script para la consolidación del modelo tras el entrenamiento.

Fusión: Realiza el proceso de merge_and_unload para integrar las matrices de pesos entrenadas (adaptador) en el modelo base.

Salida: Exporta el modelo unificado en precisión FP16 junto con el procesador multimodal, listo para su despliegue en entornos de inferencia.

Flujo de Trabajo
Preparación: Verificar que el dataset dataset_seguridad_vau.jsonl esté actualizado en el directorio de datos.

Entrenamiento: Ejecutar train_qlora.py dentro del entorno virtual env_lab.

Exportación: Ejecutar merge_weights.py para generar el modelo final optimizado en la ruta de salida.

Requisitos Técnicos
Hardware: GPU con soporte para bfloat16 y arquitectura compatible con SDPA (Scaled Dot Product Attention).

Software: Librerías transformers, peft, bitsandbytes y qwen_vl_utils instaladas en el servidor.
