🚀 VAU Export & Optimization Module
Este módulo contiene las herramientas necesarias para convertir, optimizar y exportar modelos de visión (VLM) entrenados en el servidor hacia dispositivos de borde, específicamente optimizados para la NVIDIA Jetson Orin Nano.

🛠️ Contenido del Módulo
export_vlm.py: Script en Python encargado de cargar los pesos del modelo (ej. Qwen-VL o InternVL2) y prepararlos para su serialización. Realiza la conversión de formatos de checkpoint y asegura que los archivos de configuración sean compatibles con entornos embebidos.

compile_to_gguf.sh: Script de automatización para la cuantización de modelos utilizando llama.cpp o herramientas similares hacia el formato GGUF. Esto permite ejecutar modelos pesados en la Jetson Orin Nano optimizando el uso de la VRAM mediante cuantización de 4 bits (INT4) o 5 bits.

📦 Flujo de Trabajo para Jetson Orin Nano
Extracción de Pesos: Ejecutar export_vlm.py para generar una versión limpia del modelo post-entrenamiento.

Compilación: Correr compile_to_gguf.sh para transformar el modelo a un formato eficiente.

Despliegue: Copiar el archivo .gguf resultante a la Jetson y cargarlo mediante el api_server o el cliente de inferencia local.

⚠️ Notas de Rendimiento
Para la Orin Nano, se recomienda priorizar la cuantización Q4_K_M o Q5_K_M para mantener un balance entre la precisión de detección de anomalías y la tasa de frames por segundo (FPS).
