# 👁️ VAU-Systems (Vision Analytics & Understanding)

VAU-Systems es una arquitectura de seguridad cognitiva *Edge-AI* de alto rendimiento, diseñada para operar en hardware restringido (NVIDIA Jetson Orin Nano). El sistema combina detección de objetos en tiempo real con razonamiento multimodal profundo, operando bajo una filosofía de microservicios asíncronos y procesamiento *Zero-Copy*.

---

## 🏗️ Arquitectura del Sistema

El núcleo de VAU-Systems está estrictamente desacoplado para garantizar que el renderizado web o los fallos de red nunca bloqueen la inferencia de la GPU. 

```mermaid
flowchart LR
    %% (flowchart LR
    %% ORQUESTADOR Y PIPELINE
    subgraph Orquestador ["1. Orquestador Asíncrono (asyncio)"]
        direction TB
        subgraph Task1 ["Hilo Productor (30 FPS)"]
            CAM["cv2.VideoCapture"] --> SAB["YOLO-World + BoT-SORT"]
        end
        
        subgraph Task2 ["Hilo Consumidor (Background)"]
            QWEN["Qwen2-VL 2B"] --> DICT["Memoria de Eventos"]
        end
        
        COLA[/"asyncio.Queue (Buffer de Clips)"/]
        
        Task1 -->|"Produce BBox"| COLA
        COLA -->|"Consume Tensor 4D"| Task2
    end

    %% GESTOR DE MEMORIA
    subgraph HAL ["2. Hardware Abstraction Layer (HAL)"]
        direction TB
        HW{"platform.machine()"}
        HW -->|"aarch64 (Jetson)"| VRAM["jetson_utils Zero-Copy"]
        HW -->|"x86_64 (PC)"| RAM["NumPy Deque en RAM"]
        VRAM --> ENS["Generador de Tubo 4D RGB"]
        RAM --> ENS
    end

    %% CONEXIONES CRUZADAS
    CAM -.->|"Frames al Histórico"| HW
    SAB -.->|"Coordenadas Recorte"| ENS
    ENS == "Inyecta Tensor" ==> COLA
    
    %% SALIDA
    DICT --> HUD["HUD OpenCV / CUDA"])

El sistema se divide en tres pilares fundamentales:

### 1. El Sabueso (Percepción Visual)
*   **Motor:** YOLO-World (v8) + BoT-SORT ReID.
*   **Función:** Escaneo de alta velocidad (FPS) para detectar entidades (personas, mochilas, armas) y mantener un seguimiento temporal persistente (Tracks).
*   **Optimización:** Inferencia en Tensor Cores mediante FP16/AMP.

### 2. El Juez (Cognición VLM)
*   **Motor:** Qwen2-VL-2B-Instruct (Cuantizado vía NF4 / LoRA).
*   **Función:** Evalúa "Tubos Temporales" (recortes de video de un sujeto) extraídos por el Gestor de Memoria HAL. Determina mediante razonamiento lógico si la acción constituye una actividad normal o una amenaza (ej. robo, pelea).

### 3. API Gateway & Centro de Mando
*   **Motor:** FastAPI + WebSockets + HTML5 Canvas.
*   **Función:** Interfaz de usuario servida en red local o VPN (Tailscale). El renderizado visual de cajas y alertas se realiza en el cliente (*Client-Side Rendering*), liberando a la Jetson de operaciones de OpenCV.
*   **Comunicación:** Utiliza **ZeroMQ** (IPC) para recibir telemetría y veredictos del motor de inferencia con latencia sub-milisegundo.

---

## 📂 Estructura del Proyecto

```text
VAU-Systems/
├── api/                     # API Gateway (Servidor Web y WebSockets)
│   ├── static/              # Centro de Mando (HTML/JS/CSS desacoplado)
│   ├── chatbot.py           # Integración con WhatsApp API para alertas
│   └── ...
├── config/                  # Archivos YAML y configuraciones globales
├── datasets/                # Almacenamiento de video raw y JSONL
├── inference/               # Subsistema principal de IA
│   ├── core/                # Orquestador asíncrono y Gestor HAL
│   └── nodes/               # Módulos del Sabueso (YOLO) y el Juez (Qwen)
├── models/                  # Pesos de modelos (.pt), adaptadores LoRA
├── training/                # Scripts de Finetuning (QLoRA, YOLO)
├── launch_api_models.py     # Punto de entrada: Motor de IA
├── launch_api_server.py     # Punto de entrada: Servidor Web
└── vau_control.sh           # CLI Wrapper para gestión del sistema

--

🚀 Instalación y Despliegue
Requisitos Previos
SO: Arch Linux / Ubuntu 22.04+ (Optimizados para arquitecturas ARM/Tegra).

Hardware: Dispositivo NVIDIA (Jetson Orin Nano 8GB o superior).

Dependencias de red: tailscale (Recomendado para acceso seguro y remoto).

ZeroMQ: sudo pacman -S zeromq (Arch) o sudo apt install libzmq3-dev.

Configuración del Entorno
Clona el repositorio y accede a la raíz:

Bash
git clone <tu-repo-vau>
cd VAU-Systems

2. Crea y activa el entorno virtual:
   ```bash
   python -m venv vau_env
   source vau_env/bin/activate
   
Instala las dependencias:

Bash
pip install -r requirements.txt


---

## ⚙️ Uso Operativo

El sistema se gestiona de manera modular. Puedes levantar la inferencia y el servidor de forma independiente.

**Terminal 1 (Motor de Inferencia):**
```bash
source vau_env/bin/activate
python launch_api_models.py
Terminal 2 (API & Centro de Mando):

Bash
source vau_env/bin/activate
python launch_api_server.py
Acceso Remoto
Para acceder al Centro de Mando Operativo de forma segura desde cualquier ubicación, asegúrate de tener el nodo activo en Tailscale:

Bash
tailscale ip -4
Abre en el navegador: http://<IP_TAILSCALE>:8000/dashboard

🧠 Flujo de Entrenamiento (Finetuning)
VAU-Systems incluye una suite de entrenamiento localizada en la carpeta training/ para adaptar los modelos al entorno específico del cliente:

preparar_dataset.py: Convierte recortes de video locales a formato JSONL conversacional compatible con Qwen.

train_qlora.py: Ejecuta el ajuste fino (QLoRA NF4) del VLM.

fusion_qwen.py: Fusiona el adaptador LoRA resultante con el modelo base para un despliegue ligero en la Jetson.
