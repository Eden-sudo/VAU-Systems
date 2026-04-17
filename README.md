## Arquitectura del Sistema (VAU: Visual Anomaly Understanding)

VAU es un sistema de visión artificial diseñado para detectar y analizar comportamientos anómalos, optimizado para ejecutarse en hardware de borde como la NVIDIA Jetson Orin Nano. 

**Mecánica de Operación (Atención Dirigida):**
El sistema funciona bajo un esquema de asignación dinámica. El usuario define mediante texto (prompt) qué elementos vigilar (por ejemplo, "trabajador manipulando carga" o "cliente en pasillo"). A partir de esa instrucción, el sistema aísla al objetivo y aplica un razonamiento lógico paso a paso para determinar si la acción es segura o si representa una alerta (como un robo o un acto inseguro en logística o retail).

Para lograr esto sin saturar el hardware, el flujo de procesamiento se divide en 3 fases:

### 1. Fase de Percepción y Enfoque
* **Modelos:** YOLOv8n (Prueba de Concepto) → Evolución a Grounding DINO + SAM 2.
* **Función:** Escanea el video en tiempo real para localizar a los sujetos asignados en el prompt. Una vez detectados, recorta la imagen dejando únicamente a la persona y su entorno inmediato.
* **Objetivo:** Eliminar todo el "ruido" visual del fondo. Al procesar solo recortes de alta calidad en lugar de la cámara completa, se ahorra drásticamente la memoria RAM de la placa.

### 2. Fase de Razonamiento
* **Modelos:** Moondream2 / Qwen2.5-VL (Modelos de Lenguaje Visual).
* **Función:** Recibe los recortes limpios de la primera fase y los analiza de forma secuencial (quién es, qué hace, en qué contexto). 
* **Objetivo:** Evaluar la escena cruzando la información visual con las reglas de seguridad de la empresa. Emite un veredicto estructurado en texto, mitigando errores y falsos positivos al fundamentar su respuesta.

### 3. Fase de Despliegue en Hardware
* **Hardware:** NVIDIA Jetson (Serie Orin Nano / Nano).
* **Optimización:** Cuantización (reducción de precisión matemática) y gestión de memoria.
* **Objetivo:** Comprimir el peso de los modelos de inteligencia artificial para que el sistema entero corra a alta velocidad, de manera continua y estable dentro de la Jetson Nano, sin sobrecalentarla ni agotar sus recursos.
