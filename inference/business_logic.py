import datetime

def generar_prompt_ask_hint():
    """Genera el prompt estructurado para forzar el razonamiento del VLM."""
    return """Eres un inspector de seguridad de un centro comercial. Analiza la imagen.
Responde estrictamente en este formato, siendo muy breve:
1. Objeto: [Qué toca o sostiene el sujeto]
2. Cinematica: [Qué movimiento está haciendo]
3. Contexto: [En qué zona está (ej. pasillo, caja, puerta)]
4. Veredicto: [NORMAL o ALERTA] - [Razón breve]"""

def evaluar_anomalia(respuesta_ia):
    """Parsea la respuesta del VLM y aplica reglas de negocio."""
    hora_actual = datetime.datetime.now().hour
    alerta = False
    mensaje = "ESTADO NORMAL"

    # Extraer de forma segura la línea del veredicto
    lineas = [linea.strip() for linea in respuesta_ia.split('\n') if linea.strip()]
    veredicto_linea = next((l for l in lineas if "4. Veredicto:" in l), "")

    # Regla 1: Análisis puro del VLM
    if "ALERTA" in veredicto_linea.upper():
        alerta = True
        mensaje = veredicto_linea.split("-")[-1].strip()

    # Regla 2: Regla estricta de horario comercial (10 PM a 6 AM)
    if not alerta and (hora_actual >= 22 or hora_actual <= 6):
        texto_completo = respuesta_ia.lower()
        if "caja" in texto_completo or "estanter" in texto_completo or "puerta" in texto_completo:
            alerta = True
            mensaje = "ALERTA: Actividad cerca de zona sensible fuera de horario."

    return alerta, mensaje
