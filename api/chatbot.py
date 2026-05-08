import httpx
import os

# Configuración básica (Idealmente, estas claves van en un archivo .env)
WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN", "TU_TOKEN_AQUI")
WHATSAPP_PHONE_ID = os.getenv("WHATSAPP_PHONE_ID", "TU_PHONE_ID")
NUMERO_DESTINO = os.getenv("NUMERO_GUARDIA", "NUMERO_DE_PRUEBA")

async def notificar_alerta_whatsapp(alerta_json: dict):
    """
    Dispara un mensaje de WhatsApp cuando Qwen detecta una amenaza.
    """
    url = f"https://graph.facebook.com/v17.0/{WHATSAPP_PHONE_ID}/messages"
    
    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json"
    }
    
    # Formateamos el mensaje extraído del dictamen del Juez
    texto_mensaje = (
        f"🚨 *ALERTA DE SEGURIDAD VAU* 🚨\n\n"
        f"Sujeto ID: #{alerta_json.get('track_id', 'Desconocido')}\n"
        f"Nivel de Riesgo: {alerta_json.get('datos', {}).get('riesgo', 'N/A')}/10\n"
        f"Detalle: {alerta_json.get('datos', {}).get('descripcion', 'Movimiento sospechoso detectado.')}"
    )
    
    payload = {
        "messaging_product": "whatsapp",
        "to": NUMERO_DESTINO,
        "type": "text",
        "text": {"body": texto_mensaje}
    }
    
    # Enviamos la petición sin bloquear el event loop
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, headers=headers, json=payload, timeout=5.0)
            if response.status_code == 200:
                print(f"[CHATBOT] Alerta enviada con éxito a {NUMERO_DESTINO}.")
            else:
                print(f"[!] Error enviando WhatsApp: {response.text}")
        except Exception as e:
            print(f"[!] Error de conexión con la API de WhatsApp: {e}")
