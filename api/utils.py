import psutil
import asyncio

async def obtener_telemetria_hardware():
    """
    Lee el estado del sistema de forma asíncrona.
    En Arch Linux / Jetson, esto es vital para monitorear el consumo de VAU.
    """
    # Ejecutamos las lecturas bloqueantes en un hilo separado
    cpu_percent = await asyncio.to_thread(psutil.cpu_percent, interval=0.1)
    ram = await asyncio.to_thread(psutil.virtual_memory)
    
    return {
        "cpu_uso_porcentaje": cpu_percent,
        "ram_total_gb": round(ram.total / (1024**3), 2),
        "ram_usada_gb": round(ram.used / (1024**3), 2),
        "ram_porcentaje": ram.percent,
        "estado_sistema": "CRITICO" if ram.percent > 90 else "OPTIMO"
    }
