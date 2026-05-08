// api/static/app.js

document.addEventListener('DOMContentLoaded', () => {
    
    // --- 1. GESTIÓN DE PESTAÑAS (TABS) ---
    const botonesTabs = document.querySelectorAll('.btn-tab');
    const contenidosTabs = document.querySelectorAll('.tab-content');

    botonesTabs.forEach(btn => {
        btn.addEventListener('click', () => {
            // Resetear estilos de todos los botones
            botonesTabs.forEach(b => {
                b.classList.remove('border-blue-500', 'text-blue-400');
                b.classList.add('border-transparent', 'text-gray-400');
            });
            // Ocultar todo el contenido
            contenidosTabs.forEach(c => c.classList.add('hidden'));

            // Activar el botón y contenido seleccionado
            btn.classList.add('border-blue-500', 'text-blue-400');
            btn.classList.remove('border-transparent', 'text-gray-400');
            document.getElementById(btn.dataset.target).classList.remove('hidden');
        });
    });

    // --- 2. SETUP DE RENDERIZADO (CANVAS) ---
    const video = document.getElementById('video-stream');
    const canvas = document.getElementById('capa-dibujo');
    const ctx = canvas.getContext('2d');

    // Video placeholder para pruebas
    video.src = "https://www.w3schools.com/html/mov_bbb.mp4"; 

    video.addEventListener('loadedmetadata', () => {
        // Alinear resoluciones internas del canvas con el video original
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
    });

    // --- 3. CONEXIÓN WEBSOCKET ---
    // Usamos window.location.host para que funcione tanto en localhost como en Tailscale automáticamente
    const protocoloWS = window.location.protocol === 'https:' ? 'wss://' : 'ws://';
    const ws = new WebSocket(`${protocoloWS}${window.location.host}/ws/ui`);
    const btnEstado = document.getElementById('conexion-estado');

    ws.onopen = () => {
        btnEstado.innerText = "SISTEMA ONLINE";
        btnEstado.className = "px-3 py-1 rounded text-sm font-bold bg-green-900 text-green-400 border border-green-500";
    };

    ws.onmessage = (event) => {
        const paquete = JSON.parse(event.data);

        if (paquete.tipo === "percepcion_yolo") {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            paquete.detecciones.forEach(det => {
                ctx.strokeStyle = "#00FF00";
                ctx.lineWidth = 3;
                ctx.strokeRect(det.x, det.y, det.w, det.h);
                
                ctx.fillStyle = "#00FF00";
                ctx.font = "16px monospace";
                ctx.fillText(`ID: #${det.id}`, det.x, det.y - 5);
            });
        }
        else if (paquete.tipo === "alerta_seguridad") {
            const lista = document.getElementById('lista-alertas');
            const placeholder = document.querySelector('.placeholder-alerta');
            if (placeholder) placeholder.remove();

            const li = document.createElement('li');
            li.className = "p-3 bg-red-900/40 border-l-4 border-red-500 rounded text-gray-200 animate-pulse";
            li.innerHTML = `<span class="font-bold text-red-400">⚠️ ID #${paquete.track_id}</span><br>${paquete.datos.descripcion}`;
            lista.prepend(li); // Agregar al principio
        }
        else if (paquete.tipo === "telemetria") {
            document.getElementById('cpu-uso').innerText = `${paquete.cpu}%`;
            document.getElementById('barra-cpu').style.width = `${paquete.cpu}%`;
            
            document.getElementById('ram-uso').innerText = `${paquete.ram} GB`;
            // Jetson Nano tiene 8GB, calculamos porcentaje
            document.getElementById('barra-ram').style.width = `${(paquete.ram / 8) * 100}%`; 
        }
    };

    ws.onclose = () => {
        btnEstado.innerText = "SISTEMA OFFLINE";
        btnEstado.className = "px-3 py-1 rounded text-sm font-bold bg-red-900 text-red-500 border border-red-600";
    };

    // --- 4. CONTROLES DE LA UI ---
    document.getElementById('input-umbral').addEventListener('input', (e) => {
        document.getElementById('valor-umbral').innerText = e.target.value;
        // Aquí enviaríamos el cambio por WebSocket o HTTP al backend
    });
});
