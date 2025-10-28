# Usamos una imagen de Python 3.10, compatible con MediaPipe.
FROM python:3.10-slim

# Instalar dependencias del sistema necesarias para algunas librerías Python
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Establecer el directorio de trabajo dentro del contenedor
WORKDIR /app

# COPIA CORREGIDA 1: Copia requirements.txt (asumiendo que está en el contexto raíz /backend)
COPY requirements.txt . 
RUN pip install --no-cache-dir -r requirements.txt

# COPIA CORREGIDA 2: Copia el resto del código del backend (asumiendo que todo está en /backend)
COPY . .

# Comando de despliegue con Gunicorn (servidor de producción)
# Render inyecta $PORT automáticamente.
CMD ["gunicorn", "--bind", "0.0.0.0:$PORT", "run:app"]
