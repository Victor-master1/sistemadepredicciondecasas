# Usamos una imagen de Python 3.10, compatible con librerías de Machine Learning (MediaPipe/OpenCV).
FROM python:3.10-slim

# Paso 1: Instalar dependencias del sistema operativo.
# Añadimos las librerías gráficas (libGL.so.1, etc.) necesarias para que OpenCV/cv2 funcione 
# dentro del contenedor sin errores 'ImportError: libGL.so.1'.
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    gcc \
    libgl1 \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Establecer el directorio de trabajo dentro del contenedor
WORKDIR /app

# Paso 2: Copiar e instalar dependencias de Python.
# Utilizamos la copia de requirements.txt para optimizar el cacheo.
# Se asume que requirements.txt está en la raíz del contexto de compilación (/backend).
COPY requirements.txt . 
RUN pip install --no-cache-dir -r requirements.txt

# Paso 3: Copiar el resto del código del backend (incluyendo run.py, app/, facial/, etc.).
COPY . .

# Paso 4: Comando de despliegue.
# Inicia Gunicorn, el servidor de producción.
# El comando usa la sintaxis shell para que la variable $PORT de Render se expanda correctamente.
# La aplicación Flask debe estar expuesta como la variable 'app' en el módulo 'run' (run.py:app).
CMD gunicorn --bind 0.0.0.0:$PORT run:app
