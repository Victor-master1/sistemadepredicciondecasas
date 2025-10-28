import os
from flask import Flask
from flask_cors import CORS
from app import create_app
from facial.facial_routes import facial_bp

# 1. Creamos la instancia de la aplicación directamente
# Hacemos esto fuera del bloque 'if __name__ == "__main__"'
def create_full_app():
    app = create_app()
    
    # 2. **AJUSTE DE CORS** - Necesitas la URL de tu frontend de Railway/Vercel aquí.
    # Usaremos una variable de entorno para la URL del frontend, ¡esto es CRÍTICO!
    # Nota: localhost:3000 es solo para desarrollo local.
    FRONTEND_URL = os.environ.get("FRONTEND_URL", "http://localhost:3000")
    
    CORS(app, resources={r"/api/*": {
        # **IMPORTANTE: Reemplaza con la URL de tu frontend cuando esté desplegado**
        "origins": [FRONTEND_URL, "http://localhost:3000"], 
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "supports_credentials": True
    }})
    
    app.register_blueprint(facial_bp)
    
    return app

# La variable 'app' en el nivel superior que Gunicorn leerá.
app = create_full_app()

if __name__ == "__main__":
    # 3. **AJUSTE DE PUERTO** - Usamos la variable de entorno $PORT de Railway
    port = int(os.environ.get('PORT', 5000))
    
    print(f"🚀 Servidor web de Flask iniciado en el puerto {port}...")
    # Asegúrate de usar host='0.0.0.0' para escuchar peticiones externas al contenedor
    app.run(host='0.0.0.0', port=port, debug=True)
```


