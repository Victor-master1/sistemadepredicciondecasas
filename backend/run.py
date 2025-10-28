from flask import Flask
from flask_cors import CORS
from app import create_app
from facial.facial_routes import facial_bp

def create_full_app():
    app = create_app()
    
    CORS(app, resources={r"/api/*": {
        "origins": ["http://localhost:3000"],
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "supports_credentials": True
    }})
    
    app.register_blueprint(facial_bp)
    
    return app

if __name__ == "__main__":
    app = create_full_app()
    print("🚀 Servidor web de Flask iniciado en el puerto 5000...")
    app.run(port=5000, debug=True)