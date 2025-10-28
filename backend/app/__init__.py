# En app/__init__.py

from flask import Flask
from flask_cors import CORS
from dotenv import load_dotenv

def create_app():
    load_dotenv()
    app = Flask(__name__)
    CORS(app, resources={r"/api/*": {"origins": "*"}})
    
    from app.routes.dataset_routes import dataset_bp
    # ✅ CORRECCIÓN: Volvemos a la URL original para que todo funcione como antes.
    app.register_blueprint(dataset_bp, url_prefix="/api") 

    from app.routes.entrenamiento_routes import entrenamiento_bp
    app.register_blueprint(entrenamiento_bp, url_prefix="/api/entrenamientos")

    from app.routes.experimentos_routes import experimentos_bp
    app.register_blueprint(experimentos_bp, url_prefix="/api/experimentos")

    from app.routes.dashboard_routes import dashboard_bp
    app.register_blueprint(dashboard_bp, url_prefix="/api")

    from app.routes.prediccion_routes import prediccion_bp
    app.register_blueprint(prediccion_bp, url_prefix='/api')
    

    return app