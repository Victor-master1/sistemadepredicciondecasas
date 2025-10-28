# En app/routes/entrenamiento_routes.py

from flask import Blueprint, request, jsonify
from app.services.entrenamiento_service import iniciar_nuevo_entrenamiento

entrenamiento_bp = Blueprint("entrenamiento_bp", __name__)

@entrenamiento_bp.route("", methods=["POST"])
def iniciar_entrenamiento_route():
    """
    Recibe la configuración del frontend, la pasa al servicio de entrenamiento
    y devuelve el nuevo experimento creado.
    """
    try:
        configuracion = request.get_json()
        if not configuracion:
            return jsonify({"error": "No se recibió ninguna configuración"}), 400

        nuevo_experimento = iniciar_nuevo_entrenamiento(configuracion)
        
        return jsonify(nuevo_experimento), 201

    except ValueError as ve: # ✅ Captura errores de validación específicos
        print(f"🔥 Error de validación del usuario: {ve}")
        return jsonify({"error": str(ve)}), 400 # Devuelve un error 400 claro
    except Exception as e:
        print(f"🚨 ERROR en la ruta de entrenamiento: {e}")
        return jsonify({"error": "Ocurrió un error interno en el servidor"}), 500