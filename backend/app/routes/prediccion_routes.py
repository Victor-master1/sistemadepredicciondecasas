from flask import Blueprint, request, jsonify
# NUEVO: Importamos la función del servicio de predicción
from app.services.prediccion_service import realizar_prediccion_casa

prediccion_bp = Blueprint("prediccion_bp", __name__)

@prediccion_bp.route("/prediccion", methods=["POST"])
def predecir_precio_casa():
    """
    Recibe los datos de la casa y el ID del experimento,
    llama al servicio para obtener la predicción y devuelve el resultado.
    """
    try:
        payload = request.get_json()
        if not payload or 'experimento_id' not in payload or 'datos' not in payload:
            return jsonify({"error": "Faltan datos requeridos (experimento_id, datos)"}), 400

        experimento_id = payload['experimento_id']
        datos_casa = payload['datos']

        # Llamar al servicio que hará el trabajo pesado
        resultado_prediccion = realizar_prediccion_casa(experimento_id, datos_casa)

        return jsonify(resultado_prediccion), 200

    except FileNotFoundError as fnf_error:
         print(f"🔥 Error: Archivo de modelo o artefacto no encontrado: {fnf_error}")
         return jsonify({"error": f"No se pudieron cargar los archivos del modelo: {fnf_error}"}), 404
    except ValueError as ve:
        print(f"🔥 Error de validación o datos: {ve}")
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        # Captura cualquier otro error durante la predicción
        print(f"🚨 ERROR en la ruta de predicción: {e}")
        # Considera no exponer detalles internos en producción
        return jsonify({"error": f"Ocurrió un error interno al realizar la predicción: {str(e)}"}), 500
