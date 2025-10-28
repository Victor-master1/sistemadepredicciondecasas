from flask import Blueprint, request, jsonify
from app.services.limpieza_service import limpiar_dataset
# ASUMIMOS que estas funciones ahora devuelven el JSON de datos, no una URL
from app.services.analisis_service import (
    generar_histograma,
    generar_boxplot,
    generar_serie_temporal,
    generar_mapa_calor,
    generar_datos_3d
)

dataset_bp = Blueprint("dataset_bp", __name__)

# -------------------
# Limpiar dataset
# -------------------
@dataset_bp.route("/datasets/<dataset_id>/limpiar", methods=["POST"])
def limpiar(dataset_id):
    try:
        operaciones = request.json or {}
        # El resultado ahora debe incluir el campo 'estadisticas'
        resultado = limpiar_dataset(dataset_id, operaciones)
        return jsonify(resultado), 200
    except Exception as e:
        # Aquí se podría devolver un error 400 si las operaciones son inválidas,
        # pero 500 está bien si es un error interno.
        return jsonify({"error": str(e)}), 500

# -------------------
# Histograma de precios: Devuelve el JSON de datos del histograma
# -------------------
@dataset_bp.route("/datasets/<dataset_id>/histograma-precio", methods=["GET"])
def histograma_precio(dataset_id):
    try:
        # Se asume que generar_histograma devuelve la LISTA DE OBJETOS JSON
        data = generar_histograma(dataset_id)
        # El frontend espera el arreglo directamente, no un objeto con clave "url"
        return jsonify(data), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -------------------
# Boxplot por zonas: Devuelve el JSON de datos del boxplot
# -------------------
@dataset_bp.route("/datasets/<dataset_id>/boxplot-zonas", methods=["GET"])
def boxplot_zonas(dataset_id):
    try:
        # Se asume que generar_boxplot devuelve la LISTA DE OBJETOS JSON
        data = generar_boxplot(dataset_id)
        return jsonify(data), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -------------------
# Serie temporal: Devuelve el JSON de datos de la serie temporal
# -------------------
@dataset_bp.route("/datasets/<dataset_id>/serie-temporal", methods=["GET"])
def serie_temporal(dataset_id):
    try:
        # Se asume que generar_serie_temporal devuelve la LISTA DE OBJETOS JSON
        data = generar_serie_temporal(dataset_id)
        return jsonify(data), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -------------------
# Mapa de calor: Devuelve el JSON de datos del mapa de calor
# -------------------
@dataset_bp.route("/datasets/<dataset_id>/mapa-calor", methods=["GET"])
def mapa_calor(dataset_id):
    try:
        # Se asume que generar_mapa_calor devuelve la LISTA DE OBJETOS JSON
        data = generar_mapa_calor(dataset_id)
        return jsonify(data), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -------------------
# Datos 3D: Devuelve el JSON de datos de dispersión
# -------------------
@dataset_bp.route("/datasets/<dataset_id>/datos-3d", methods=["GET"])
def datos_3d(dataset_id):
    try:
        # Se asume que generar_datos_3d devuelve la LISTA DE OBJETOS JSON
        data = generar_datos_3d(dataset_id)
        return jsonify(data), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
