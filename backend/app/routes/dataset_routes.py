from flask import Blueprint, request, jsonify, send_file
from app.services.limpieza_service import limpiar_dataset
from app.services.datasets_service import ( 
    obtener_lista_datasets, 
    subir_dataset_csv_metadata,
    obtener_archivo_dataset,
    eliminar_dataset_por_id
)
from app.services.analisis_service import (
    obtener_dataframe_crudo,
    obtener_vista_previa_paginada,
    estadisticas_dataset,
    obtener_columnas,
    distribucion_clases,
    generar_histograma,
    generar_boxplot,
    generar_serie_temporal,
    generar_mapa_calor,
    generar_datos_3d,
    generar_analisis_radar,
    generar_sankey_flujo,
    generar_grafico_burbujas,
    generar_analisis_sensibilidad,
    generar_clustering,
    generar_segmentacion_mercado,
    detectar_anomalias_precios,
    calcular_score_inversion
)

dataset_bp = Blueprint("dataset_bp", __name__)

# =========================================================================
# 1. RUTAS DE CRUD PRINCIPAL (Completadas para Datasets.tsx)
# =========================================================================

@dataset_bp.route("/datasets", methods=["GET"])
def listar_datasets():
    """Ruta para listar todos los datasets disponibles. SOLUCIONA: GET /api/datasets 404."""
    try:
        datasets = obtener_lista_datasets() 
        return jsonify(datasets), 200
    except Exception as e:
        print(f"Error al listar datasets: {e}")
        return jsonify({"error": "No se pudo obtener la lista de datasets"}), 500

@dataset_bp.route("/datasets", methods=["POST"])
def subir_dataset():
    """Ruta para registrar metadata del dataset. Frontend envía: { nombre, archivo_url, usuario_id }."""
    try:
        data = request.json
        if not data:
            return jsonify({"error": "Datos de metadata faltantes"}), 400
        
        resultado = subir_dataset_csv_metadata(data) 
        return jsonify(resultado), 201
    except Exception as e:
        print(f"Error al subir metadata del dataset: {e}")
        return jsonify({"error": str(e)}), 500

@dataset_bp.route("/datasets/<dataset_id>", methods=["DELETE"])
def eliminar_dataset(dataset_id):
    """Ruta para eliminar un dataset por ID. Requerido por Datasets.tsx."""
    try:
        eliminar_dataset_por_id(dataset_id) 
        return jsonify({"mensaje": f"Dataset {dataset_id} eliminado exitosamente."}), 204
    except Exception as e:
        print(f"Error al eliminar dataset: {e}")
        return jsonify({"error": str(e)}), 500

@dataset_bp.route("/datasets/<dataset_id>/descargar", methods=["GET"])
def descargar_dataset(dataset_id):
    """Descarga el archivo CSV asociado al dataset."""
    try:
        archivo_buffer, nombre_archivo = obtener_archivo_dataset(dataset_id) 
        
        return send_file(
            archivo_buffer,
            mimetype='text/csv',
            as_attachment=True,
            download_name=nombre_archivo
        )
    except Exception as e:
        return jsonify({"error": f"Error al descargar el archivo: {str(e)}"}), 500

# =========================================================================
# 2. RUTAS DE LIMPIEZA Y ANÁLISIS 
# =========================================================================

@dataset_bp.route("/datasets/<dataset_id>/limpiar", methods=["POST"])
def limpiar(dataset_id):
    """Aplica las operaciones de limpieza y devuelve las estadísticas completas."""
    try:
        operaciones = request.json or {}
        # Resultado incluye la estructura ResultadoLimpieza COMPLETA (con 'estadisticas')
        resultado = limpiar_dataset(dataset_id, operaciones) 
        return jsonify(resultado), 200
    except Exception as e:
        return jsonify({"error": f"Error al limpiar: {str(e)}"}), 500


@dataset_bp.route("/datasets/<dataset_id>/vista-previa", methods=["GET"])
def vista_previa(dataset_id):
    """Sirve la vista previa paginada y maneja la conversión de nulos a '[NULL]'."""
    try:
        pagina = int(request.args.get('pagina', 1))
        df = obtener_dataframe_crudo(dataset_id)
        datos_paginados = obtener_vista_previa_paginada(df, pagina)
        return jsonify(datos_paginados), 200
    except Exception as e:
        # Devuelve la estructura de datos esperada para evitar errores de renderizado en el frontend
        return jsonify({"datos": [], "pagina_actual": 1, "total_paginas": 1, "total_filas": 0, "error": str(e)}), 500


@dataset_bp.route("/datasets/<dataset_id>/columnas", methods=["GET"])
def info_columnas(dataset_id):
    """Devuelve la información detallada de las columnas (ColumnaStat[])."""
    try:
        df = obtener_dataframe_crudo(dataset_id)
        info = obtener_columnas(df)
        return jsonify(info), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@dataset_bp.route("/datasets/<dataset_id>/estadisticas", methods=["GET"])
def estadisticas(dataset_id):
    """Devuelve el resumen estadístico del dataset (EstadisticasDatos)."""
    try:
        df = obtener_dataframe_crudo(dataset_id)
        stats = estadisticas_dataset(df)
        return jsonify(stats), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@dataset_bp.route("/datasets/<dataset_id>/distribucion-clases", methods=["GET"])
def distribucion(dataset_id):
    """Devuelve la distribución de clases (DistribucionClases[])."""
    try:
        df = obtener_dataframe_crudo(dataset_id)
        dist = distribucion_clases(df)
        return jsonify(dist), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --- Rutas de Visualización ---

@dataset_bp.route("/datasets/<dataset_id>/histograma-precio", methods=["GET"])
def histograma_precio(dataset_id):
    """Devuelve datos JSON para el Histograma."""
    try:
        df = obtener_dataframe_crudo(dataset_id)
        data = generar_histograma(df)
        return jsonify(data), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@dataset_bp.route("/datasets/<dataset_id>/boxplot-zonas", methods=["GET"])
def boxplot_zonas(dataset_id):
    """Devuelve datos JSON para el Boxplot."""
    try:
        df = obtener_dataframe_crudo(dataset_id)
        data = generar_boxplot(df)
        return jsonify(data), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@dataset_bp.route("/datasets/<dataset_id>/serie-temporal", methods=["GET"])
def serie_temporal(dataset_id):
    """Devuelve datos JSON para la Serie Temporal."""
    try:
        df = obtener_dataframe_crudo(dataset_id)
        data = generar_serie_temporal(df)
        return jsonify(data), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@dataset_bp.route("/datasets/<dataset_id>/mapa-calor", methods=["GET"])
def mapa_calor(dataset_id):
    """Devuelve datos JSON para el Mapa de Calor (Precio por Zona)."""
    try:
        df = obtener_dataframe_crudo(dataset_id)
        data = generar_mapa_calor(df)
        return jsonify(data), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@dataset_bp.route("/datasets/<dataset_id>/datos-3d", methods=["GET"])
def datos_3d(dataset_id):
    """Devuelve datos JSON para el gráfico de Dispersión 3D/2D."""
    try:
        df = obtener_dataframe_crudo(dataset_id)
        data = generar_datos_3d(df)
        return jsonify(data), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# =========================================================================
# 3. NUEVO: RUTAS DE ANÁLISIS AVANZADO
# =========================================================================

@dataset_bp.route("/datasets/<dataset_id>/analisis-radar", methods=["GET"])
def analisis_radar_route(dataset_id):
    """Devuelve datos para el gráfico Radar Comparativo."""
    try:
        data = generar_analisis_radar(dataset_id)
        return jsonify(data), 200
    except Exception as e:
        print(f"🚨 ERROR en analisis_radar_route: {e}")
        return jsonify({"error": str(e)}), 500

@dataset_bp.route("/datasets/<dataset_id>/sankey-flujo", methods=["GET"])
def sankey_flujo_route(dataset_id):
    """Devuelve datos para el diagrama Sankey."""
    try:
        data = generar_sankey_flujo(dataset_id)
        return jsonify(data), 200
    except Exception as e:
        print(f"🚨 ERROR en sankey_flujo_route: {e}")
        return jsonify({"error": str(e)}), 500

@dataset_bp.route("/datasets/<dataset_id>/grafico-burbujas", methods=["GET"])
def grafico_burbujas_route(dataset_id):
    """Devuelve datos para el gráfico de Burbujas."""
    try:
        data = generar_grafico_burbujas(dataset_id)
        return jsonify(data), 200
    except Exception as e:
        print(f"🚨 ERROR en grafico_burbujas_route: {e}")
        return jsonify({"error": str(e)}), 500

@dataset_bp.route("/datasets/<dataset_id>/analisis-sensibilidad", methods=["GET"])
def analisis_sensibilidad_route(dataset_id):
    """Devuelve datos para el análisis de Sensibilidad."""
    try:
        data = generar_analisis_sensibilidad(dataset_id)
        return jsonify(data), 200
    except Exception as e:
        print(f"🚨 ERROR en analisis_sensibilidad_route: {e}")
        return jsonify({"error": str(e)}), 500

@dataset_bp.route("/datasets/<dataset_id>/clustering", methods=["GET"])
def clustering_route(dataset_id):
    """Devuelve resultados del análisis de Clustering."""
    try:
        data = generar_clustering(dataset_id)
        return jsonify(data), 200
    except Exception as e:
        print(f"🚨 ERROR en clustering_route: {e}")
        return jsonify({"error": str(e)}), 500
    
# =========================================================================
# 4. NUEVO: RUTAS DE ANÁLISIS DE MERCADO
# =========================================================================

@dataset_bp.route("/datasets/<dataset_id>/segmentacion-mercado", methods=["GET"])
def segmentacion_mercado_route(dataset_id):
    """Devuelve datos de segmentación del mercado (ej. por precio)."""
    try:
        data = generar_segmentacion_mercado(dataset_id)
        return jsonify(data), 200
    except Exception as e:
        print(f"🚨 ERROR en segmentacion_mercado_route: {e}")
        # Devolver estructura vacía esperada por el frontend en caso de error
        return jsonify({"distribucion": [], "estadisticas": {}}), 500

@dataset_bp.route("/datasets/<dataset_id>/anomalias-precios", methods=["GET"])
def anomalias_precios_route(dataset_id):
    """Detecta y devuelve propiedades con precios anómalos (gangas/sobrevaloradas)."""
    try:
        data = detectar_anomalias_precios(dataset_id)
        return jsonify(data), 200
    except Exception as e:
        print(f"🚨 ERROR en anomalias_precios_route: {e}")
        # Devolver estructura vacía esperada
        return jsonify({"gangas": [], "sobrevaloradas": [], "resumen": {}}), 500

@dataset_bp.route("/datasets/<dataset_id>/score-inversion", methods=["GET"])
def score_inversion_route(dataset_id):
    """Calcula y devuelve un score de potencial de inversión para las propiedades."""
    try:
        data = calcular_score_inversion(dataset_id)
        return jsonify(data), 200
    except Exception as e:
        print(f"🚨 ERROR en score_inversion_route: {e}")
        # Devolver estructura vacía esperada
        return jsonify({"propiedades": [], "estadisticas": {}}), 500