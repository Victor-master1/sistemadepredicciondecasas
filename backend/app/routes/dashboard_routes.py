from flask import Blueprint, jsonify
from app.services.supabase_service import supabase
from datetime import datetime
import pytz # Asegúrate de haber hecho 'pip install pytz'

# --- Blueprint para las rutas del Dashboard ---
dashboard_bp = Blueprint('dashboard_bp', __name__)

@dashboard_bp.route('/estadisticas', methods=['GET'])
def obtener_estadisticas():
    """
    Calcula y devuelve las estadísticas clave para el dashboard.
    """
    try:
        # 1. Contar total de datasets
        datasets_resp = supabase.table('datasets').select('id', count='exact').execute()
        total_datasets = datasets_resp.count if datasets_resp.count is not None else 0

        # 2. Contar total de experimentos y obtener sus datos
        experimentos_resp = supabase.table('experimentos').select('id, estado, fecha_creacion', count='exact').execute()
        total_experimentos = experimentos_resp.count if experimentos_resp.count is not None else 0
        
        experimentos_data = experimentos_resp.data if experimentos_resp.data else []

        # 3. Contar experimentos "activos" (los que están en proceso)
        experimentos_activos = len([
            exp for exp in experimentos_data 
            if exp.get('estado') not in ['completado', 'error']
        ])

        # 4. Encontrar la fecha del último entrenamiento
        ultimo_entrenamiento = "N/A"  # Valor por defecto
        if experimentos_data:
            # Ordenamos los experimentos por fecha para encontrar el más reciente
            experimentos_data.sort(key=lambda x: x['fecha_creacion'], reverse=True)
            
            # --- SOLUCIÓN: Enviar la fecha en formato ISO ---
            # Enviamos la fecha original de la base de datos.
            # El frontend (JavaScript) entiende este formato universal perfectamente.
            ultimo_entrenamiento = experimentos_data[0]['fecha_creacion']
            # --- FIN DE LA SOLUCIÓN ---

        # Estructura final que el frontend de React necesita
        estadisticas = {
            'total_datasets': total_datasets,
            'total_experimentos': total_experimentos,
            'experimentos_activos': experimentos_activos,
            'ultimo_entrenamiento': ultimo_entrenamiento
        }
        
        return jsonify(estadisticas), 200

    except Exception as e:
        print(f"🚨 ERROR en /api/estadisticas: {e}")
        return jsonify({"error": "No se pudieron obtener las estadísticas"}), 500

