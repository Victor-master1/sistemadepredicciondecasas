import joblib
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import io
import json
import datetime # <- CAMBIO: Importar datetime para calcular antigüedad
from app.services.supabase_service import supabase
from app.services.entrenamiento_service import NeuralNet
from app.services.entrenamiento_service import ARTEFACTOS_BUCKET_NAME

# --- INICIO DE LA NUEVA LÓGICA DE NEGOCIO ---

def _calcular_metricas_completas(
    precio_predicho: float,
    datos_casa: dict,
    metricas_modelo: dict
) -> dict:
    """
    Calcula el análisis financiero completo basado en heurísticas.
    Esta es la función que genera el JSON complejo para el frontend.
    """
    print("-> Calculando análisis financiero completo (heurísticas)...")

    # --- 1. Extraer y limpiar datos de entrada ---
    try:
        # Usamos .get() con valores por defecto para evitar KeyErrors si faltan
        # El frontend envía todo como float (por parseFloat), incluso 'zona'
        ano_construccion = int(float(datos_casa.get('ano_construccion', 2000)))
        # Convertir zona a string para comparaciones
        zona = str(int(float(datos_casa.get('zona', 3)))) 
    except Exception as e:
        print(f"⚠️ Advertencia al leer datos_casa para heurísticas: {e}. Usando defaults.")
        ano_construccion = 2000
        zona = '3'
    
    antiguedad = datetime.datetime.now().year - ano_construccion

    # --- 2. Definir Heurísticas Base ---
    # Estas son reglas de negocio simples. ¡Puedes mejorarlas!
    
    # Tasas base
    tasa_reval_base = 0.03 # 3%
    tasa_alquiler_base = 0.05 # 5%
    tasa_mant_base = 0.015 # 1.5%

    # Modificadores por ZONA (ej. '1' es premium, '5' es económico)
    if zona == '1':
        tasa_reval_base += 0.015 # 4.5%
        tasa_alquiler_base += 0.01 # 6%
    elif zona == '5':
        tasa_reval_base -= 0.01 # 2%
        tasa_alquiler_base -= 0.01 # 4%
        
    # Modificadores por ANTIGÜEDAD
    if antiguedad < 5: # Nuevo
        tasa_reval_base += 0.005 # Más revalorización
    elif antiguedad > 30: # Viejo
        tasa_reval_base -= 0.005
        tasa_alquiler_base -= 0.005
        tasa_mant_base = 0.025 # Mantenimiento más caro

    # --- 3. Calcular Revalorización ---
    tasa_r = tasa_reval_base
    v_1 = precio_predicho * (1 + tasa_r)**1
    v_3 = precio_predicho * (1 + tasa_r)**3
    v_5 = precio_predicho * (1 + tasa_r)**5
    v_10 = precio_predicho * (1 + tasa_r)**10
    
    revalorizacion = {
        "tasa_anual": round(tasa_r * 100, 2),
        "valor_1_ano": round(v_1, 2),
        "incremento_1_ano": round(v_1 - precio_predicho, 2),
        "valor_3_anos": round(v_3, 2),
        "incremento_3_anos": round(v_3 - precio_predicho, 2),
        "valor_5_anos": round(v_5, 2),
        "incremento_5_anos": round(v_5 - precio_predicho, 2),
        "valor_10_anos": round(v_10, 2),
        "incremento_10_anos": round(v_10 - precio_predicho, 2),
    }

    # --- 4. Calcular Costos y Rentabilidad ---
    costo_anual_mant = precio_predicho * tasa_mant_base
    costos_mantenimiento = {
        "costo_anual": round(costo_anual_mant, 2),
        "costo_mensual": round(costo_anual_mant / 12, 2),
        "porcentaje_valor": round(tasa_mant_base * 100, 2)
    }

    ingreso_anual_alq = precio_predicho * tasa_alquiler_base
    ingreso_neto_anual = ingreso_anual_alq - costo_anual_mant
    roi_anual = (ingreso_neto_anual / precio_predicho) * 100 if precio_predicho > 0 else 0
    anos_recuperacion = precio_predicho / ingreso_neto_anual if ingreso_neto_anual > 0 else 999
    
    rentabilidad_alquiler = {
        "ingreso_mensual": round(ingreso_anual_alq / 12, 2),
        "ingreso_anual": round(ingreso_anual_alq, 2),
        "roi_anual": round(roi_anual, 2),
        "gastos_anuales": round(costo_anual_mant, 2),
        "ingreso_neto_anual": round(ingreso_neto_anual, 2),
        "anos_recuperacion": round(anos_recuperacion, 1)
    }
    
    # --- 5. Calcular Riesgo ---
    factores_riesgo = []
    score_riesgo_num = 50 # Base media
    
    if zona == '1':
        factores_riesgo.append({"factor": "Ubicación", "impacto": "positivo"})
        score_riesgo_num -= 20
    elif zona == '5':
        factores_riesgo.append({"factor": "Ubicación", "impacto": "negativo"})
        score_riesgo_num += 20
        
    if antiguedad < 10:
        factores_riesgo.append({"factor": "Antigüedad", "impacto": "positivo"})
        score_riesgo_num -= 10
    elif antiguedad > 30:
        factores_riesgo.append({"factor": "Antigüedad", "impacto": "negativo"})
        score_riesgo_num += 10
    
    if not factores_riesgo:
        factores_riesgo.append({"factor": "Mercado", "impacto": "neutro"})

    score_riesgo_num = max(10, min(90, score_riesgo_num)) # Limitar score 10-90

    if score_riesgo_num <= 33:
        nivel_riesgo = "Bajo"
        color_riesgo = "success"
    elif score_riesgo_num <= 66:
        nivel_riesgo = "Medio"
        color_riesgo = "warning"
    else:
        nivel_riesgo = "Alto"
        color_riesgo = "error"

    riesgo_inversion = {
        "score_riesgo": score_riesgo_num,
        "nivel_riesgo": nivel_riesgo,
        "color_riesgo": color_riesgo,
        "factores_riesgo": factores_riesgo
    }

    # --- 6. Calcular Tiempo de Venta / Demanda ---
    score_demanda_num = 50 # Base media
    dias_min = 90
    dias_max = 120

    if precio_predicho < 150000: # Barato
        score_demanda_num = 80
        dias_min, dias_max = 45, 75
    elif precio_predicho > 600000: # Caro
        score_demanda_num = 30
        dias_min, dias_max = 120, 180

    if score_demanda_num >= 70:
        nivel_demanda = "Alta"
    elif score_demanda_num >= 40:
        nivel_demanda = "Media"
    else:
        nivel_demanda = "Baja"
        
    tiempo_venta = {
        "dias_minimo": dias_min,
        "dias_maximo": dias_max,
        "nivel_demanda": nivel_demanda,
        "score_demanda": score_demanda_num
    }

    # --- 7. Otros Valores ---
    # Confianza basada en el R2 del modelo
    confianza = (metricas_modelo.get('r2', 0.8) * 100)
    confianza = round(max(60.0, min(98.0, confianza + 10)), 1) # R2 de 0.8 -> 90% confianza

    tendencia_precio = "Estable"
    if zona == '1':
        tendencia_precio = "subida"
    elif zona == '5' and antiguedad > 20:
        tendencia_precio = "bajada"
        
    tiempo_vida_estimado = max(10, 80 - antiguedad)

    # --- 8. Consolidar Resultados ---
    resultado_final = {
        "tiempo_vida_estimado": tiempo_vida_estimado,
        "tendencia_precio": tendencia_precio,
        "confianza": confianza,
        "tiempo_venta": tiempo_venta,
        "revalorizacion": revalorizacion,
        "rentabilidad_alquiler": rentabilidad_alquiler,
        "riesgo_inversion": riesgo_inversion,
        "costos_mantenimiento": costos_mantenimiento
    }

    print("-> Análisis financiero completado.")
    return resultado_final

# --- FIN DE LA NUEVA LÓGICA ---


def cargar_artefacto_desde_storage(path_in_bucket: str):
    """Descarga un artefacto (modelo, scaler) desde Supabase Storage."""
    try:
        print(f"   -> Descargando artefacto desde: {ARTEFACTOS_BUCKET_NAME}/{path_in_bucket}")
        storage_response = supabase.storage.from_(ARTEFACTOS_BUCKET_NAME).download(path_in_bucket)
        if not storage_response:
            raise FileNotFoundError(f"No se pudo descargar el artefacto en la ruta: {path_in_bucket}")

        buffer = io.BytesIO(storage_response)
        buffer.seek(0)
        
        if path_in_bucket.endswith(".joblib"):
            artefacto = joblib.load(buffer)
            print(f"   -> Artefacto joblib cargado: {type(artefacto)}")
            return artefacto
        elif path_in_bucket.endswith(".pth"):
            print(f"   -> State dict de PyTorch descargado.")
            return buffer
        else:
            raise ValueError(f"Tipo de archivo de artefacto no reconocido: {path_in_bucket}")

    except Exception as e:
        print(f"🔥🔥 Error al descargar/cargar artefacto {path_in_bucket}: {e}")
        raise FileNotFoundError(f"Fallo al cargar {path_in_bucket}: {e}")


def realizar_prediccion_casa(experimento_id: str, datos_casa: dict) -> dict:
    """
    Carga el modelo, preprocesa, predice y llama a la lógica de negocio
    para devolver el análisis completo.
    """
    print(f"🚀 Predicción solicitada para experimento: {experimento_id}")

    # --- 1. Obtener información del experimento (AHORA INCLUYE 'metricas') ---
    exp_res = supabase.table("experimentos").select(
        "configuracion, artefactos_info, importancia_features, metricas" # <- CAMBIO: añadido 'metricas'
    ).eq("id", experimento_id).single().execute()
    
    if not exp_res.data:
        raise ValueError(f"Experimento con ID '{experimento_id}' no encontrado.")

    experimento_data = exp_res.data
    try:
        configuracion = json.loads(experimento_data.get('configuracion', '{}'))
        artefactos_info = json.loads(experimento_data.get('artefactos_info', '{}'))
        importancia_features_guardada = json.loads(experimento_data.get('importancia_features', '[]')) or []
        metricas_modelo = json.loads(experimento_data.get('metricas', '{}')) # <- CAMBIO: cargar metricas

        if not artefactos_info or 'error' in artefactos_info or 'urls' not in artefactos_info:
            raise ValueError(f"Información de artefactos inválida o faltante para el experimento {experimento_id}. Error: {artefactos_info.get('error', 'No hay URLs')}")

    except json.JSONDecodeError as json_err:
        raise ValueError(f"Error al parsear información del experimento {experimento_id}: {json_err}")

    print("-> Información del experimento y artefactos cargada.")
    model_type = artefactos_info.get('model_type')
    model_path = artefactos_info.get('model_path')
    scaler_x_path = artefactos_info.get('scaler_x_path')
    scaler_y_path = artefactos_info.get('scaler_y_path')
    le_path = artefactos_info.get('label_encoder_path')
    columnas_entrenamiento = artefactos_info.get('columns', []) # <- El frontend usa 'columnas_entrada' de la tabla, esto es para el scaler

    if not model_path or not scaler_x_path or not columnas_entrenamiento:
         raise ValueError("Falta información esencial de artefactos (modelo, scaler_x o columnas).")

    # --- 2. Cargar artefactos desde Supabase Storage ---
    print("-> Cargando artefactos desde Storage...")
    scaler_x = cargar_artefacto_desde_storage(scaler_x_path)
    scaler_y = cargar_artefacto_desde_storage(scaler_y_path) if scaler_y_path else None
    label_encoder = cargar_artefacto_desde_storage(le_path) if le_path else None

    modelo = None
    if model_type == 'red_neuronal':
        state_dict_buffer = cargar_artefacto_desde_storage(model_path)
        nn_input_size = artefactos_info.get('nn_input_size')
        nn_num_classes = artefactos_info.get('nn_num_classes')
        nn_is_regression = artefactos_info.get('nn_is_regression')
        if None in [nn_input_size, nn_num_classes, nn_is_regression]:
            raise ValueError("Falta información para reconstruir la Red Neuronal.")

        modelo = NeuralNet(nn_input_size, nn_num_classes, is_regression=nn_is_regression)
        modelo.load_state_dict(torch.load(state_dict_buffer, map_location=torch.device('cpu')))
        modelo.eval()
        print("-> Modelo PyTorch cargado y reconstruido.")
    elif model_type == 'regresion':
        modelo = cargar_artefacto_desde_storage(model_path)
        print("-> Modelo Sklearn cargado.")
    else:
        raise ValueError(f"Tipo de modelo '{model_type}' no soportado para predicción.")

    # --- 3. Preprocesar datos de entrada ---
    print("-> Preprocesando datos de entrada...")
    try:
        # El frontend envía 'datos_casa' con las claves correctas
        # Necesitamos asegurarnos de que el DataFrame tenga el mismo orden de columnas que 'columnas_entrenamiento'
        datos_procesados = {}
        columnas_scaler = scaler_x.feature_names_in_ if hasattr(scaler_x, 'feature_names_in_') else columnas_entrenamiento
        
        for col in columnas_scaler:
            if col in datos_casa:
                try:
                    # El frontend ya envía floats, pero re-aseguramos
                    datos_procesados[col] = pd.to_numeric(datos_casa[col])
                except ValueError:
                     raise ValueError(f"Valor inválido para la columna '{col}': '{datos_casa[col]}'. Se esperaba un número.")
            else:
                 # Si una columna que el scaler espera no viene, usamos 0
                 print(f"⚠️ Advertencia: Columna '{col}' (esperada por scaler) no encontrada. Usando 0.")
                 datos_procesados[col] = 0

        input_df = pd.DataFrame([datos_procesados], columns=columnas_scaler) # Forzar orden del scaler
        print("   DataFrame de entrada (para scaler) creado:")
        print(input_df.head().to_string())

        input_scaled = scaler_x.transform(input_df)
        print("   Datos de entrada escalados.")

    except Exception as preproc_err:
         raise ValueError(f"Error al preprocesar los datos de entrada: {preproc_err}")

    # --- 4. Realizar Predicción ---
    print("-> Realizando predicción...")
    prediccion_final = None

    if model_type == 'red_neuronal':
        input_tensor = torch.tensor(input_scaled, dtype=torch.float32)
        with torch.no_grad():
            output_scaled = modelo(input_tensor)
            pred_scaled = output_scaled.numpy().flatten()[0] # Asumir regresión

            if scaler_y:
                 prediccion_final = scaler_y.inverse_transform([[pred_scaled]])[0][0]
                 print(f"   Predicción NN escalada: {pred_scaled}, Desescalada: {prediccion_final}")
            else:
                 prediccion_final = pred_scaled
                 print(f"   Predicción NN (no desescalada): {prediccion_final}")

    elif model_type == 'regresion':
        prediccion_final = modelo.predict(input_scaled)[0]
        # Asumir que 'y' no se escaló para modelos sklearn
        print(f"   Predicción Sklearn: {prediccion_final}")

    if prediccion_final is None:
         raise Exception("La predicción no pudo ser calculada.")

    # --- 5. Preparar Resultado (REEMPLAZANDO PLACEHOLDERS) ---
    print("-> Preparando resultado final...")
    
    # Formatear factores importantes (tu lógica original estaba bien)
    factores_importantes_formateados = [
        # El frontend espera 'nombre' e 'impacto'
        {"nombre": f['feature'], "impacto": round(max(0, min(100, f['importancia'] * 100)), 1)}
        for f in importancia_features_guardada[:5] if f['importancia'] > 0.01
    ] if importancia_features_guardada else []

    # --- CAMBIO: Llamar a la nueva función de lógica de negocio ---
    metricas_completas = _calcular_metricas_completas(
        float(prediccion_final),
        datos_casa,
        metricas_modelo
    )

    # --- 6. Construir el JSON de respuesta final ---
    resultado = {
        "precio_predicho": round(float(prediccion_final), 2),
        "factores_importantes": factores_importantes_formateados,
        
        # Unir todas las métricas calculadas (revalorizacion, riesgo, etc.)
        **metricas_completas 
    }

    print(f"✅ Predicción y análisis completados.")
    return resultado

