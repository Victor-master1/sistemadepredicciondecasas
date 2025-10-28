import pandas as pd
import numpy as np
from io import BytesIO
from datetime import datetime
from app.services.supabase_service import supabase
# Importamos OBTENER_DATAFRAME_CRUDO, que SÍ se necesita
from app.services.analisis_service import obtener_dataframe_crudo 
# (Se eliminó la importación rota de 'NULL_REPRESENTATION')

def limpiar_dataset(dataset_id: str, operaciones: dict):
    """
    Aplica las operaciones de limpieza y devuelve el resultado completo 
    con las estadísticas (ResultadoLimpieza).
    """
    try:
        df_original = obtener_dataframe_crudo(dataset_id)
        if df_original.empty:
            raise ValueError("El dataset original está vacío, no se puede limpiar.")
            
        df_limpio = df_original.copy()
        filas_originales = len(df_original)

        # 1. Contar estadísticas originales
        duplicados_originales = df_original.duplicated().sum()
        
        # 2. Aplicar operaciones de limpieza
        duplicados_eliminados = 0
        if operaciones.get('eliminar_duplicados'):
            filas_antes_dup = len(df_limpio)
            df_limpio.drop_duplicates(inplace=True)
            duplicados_eliminados = filas_antes_dup - len(df_limpio)
        
        nulos_eliminados = 0
        if operaciones.get('eliminar_nulos'):
            nulos_antes_drop = df_limpio.isnull().sum().sum()
            df_limpio.dropna(inplace=True)
            nulos_eliminados = nulos_antes_drop - df_limpio.isnull().sum().sum()
            
        # 3. Calcular estadísticas finales (con la corrección de división por cero)
        filas_limpias = len(df_limpio)
        filas_eliminadas = filas_originales - filas_limpias
        
        if filas_originales > 0:
            porcentaje_datos_eliminados = (filas_eliminadas / filas_originales) * 100
        else:
            porcentaje_datos_eliminados = 0.0
        
        estadisticas = {
            "filas_originales": filas_originales,
            "filas_limpias": filas_limpias,
            "filas_eliminadas": filas_eliminadas,
            "porcentaje_datos_eliminados": round(porcentaje_datos_eliminados, 2),
            "nulos_por_columna": df_limpio.isnull().sum().astype(int).to_dict(),
            "total_nulos": int(df_limpio.isnull().sum().sum()),
            "duplicados_detectados": int(df_limpio.duplicated().sum()),
            "columnas_eliminadas": [], 
            "nulos_eliminados": int(nulos_eliminados),
            "duplicados_eliminados": int(duplicados_eliminados),
        }
        
        # 4. Guardar y registrar el nuevo dataset limpio
        csv_buffer = BytesIO()
        df_limpio.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)

        # --- Interacción Robusta con Supabase ---
        
        # 4.1. Obtener info del dataset original
        res_info = supabase.table("datasets").select("nombre, usuario_id").eq("id", dataset_id).single().execute()
        if not res_info.data:
            raise ValueError(f"No se encontró el dataset original con ID {dataset_id}")
        dataset_info = res_info.data

        nombre_base = dataset_info["nombre"].rsplit('.', 1)[0]
        nombre_archivo = f"limpios/{dataset_info['usuario_id']}/{nombre_base}_limpio_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv"

        # 4.2. Subir el nuevo archivo CSV limpio al Storage
        bucket = supabase.storage.from_("datasets")
        upload_res = bucket.upload(nombre_archivo, csv_buffer.getvalue())
        if not upload_res:
             raise Exception(f"Error al subir el archivo limpio a Supabase Storage: {upload_res.get('error', 'Error desconocido')}")
        
        archivo_url_res = bucket.get_public_url(nombre_archivo)
        if not archivo_url_res:
            raise Exception("No se pudo obtener la URL pública del archivo limpio.")
        
        # 4.3. Registrar el nuevo dataset en la tabla de Supabase (PostgreSQL)
        nuevo_dataset_data = {
            "nombre": dataset_info["nombre"] + " (Limpio)",
            "archivo_url": archivo_url_res,
            "filas": filas_limpias,
            "columnas": len(df_limpio.columns),
            "fecha_subida": datetime.utcnow().isoformat(),
            "usuario_id": dataset_info["usuario_id"],
            "es_limpio": True,
            "dataset_original_id": dataset_id
        }
        
        insert_response = supabase.table("datasets").insert(nuevo_dataset_data).execute()
        if not insert_response.data or len(insert_response.data) == 0:
            raise Exception(f"Error al insertar el registro del dataset limpio en la base de datos: {insert_response.get('error', 'No se devolvieron datos')}")
            
        dataset_limpio_creado = insert_response.data[0]

        # 5. Devolver la respuesta completa esperada por el frontend
        return {
            "mensaje": "Limpieza completada exitosamente.",
            "dataset_limpio_id": dataset_limpio_creado['id'],
            "filas_resultantes": filas_limpias,
            "archivo_url": archivo_url_res,
            "estadisticas": estadisticas
        }

    except Exception as e:
        # Imprime el error en la consola de Flask para depuración
        print(f"🚨 ERROR CRÍTICO en limpiar_dataset: {e}") 
        # Asegúrate de que los errores se lancen para que el route los capture
        raise

