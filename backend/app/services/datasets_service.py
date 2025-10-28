import pandas as pd
import io
from app.services.supabase_service import supabase # Asumimos que tienes el cliente Supabase inicializado
from datetime import datetime
from io import BytesIO
from typing import Tuple, Any

# =============================================================================
# 1. Rutas de Lectura (GET /datasets)
# =============================================================================

def obtener_lista_datasets() -> list:
    """
    Obtiene todos los registros de datasets de la tabla 'datasets'.
    """
    try:
        # Consulta a la tabla 'datasets' para obtener la lista.
        # Ordenamos por fecha de subida para mostrar los más recientes primero.
        res = supabase.table("datasets").select("*").order("fecha_subida", desc=True).execute()
        
        # El frontend espera que 'filas' y 'columnas' sean números.
        # Se asegura que la columna 'es_limpio' sea tratada como un booleano (si no es nula).
        data_list = []
        for item in res.data:
            data_list.append({
                'id': item['id'],
                'nombre': item['nombre'],
                'archivo_url': item['archivo_url'],
                'filas': int(item.get('filas', 0)),
                'columnas': int(item.get('columnas', 0)),
                'fecha_subida': item['fecha_subida'],
                'usuario_id': item['usuario_id'],
                'es_limpio': bool(item.get('es_limpio', False)) if item.get('es_limpio') is not None else False,
                'dataset_original_id': item.get('dataset_original_id')
            })
            
        return data_list
    except Exception as e:
        print(f"Error al obtener lista de datasets desde Supabase: {e}")
        # Es vital devolver una lista vacía en caso de error para evitar que el frontend falle.
        return []

# =============================================================================
# 2. Rutas de Creación (POST /datasets)
# =============================================================================

def subir_dataset_csv_metadata(data: dict) -> dict:
    """
    Registra la metadata de un nuevo dataset en la tabla 'datasets'.
    Asumimos que el archivo ya se subió al Storage de Supabase desde el frontend.
    """
    try:
        # Usamos obtener_dataframe_crudo (del análisis_service) para obtener 
        # las filas y columnas reales antes de guardar la metadata.
        # Esto requiere una modificación temporal en obtener_dataframe_crudo para 
        # aceptar la URL o una llamada al servicio de análisis, pero por simplicidad
        # aquí, usaremos los datos que envía el frontend y añadimos la fecha.
        
        # Simulamos la obtención de filas y columnas si no se proporcionan, 
        # aunque lo ideal es leer el CSV antes de esta llamada.
        
        metadata = {
            "nombre": data.get("nombre"),
            "archivo_url": data.get("archivo_url"),
            "usuario_id": data.get("usuario_id"),
            "filas": data.get("filas", 0),
            "columnas": data.get("columnas", 0),
            "fecha_subida": datetime.utcnow().isoformat(),
            "es_limpio": False,
        }
        
        # Insertar el nuevo registro en la base de datos
        res = supabase.table("datasets").insert([metadata]).execute()
        return res.data[0] # Devolvemos el registro insertado
        
    except Exception as e:
        print(f"Error al subir metadata: {e}")
        raise

# =============================================================================
# 3. Rutas de Eliminación (DELETE /datasets/:id)
# =============================================================================

def eliminar_dataset_por_id(dataset_id: str):
    """
    Elimina un registro de dataset de la tabla 'datasets'.
    Asumimos que el frontend ya eliminó el archivo del Storage de Supabase.
    """
    try:
        # Elimina el registro por ID
        supabase.table("datasets").delete().eq("id", dataset_id).execute()
        # Se podría añadir lógica para eliminar registros relacionados (cascada)
    except Exception as e:
        print(f"Error al eliminar dataset {dataset_id}: {e}")
        raise

# =============================================================================
# 4. Descarga (GET /datasets/:id/descargar)
# =============================================================================

def obtener_archivo_dataset(dataset_id: str) -> Tuple[BytesIO, str]:
    """
    Descarga el archivo CSV asociado al dataset desde Supabase Storage.
    """
    try:
        # 1. Obtener la URL del archivo
        res = supabase.table("datasets").select("nombre, archivo_url").eq("id", dataset_id).single().execute()
        if not res.data:
            raise ValueError(f"Dataset {dataset_id} no encontrado.")
            
        nombre_dataset = res.data["nombre"]
        archivo_url = res.data["archivo_url"]
        
        # 2. Extraer la ruta del archivo del Storage (ej: 'datasets/user_id/timestamp_file.csv')
        # Esto es necesario si usamos el cliente de Python para descargar en lugar de requests
        path_in_storage = archivo_url.split('/storage/v1/object/public/datasets/')[-1]

        # 3. Descargar el archivo del Storage
        # Nota: Usamos el método de descarga del cliente Supabase
        storage_res = supabase.storage.from_("datasets").download(path_in_storage)
        
        if not storage_res:
             raise Exception("Fallo al descargar el archivo del Storage.")

        # 4. Envolver en BytesIO para que Flask pueda enviarlo (send_file)
        archivo_buffer = BytesIO(storage_res)
        archivo_buffer.seek(0)
        
        return archivo_buffer, nombre_dataset
        
    except Exception as e:
        print(f"Error al obtener archivo para {dataset_id}: {e}")
        raise