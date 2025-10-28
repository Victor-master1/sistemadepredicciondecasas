import pandas as pd
import io
import requests
import numpy as np
from app.services.supabase_service import supabase
# Importar KMeans para clustering
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer # Para manejar nulos
# Importar IsolationForest para detección de anomalías
from sklearn.ensemble import IsolationForest

# Constante para representar nulos en la vista previa
NULL_STRING = "[NULL]"

# =============================================================================
# 1️⃣ Obtener DataFrame "Crudo"
# =============================================================================
def obtener_dataframe_crudo(dataset_id: str) -> pd.DataFrame:
    """
    Descarga el archivo CSV desde Supabase y lo carga en un DataFrame de Pandas,
    manteniendo los datos en su estado original (con valores nulos).
    """
    try:
        dataset_res = supabase.table("datasets").select("archivo_url").eq("id", dataset_id).single().execute()
        if not dataset_res.data:
            raise ValueError(f"❌ Dataset con ID '{dataset_id}' no encontrado.")

        archivo_url = dataset_res.data.get("archivo_url")
        if not archivo_url:
            raise ValueError("❌ El registro del dataset no tiene una URL de archivo.")

        resp = requests.get(archivo_url)
        resp.raise_for_status() # Lanza error si la descarga falla

        # Usar io.BytesIO para leer el contenido binario directamente
        df = pd.read_csv(io.BytesIO(resp.content))

        if df.empty:
            print(f"⚠️ Advertencia: El dataset {dataset_id} está vacío o no se pudo leer correctamente.")
            # Considerar devolver un DataFrame vacío si es preferible a un error
            # return pd.DataFrame()

        return df

    except requests.exceptions.RequestException as req_err:
        print(f"🚨 [ERROR] de red al descargar {archivo_url}: {req_err}")
        raise ValueError(f"No se pudo descargar el archivo del dataset desde {archivo_url}.") from req_err
    except pd.errors.EmptyDataError:
        print(f"🚨 [ERROR] El archivo CSV {archivo_url} está vacío.")
        raise ValueError(f"El archivo CSV para el dataset {dataset_id} está vacío.") from pd.errors.EmptyDataError
    except Exception as e:
        print(f"🚨 [ERROR] inesperado en obtener_dataframe_crudo para {dataset_id}: {e}")
        raise # Re-lanzar para que la ruta lo maneje

# =============================================================================
# 2️⃣ Funciones de Análisis Básico
# =============================================================================

def estadisticas_dataset(df: pd.DataFrame) -> dict:
    """ Calcula estadísticas generales sobre el DataFrame. """
    try:
        total_filas = len(df)
        total_columnas = df.shape[1]
        if total_filas == 0 or total_columnas == 0: return {"total_filas": 0, "total_columnas": 0, "total_nulos": 0, "total_duplicados": 0, "porcentaje_nulos": 0.0}
        total_nulos = int(df.isnull().sum().sum())
        total_duplicados = int(df.duplicated().sum())
        denominador = total_filas * total_columnas
        porcentaje_nulos = (total_nulos / denominador * 100) if denominador > 0 else 0.0
        return {"total_filas": total_filas, "total_columnas": total_columnas, "total_nulos": total_nulos, "total_duplicados": total_duplicados, "porcentaje_nulos": round(porcentaje_nulos, 2)}
    except Exception as e:
        print(f"🚨 [ERROR] en estadisticas_dataset: {e}")
        raise RuntimeError("No se pudieron calcular las estadísticas del dataset.") from e

def obtener_columnas(df: pd.DataFrame) -> list:
    """ Obtiene información detallada de cada columna. """
    try:
        total_filas = len(df)
        info_columnas = []
        if total_filas == 0: return info_columnas
        nulos_por_columna = df.isnull().sum()
        for c in df.columns:
            col_info = {"nombre": c, "tipo": str(df[c].dtype), "valores_nulos": int(nulos_por_columna[c]), "valores_unicos": int(df[c].nunique())}
            # CORRECCIÓN: Verificar np.isfinite para evitar error con infinito en std
            if pd.api.types.is_numeric_dtype(df[c]) and df[c].notna().any():
                col_min = df[c].min()
                col_max = df[c].max()
                col_mean = df[c].mean()
                col_std = df[c].std()
                col_info["min"] = float(col_min) if np.isfinite(col_min) else None
                col_info["max"] = float(col_max) if np.isfinite(col_max) else None
                col_info["promedio"] = float(col_mean) if np.isfinite(col_mean) else None
                col_info["desviacion"] = float(col_std) if pd.notna(col_std) and np.isfinite(col_std) else None # Verificar NaN y Infinito
            info_columnas.append(col_info)
        return info_columnas
    except Exception as e:
        print(f"🚨 [ERROR] en obtener_columnas: {e}")
        raise RuntimeError("No se pudo obtener la información de las columnas.") from e

def obtener_vista_previa_paginada(df: pd.DataFrame, pagina: int = 1, filas_por_pagina: int = 20) -> dict:
    """ Obtiene una porción paginada y reemplaza NaN/NaT con NULL_STRING. """
    try:
        total_filas = len(df)
        total_paginas = max(1, (total_filas + filas_por_pagina - 1) // filas_por_pagina)
        pagina_actual = max(1, min(pagina, total_paginas))
        inicio = (pagina_actual - 1) * filas_por_pagina
        fin = inicio + filas_por_pagina
        df_pagina = df.iloc[inicio:fin]
        # CORRECCIÓN: Manejar Infinito también
        df_temp = df_pagina.replace([np.inf, -np.inf], np.nan)
        df_pagina_str = df_temp.astype(object).fillna(NULL_STRING)
        datos = df_pagina_str.to_dict(orient='records')
        return {"datos": datos, "pagina_actual": pagina_actual, "total_paginas": total_paginas, "total_filas": total_filas}
    except Exception as e:
        print(f"🚨 [ERROR] en obtener_vista_previa_paginada: {e}")
        return {"datos": [], "pagina_actual": 1, "total_paginas": 1, "total_filas": 0, "error": str(e)}

def calcular_correlacion(df: pd.DataFrame) -> dict:
    """ Calcula la matriz de correlación para columnas numéricas. """
    try:
        df_numerico = df.select_dtypes(include='number')
        if df_numerico.empty: return {"variables": [], "matriz": []}
        # CORRECCIÓN: Eliminar columnas con varianza cero antes de corr()
        df_numerico_var = df_numerico.loc[:, df_numerico.std() > 1e-9]
        if df_numerico_var.empty: return {"variables": [], "matriz": []}
        matriz_corr = df_numerico_var.corr().round(4).fillna(0)
        return {"variables": matriz_corr.columns.tolist(), "matriz": matriz_corr.values.tolist()}
    except Exception as e:
        print(f"🚨 [ERROR] en calcular_correlacion: {e}")
        raise RuntimeError("No se pudo calcular la matriz de correlación.") from e

def distribucion_clases(df: pd.DataFrame, columna_clase: str = None) -> list:
    """ Calcula la distribución de una columna (última por defecto). """
    try:
        if df.empty: return []
        if columna_clase is None:
             if df.shape[1] == 0: return []
             columna_clase = df.columns[-1]
        elif columna_clase not in df.columns: raise ValueError(f"La columna '{columna_clase}' no existe.")
        distribucion = df[columna_clase].value_counts()
        return [{"clase": str(clase), "cantidad": int(cantidad)} for clase, cantidad in distribucion.items()]
    except Exception as e:
        print(f"🚨 [ERROR] en distribucion_clases: {e}")
        raise RuntimeError("No se pudo calcular la distribución de clases.") from e

# =============================================================================
# 3️⃣ Funciones de Visualización (Exploratorio Básico)
# =============================================================================
def generar_histograma(df: pd.DataFrame, columna: str = 'precio', bins: int = 10) -> list:
    try:
        if columna not in df.columns or not pd.api.types.is_numeric_dtype(df[columna]):
             raise ValueError(f"Columna '{columna}' no es numérica o no existe.")
        # CORRECCIÓN: Filtrar NaN y Infinito antes de histograma
        valores_validos = df[columna].replace([np.inf, -np.inf], np.nan).dropna()
        if valores_validos.empty: return []
        counts, bin_edges = np.histogram(valores_validos, bins=bins)
        data = []
        for i in range(len(counts)):
             rango = f"{bin_edges[i]:.0f}-{bin_edges[i+1]:.0f}"
             data.append({"rango": rango, "frecuencia": int(counts[i])})
        return data
    except Exception as e:
        print(f"Error en generar_histograma: {e}")
        return []

def generar_boxplot(df: pd.DataFrame, col_grupo: str = 'zona', col_valor: str = 'precio') -> list:
    """ Genera datos JSON para boxplots (usando cuantiles) agrupados por `col_grupo`. """
    print(f"-> generando_boxplot (grupo='{col_grupo}', valor='{col_valor}')") # Log para depuración
    try:
        # Verificar existencia de columnas
        if col_grupo not in df.columns or col_valor not in df.columns:
             print(f"   ⚠️ Faltan columnas '{col_grupo}' o '{col_valor}'.")
             return [] # Devolver vacío si faltan columnas

        # Verificar tipo numérico de la columna de valor
        if not pd.api.types.is_numeric_dtype(df[col_valor]):
             print(f"   ⚠️ Columna '{col_valor}' no es numérica.")
             return [] # Devolver vacío si el valor no es numérico

        # Crear copia limpia: quitar NaN/Inf en AMBAS columnas y asegurar tipos correctos
        df_clean = df[[col_grupo, col_valor]].copy()
        df_clean[col_valor] = pd.to_numeric(df_clean[col_valor], errors='coerce') # Forzar numérico
        df_clean = df_clean.replace([np.inf, -np.inf], np.nan).dropna() # Quitar Inf y NaN

        if df_clean.empty:
            print("   ⚠️ No quedan datos válidos después de limpiar NaN/Inf.")
            return [] # Devolver vacío si no hay datos limpios

        # Calcular cuantiles por grupo
        grouped = df_clean.groupby(col_grupo)[col_valor]
        # Verificar si hay grupos resultantes
        if grouped.ngroups == 0:
             print("   ⚠️ No se formaron grupos válidos (quizás col_grupo era todo NaN?).")
             return []

        stats = grouped.quantile([0, 0.25, 0.5, 0.75, 1]).unstack(level=1)

        # Verificar si stats está vacío (podría pasar si todos los grupos tenían < 1 dato válido)
        if stats.empty:
             print("   ⚠️ El cálculo de cuantiles resultó vacío (grupos sin suficientes datos?).")
             return []

        stats.columns = ['min', 'q1', 'mediana', 'q3', 'max'] # Renombrar columnas

        # Rellenar posibles NaNs en stats (si algún grupo tenía pocos datos para un cuantil)
        stats.fillna(0, inplace=True)

        # Formatear para Recharts (BarChart horizontal apilado)
        # CORRECCIÓN: asegurar que la columna de grupo se llame 'zona' en el resultado final
        data = stats.reset_index().rename(columns={col_grupo: 'zona'}).to_dict('records')

        print(f"   -> Boxplot data generado para {len(data)} zonas.") # Log éxito
        return data

    except Exception as e:
        print(f"🚨 ERROR en generar_boxplot: {e}")
        return [] # Devolver vacío en cualquier error inesperado


def generar_serie_temporal(df: pd.DataFrame, col_fecha: str = 'fecha', col_valor: str = 'precio') -> list:
    try:
        if col_fecha not in df.columns or col_valor not in df.columns: raise ValueError(f"Columnas '{col_fecha}' o '{col_valor}' no encontradas.")
        df_copy = df.copy()
        try: df_copy[col_fecha] = pd.to_datetime(df_copy[col_fecha], errors='coerce') # Coerce errores a NaT
        except Exception as date_err: raise ValueError(f"No se pudo convertir '{col_fecha}' a fecha: {date_err}")
        # CORRECCIÓN: Filtrar NaT y NaN/Inf en valor antes de agrupar
        df_copy.dropna(subset=[col_fecha, col_valor], inplace=True)
        df_copy = df_copy[np.isfinite(df_copy[col_valor])]
        if df_copy.empty: return []
        df_copy['ano'] = df_copy[col_fecha].dt.year
        serie = df_copy.groupby('ano')[col_valor].mean().round(2)
        data = serie.reset_index().rename(columns={col_valor: 'precio_promedio'}).to_dict('records')
        return data
    except Exception as e:
        print(f"Error en generar_serie_temporal: {e}")
        return []

def generar_mapa_calor(df: pd.DataFrame, col_grupo: str = 'zona', col_valor: str = 'precio') -> list:
    try:
        if col_grupo not in df.columns or col_valor not in df.columns: raise ValueError(f"Columnas '{col_grupo}' o '{col_valor}' no encontradas.")
        if not pd.api.types.is_numeric_dtype(df[col_valor]): raise ValueError(f"Columna '{col_valor}' debe ser numérica.")
        # CORRECCIÓN: Filtrar NaN/Inf antes de agrupar
        df_clean = df[[col_grupo, col_valor]].replace([np.inf, -np.inf], np.nan).dropna()
        if df_clean.empty: return []
        mapa = df_clean.groupby(col_grupo)[col_valor].mean().round(2)
        data = mapa.reset_index().rename(columns={col_valor: 'precio_promedio', col_grupo: 'zona'}).to_dict('records')
        return data
    except Exception as e:
        print(f"Error en generar_mapa_calor: {e}")
        return []

def generar_datos_3d(df: pd.DataFrame, col_x: str = 'area_m2', col_y: str = 'precio', col_z: str = 'habitaciones') -> list:
    try:
        cols = [col_x, col_y, col_z]
        if not all(c in df.columns for c in cols): raise ValueError(f"Faltan columnas: {cols}")
        if not all(pd.api.types.is_numeric_dtype(df[c]) for c in cols): raise ValueError(f"Columnas ({cols}) deben ser numéricas.")
        # CORRECCIÓN: Filtrar NaN/Inf antes de samplear
        df_clean = df[cols].replace([np.inf, -np.inf], np.nan).dropna()
        if df_clean.empty: return []
        sample_df = df_clean.sample(n=min(len(df_clean), 500), random_state=42)
        data = sample_df.rename(columns={col_x: 'area_m2', col_y: 'precio', col_z: 'habitaciones'}).to_dict('records')
        return data
    except Exception as e:
        print(f"Error en generar_datos_3d: {e}")
        return []

# =============================================================================
# 4️⃣ Funciones de Análisis Avanzado
# =============================================================================
def generar_analisis_radar(dataset_id: str) -> list:
    print(f"-> generando_analisis_radar para dataset {dataset_id}")
    try:
        # Placeholder
        return [
            {"casa": "Casa A (Ejemplo)", "metricas": [{"metrica": "Precio/m²", "valor": np.random.randint(70, 95)}, {"metrica": "Tamaño", "valor": np.random.randint(50, 70)}, {"metrica": "Habitaciones", "valor": np.random.randint(40, 60)}, {"metrica": "Antigüedad", "valor": np.random.randint(60, 80)}, {"metrica": "Calidad", "valor": np.random.randint(65, 90)}]},
            {"casa": "Casa B (Ejemplo)", "metricas": [{"metrica": "Precio/m²", "valor": np.random.randint(60, 80)}, {"metrica": "Tamaño", "valor": np.random.randint(70, 90)}, {"metrica": "Habitaciones", "valor": np.random.randint(65, 85)}, {"metrica": "Antigüedad", "valor": np.random.randint(30, 50)}, {"metrica": "Calidad", "valor": np.random.randint(75, 95)}]},
             {"casa": "Casa C (Ejemplo)", "metricas": [{"metrica": "Precio/m²", "valor": np.random.randint(30, 50)}, {"metrica": "Tamaño", "valor": np.random.randint(60, 80)}, {"metrica": "Habitaciones", "valor": np.random.randint(55, 75)}, {"metrica": "Antigüedad", "valor": np.random.randint(10, 30)}, {"metrica": "Calidad", "valor": np.random.randint(50, 70)}]},
        ]
    except Exception as e:
        print(f"🚨 ERROR en generar_analisis_radar: {e}")
        return []

def generar_sankey_flujo(dataset_id: str) -> dict:
    print(f"-> generando_sankey_flujo para dataset {dataset_id}")
    try:
        # Placeholder
        return {
            "nodes": [{"id": 0, "name": "Zona Centro"}, {"id": 1, "name": "Zona Residencial"}, {"id": 2, "name": "Zona Periférica"}, {"id": 3, "name": "Precio Bajo"}, {"id": 4, "name": "Precio Medio"}, {"id": 5, "name": "Precio Alto"}],
            "links": [{"source": 0, "target": 4, "value": np.random.randint(20, 40)}, {"source": 0, "target": 5, "value": np.random.randint(40, 60)}, {"source": 1, "target": 4, "value": np.random.randint(50, 70)}, {"source": 1, "target": 5, "value": np.random.randint(30, 50)}, {"source": 2, "target": 3, "value": np.random.randint(60, 80)}, {"source": 2, "target": 4, "value": np.random.randint(20, 30)}]
        }
    except Exception as e:
        print(f"🚨 ERROR en generar_sankey_flujo: {e}")
        return {"nodes": [], "links": []}

def generar_grafico_burbujas(dataset_id: str) -> list:
    print(f"-> generando_grafico_burbujas para dataset {dataset_id}")
    try:
        df = obtener_dataframe_crudo(dataset_id)
        if df.empty: return []
        col_x, col_y, col_size, col_cat = 'area_m2', 'precio', 'habitaciones', 'zona'
        required_cols = [col_x, col_y, col_size, col_cat]
        if not all(c in df.columns for c in required_cols):
             print(f"⚠️ Advertencia: Faltan columnas {required_cols} para burbujas.")
             return [{"x": 100, "y": 150000, "z": 300, "zona": "Ejemplo"}]
        # CORRECCIÓN: Convertir a numérico ANTES de dropna
        for col in [col_x, col_y, col_size]:
             df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=required_cols, inplace=True) # Ahora dropea si la conversión falló
        if df.empty: return []
        sample_df = df.sample(n=min(len(df), 200), random_state=42)
        # CORRECCIÓN: Manejar caso donde col_cat ya es numérico para cat_code
        if pd.api.types.is_categorical_dtype(sample_df[col_cat]) or pd.api.types.is_object_dtype(sample_df[col_cat]):
             sample_df['cat_code'] = pd.Categorical(sample_df[col_cat]).codes
        elif pd.api.types.is_numeric_dtype(sample_df[col_cat]):
             sample_df['cat_code'] = sample_df[col_cat] # Usar el valor numérico existente
        else:
             sample_df['cat_code'] = 0 # Código por defecto si no es ni categórico ni numérico
        col_code = 'cat_code'

        data_raw = sample_df[[col_x, col_y, col_size, col_cat, col_code]].rename(columns={col_x: 'x', col_y: 'y', col_size: 'size_value', col_cat: 'zona', col_code: 'z_code'}).to_dict('records')
        sizes = [d['size_value'] for d in data_raw if pd.notna(d['size_value'])]
        min_s, max_s = min(sizes) if sizes else 1, max(sizes) if sizes else 1
        range_s = max_s - min_s
        data_final = []
        for d in data_raw:
             scaled_z = 100 + ((d['size_value'] - min_s) / range_s) * 900 if pd.notna(d['size_value']) and range_s > 0 else 100
             data_final.append({'x': d['x'], 'y': d['y'], 'z': int(scaled_z), 'zona': d['zona'], 'codigo_zona': d['z_code']})
        return data_final
    except Exception as e:
        print(f"🚨 ERROR en generar_grafico_burbujas: {e}")
        return []

def generar_analisis_sensibilidad(dataset_id: str) -> list:
    print(f"-> generando_analisis_sensibilidad para dataset {dataset_id}")
    try:
        df = obtener_dataframe_crudo(dataset_id)
        if df.empty: return []
        target_col = 'precio'
        if target_col not in df.columns or not pd.api.types.is_numeric_dtype(df[target_col]): raise ValueError(f"Columna '{target_col}' inválida.")
        # CORRECCIÓN: Limpiar target col antes de correlación
        df_clean = df.copy()
        df_clean[target_col] = pd.to_numeric(df_clean[target_col], errors='coerce')
        df_clean = df_clean.replace([np.inf, -np.inf], np.nan).dropna(subset=[target_col])
        if df_clean.empty: return []

        numeric_cols = df_clean.select_dtypes(include=np.number).columns.tolist()
        if target_col in numeric_cols: numeric_cols.remove(target_col)
        valid_numeric_cols = [col for col in numeric_cols if df_clean[col].nunique() > 1 and df_clean[col].std() > 1e-9 and df_clean[col].notna().any()] # Asegurar que no sean todos NaN
        if not valid_numeric_cols: print("⚠️ No hay vars numéricas válidas."); return []

        sensibilidad_data = []
        # Calcular corr en df limpio y con columnas válidas
        correlations = df_clean[valid_numeric_cols + [target_col]].corr()[target_col].drop(target_col, errors='ignore')
        # Calcular volatilidad en df limpio
        volatility = (df_clean[valid_numeric_cols].std() / df_clean[valid_numeric_cols].mean()).abs() * 100

        for col in valid_numeric_cols:
            corr = correlations.get(col, 0); vol = volatility.get(col, 0); impact = abs(corr) * 100
            # Asegurar que los valores sean finitos antes de redondear
            sensibilidad_data.append({"variable": col, "impacto_precio": round(impact, 1) if np.isfinite(impact) else 0, "volatilidad": round(vol, 1) if np.isfinite(vol) else 0, "correlacion": round(corr, 3) if np.isfinite(corr) else 0})
        sensibilidad_data.sort(key=lambda x: x['impacto_precio'], reverse=True)
        return sensibilidad_data[:10]
    except Exception as e:
        print(f"🚨 ERROR en generar_analisis_sensibilidad: {e}")
        return []

def generar_clustering(dataset_id: str, n_clusters: int = 4) -> dict:
    print(f"-> generando_clustering (K={n_clusters}) para dataset {dataset_id}")
    COLORES_CLUSTER = ['#0ea5e9', '#8b5cf6', '#ec4899', '#f59e0b', '#10b981', '#ef4444']
    try:
        df = obtener_dataframe_crudo(dataset_id)
        if df.empty: return {"clusters": [], "centroides": [], "columnas": []}
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        cols_to_exclude = [col for col in numeric_cols if 'id' in col.lower() or df[col].nunique() < 2 or df[col].std(skipna=True) < 1e-9] # skipna=True en std
        cols_for_clustering = [col for col in numeric_cols if col not in cols_to_exclude]
        if len(cols_for_clustering) < 2: raise ValueError("Se necesitan >= 2 vars numéricas válidas.")
        print(f"   Columnas usadas: {cols_for_clustering}")
        X = df[cols_for_clustering].copy() # Usar copia para evitar SettingWithCopyWarning
        # CORRECCIÓN: Imputar ANTES de escalar
        imputer = SimpleImputer(strategy='mean'); X_imputed = imputer.fit_transform(X)
        # Convertir de nuevo a DataFrame para escalar (preserva nombres de columna)
        X_imputed_df = pd.DataFrame(X_imputed, columns=cols_for_clustering)

        scaler = StandardScaler(); X_scaled = scaler.fit_transform(X_imputed_df)
        actual_n_clusters = min(n_clusters, len(X_scaled))
        if actual_n_clusters < 2: raise ValueError("No hay suficientes muestras.")
        if actual_n_clusters != n_clusters: print(f"⚠️ n_clusters reducido a {actual_n_clusters}")
        kmeans = KMeans(n_clusters=actual_n_clusters, random_state=42, n_init=10); cluster_labels = kmeans.fit_predict(X_scaled)
        # CORRECCIÓN: Añadir labels al DataFrame ORIGINAL (df) usando el índice si es posible,
        # o crear uno nuevo si los índices se pierden. Asumimos que el índice se mantuvo.
        if len(cluster_labels) == len(df):
             df['cluster'] = cluster_labels
        else:
             print("⚠️ Error de alineación de índices en clustering. Los resultados pueden ser imprecisos.")
             # Crear df temporal solo con los datos clusterizados si hay error de índice
             df_clustered = X_imputed_df.copy()
             df_clustered['cluster'] = cluster_labels
             df = df_clustered # Sobrescribir df para el análisis de clusters

        clusters_info = []
        cols_to_describe = cols_for_clustering[:min(5, len(cols_for_clustering))]
        for i in range(actual_n_clusters):
            cluster_df = df[df['cluster'] == i]; count = len(cluster_df)
            if count == 0: continue
            caracteristicas = {}
            for col in cols_to_describe:
                 # CORRECCIÓN: Asegurar que la columna existe en cluster_df antes de agg
                 if col in cluster_df.columns:
                      stats = cluster_df[col].agg(['mean', 'min', 'max']).fillna(0)
                      # CORRECCIÓN: Asegurar que stats son finitos antes de formatear
                      caracteristicas[col] = {"promedio": f"{stats['mean']:.2f}" if np.isfinite(stats['mean']) else "N/A", "min": f"{stats['min']:.2f}" if np.isfinite(stats['min']) else "N/A", "max": f"{stats['max']:.2f}" if np.isfinite(stats['max']) else "N/A"}
                 else:
                      caracteristicas[col] = {"promedio": "N/A", "min": "N/A", "max": "N/A"}

            clusters_info.append({"id": i, "nombre": f"Grupo {i+1}", "cantidad": count, "caracteristicas": caracteristicas, "color": COLORES_CLUSTER[i % len(COLORES_CLUSTER)]})
        # CORRECCIÓN: Asegurar que kmeans tenga centroides antes de desescalar
        if hasattr(kmeans, 'cluster_centers_'):
             centroides_scaled = kmeans.cluster_centers_; centroides_original = scaler.inverse_transform(centroides_scaled)
             centroides_list = []
             cols_in_centroids_chart = cols_to_describe
             for i, centroide in enumerate(centroides_original):
                  centroide_data = {"cluster": f"Grupo {i+1}"}
                  for col_idx, col_name in enumerate(cols_for_clustering): # Iterar sobre las columnas usadas en Kmeans
                       # CORRECCIÓN: Verificar índice antes de acceder a centroide
                       if col_name in cols_in_centroids_chart and col_idx < len(centroide):
                            centroide_data[col_name] = round(centroide[col_idx], 2) if np.isfinite(centroide[col_idx]) else 0
                  centroides_list.append(centroide_data)
        else:
             centroides_list = []

        return {"clusters": clusters_info, "centroides": centroides_list, "columnas": cols_in_centroids_chart}
    except Exception as e:
        print(f"🚨 ERROR en generar_clustering: {e}")
        return {"clusters": [], "centroides": [], "columnas": []}

# =============================================================================
# 5️⃣ Funciones de Análisis de Mercado (CORREGIDAS en respuesta anterior)
# =============================================================================
def generar_segmentacion_mercado(dataset_id: str, col_precio: str = 'precio', col_area: str = 'area_m2') -> dict:
    # ... (código existente robustecido) ...
    print(f"-> generando_segmentacion_mercado para dataset {dataset_id}")
    try:
        df = obtener_dataframe_crudo(dataset_id)
        if df.empty or col_precio not in df.columns or not pd.api.types.is_numeric_dtype(df[col_precio]):
            raise ValueError(f"Dataset vacío o columna '{col_precio}' inválida.")
        # CORRECCIÓN: Limpiar precio antes de cuantiles
        precios_validos = df[col_precio].replace([np.inf, -np.inf], np.nan).dropna()
        if len(precios_validos) < 3: return {"distribucion": [], "estadisticas": {}}
        q1, q2 = precios_validos.quantile(0.33), precios_validos.quantile(0.66)
        segmentos_map = {'Economico': (df[col_precio] <= q1), 'Medio': (df[col_precio] > q1) & (df[col_precio] <= q2), 'Lujo': (df[col_precio] > q2)}
        distribucion, total_casas = [], len(df)
        for nombre, condicion in segmentos_map.items():
            segmento_df = df[condicion].copy() # Usar copia
            cantidad = len(segmento_df)
            if cantidad == 0: continue
            # CORRECCIÓN: Limpiar precio y área en el segmento antes de agg/mean
            segmento_df[col_precio] = pd.to_numeric(segmento_df[col_precio], errors='coerce')
            segmento_df = segmento_df.replace([np.inf, -np.inf], np.nan).dropna(subset=[col_precio])
            if segmento_df.empty: continue

            precio_stats = segmento_df[col_precio].agg(['mean', 'min', 'max']).fillna(0)
            area_prom = 0
            if col_area in segmento_df.columns and pd.api.types.is_numeric_dtype(segmento_df[col_area]):
                 segmento_df[col_area] = pd.to_numeric(segmento_df[col_area], errors='coerce')
                 area_prom = segmento_df[col_area].replace([np.inf, -np.inf], np.nan).mean()

            distribucion.append({"segmento": nombre, "cantidad": cantidad, "precio_promedio": round(precio_stats['mean'], 2) if np.isfinite(precio_stats['mean']) else 0, "precio_min": round(precio_stats['min'], 2) if np.isfinite(precio_stats['min']) else 0, "precio_max": round(precio_stats['max'], 2) if np.isfinite(precio_stats['max']) else 0, "porcentaje": round((cantidad / total_casas) * 100, 1) if total_casas > 0 else 0, "area_promedio": round(area_prom, 2) if pd.notna(area_prom) and np.isfinite(area_prom) else 0})

        precio_medio_total = df[col_precio].replace([np.inf, -np.inf], np.nan).mean()
        estadisticas_generales = {"precio_promedio_total": round(precio_medio_total, 2) if pd.notna(precio_medio_total) and np.isfinite(precio_medio_total) else 0, "total_propiedades": total_casas}
        return {"distribucion": distribucion, "estadisticas": estadisticas_generales}
    except Exception as e:
        print(f"🚨 ERROR en generar_segmentacion_mercado: {e}")
        return {"distribucion": [], "estadisticas": {}}


def detectar_anomalias_precios(dataset_id: str, cols_features: list = ['area_m2', 'habitaciones', 'banos', 'ano_construccion'], col_precio: str = 'precio', contamination: float = 0.05) -> dict:
    """ Detecta anomalías (outliers) usando Isolation Forest. """
    print(f"-> detectando_anomalias_precios para dataset {dataset_id}")
    default_return = {"gangas": [], "sobrevaloradas": [], "resumen": {}}
    try:
        df = obtener_dataframe_crudo(dataset_id)
        if df.empty: return default_return
        cols_para_anomalia = [col for col in cols_features if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
        if col_precio not in df.columns or not pd.api.types.is_numeric_dtype(df[col_precio]): raise ValueError(f"Columna '{col_precio}' inválida.")
        cols_para_anomalia.append(col_precio)
        if len(cols_para_anomalia) < 2: raise ValueError("Se necesitan >= 1 feature + precio para anomalías.")

        X = df[cols_para_anomalia].copy()
        original_indices = X.index # Guardar índice ANTES de dropna/imputar

        # CORRECCIÓN: Convertir a numérico y reemplazar Inf ANTES de imputar
        for col in cols_para_anomalia:
             X[col] = pd.to_numeric(X[col], errors='coerce')
        X.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Imputar NaN
        imputer = SimpleImputer(strategy='mean'); X_imputed = imputer.fit_transform(X)
        # Asegurar que no queden NaN después de imputar (si toda una columna era NaN)
        if np.isnan(X_imputed).any():
            print("⚠️ Advertencia: NaN detectados después de imputación. Reemplazando con 0.")
            X_imputed = np.nan_to_num(X_imputed, nan=0.0)

        X_imputed_df = pd.DataFrame(X_imputed, columns=cols_para_anomalia, index=original_indices)

        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        # CORRECCIÓN: Asegurar que no haya NaN/Inf antes de fit_predict
        if X_imputed_df.isnull().values.any() or np.isinf(X_imputed_df.values).any():
             raise ValueError("Datos inválidos (NaN/Inf) antes de IsolationForest.")

        anomalia_pred = iso_forest.fit_predict(X_imputed_df)
        scores_anomalia = iso_forest.decision_function(X_imputed_df)

        # Usar el DataFrame original alineado con los resultados
        df_anomalias = df.loc[original_indices].copy()
        df_anomalias['es_anomalia'] = anomalia_pred; df_anomalias['score_anomalia'] = scores_anomalia

        # Calcular Z-score y desviación % (con manejo robusto)
        precio_col_clean = df_anomalias[col_precio].replace([np.inf, -np.inf], np.nan).dropna()
        precio_medio = precio_col_clean.mean()
        precio_std = precio_col_clean.std()
        if pd.isna(precio_medio) or pd.isna(precio_std) or precio_std == 0:
             precio_medio = 0; precio_std = 1 # Valores por defecto si no se pueden calcular
             df_anomalias['precio_zscore'] = 0
             df_anomalias['desviacion_media_pct'] = 0
        else:
             df_anomalias['precio_zscore'] = (df_anomalias[col_precio] - precio_medio) / precio_std
             df_anomalias['desviacion_media_pct'] = round(((df_anomalias[col_precio] - precio_medio) / (precio_medio if precio_medio != 0 else 1)) * 100, 1)
             # Rellenar NaN que puedan quedar si el precio original era NaN
             df_anomalias['precio_zscore'].fillna(0, inplace=True)
             df_anomalias['desviacion_media_pct'].fillna(0, inplace=True)


        anomalias_df = df_anomalias[df_anomalias['es_anomalia'] == -1].copy()
        gangas_df = anomalias_df[anomalias_df['precio_zscore'] < -0.5]
        sobrevaloradas_df = anomalias_df[anomalias_df['precio_zscore'] > 0.5]

        # --- Función de formateo segura ---
        def safe_int_convert(value, default=0):
             if pd.isna(value) or np.isinf(value): return default
             try: return int(float(value))
             except (ValueError, TypeError): return default

        def formatear_anomalia(row_index, row_data, tipo):
             data = {
                 "indice": int(row_index),
                 "precio": round(row_data.get(col_precio, 0), 2) if pd.notna(row_data.get(col_precio)) else 0,
                 "tipo": tipo,
                 "score": round(row_data.get('score_anomalia', 0), 4) if pd.notna(row_data.get('score_anomalia')) else 0,
                 "desviacion_media": abs(round(row_data.get('desviacion_media_pct', 0), 1)) if pd.notna(row_data.get('desviacion_media_pct')) else 0
             }
             data['area_m2'] = round(row_data['area_m2'], 2) if 'area_m2' in row_data and pd.notna(row_data['area_m2']) else 0
             data['habitaciones'] = safe_int_convert(row_data.get('habitaciones'))
             return data
        # --- FIN ---

        gangas = [formatear_anomalia(row.Index, row._asdict(), 'Ganga') for row in gangas_df.itertuples()]
        sobrevaloradas = [formatear_anomalia(row.Index, row._asdict(), 'Sobrevalorada') for row in sobrevaloradas_df.itertuples()]
        gangas.sort(key=lambda x: x['score']); sobrevaloradas.sort(key=lambda x: x['score'])

        total_propiedades = len(df_anomalias); total_gangas = len(gangas); total_sobrevaloradas = len(sobrevaloradas); total_normales = total_propiedades - total_gangas - total_sobrevaloradas
        resumen = {
            "total_gangas": total_gangas, "porcentaje_gangas": round((total_gangas / total_propiedades) * 100, 1) if total_propiedades else 0,
            "total_sobrevaloradas": total_sobrevaloradas, "porcentaje_sobrevaloradas": round((total_sobrevaloradas / total_propiedades) * 100, 1) if total_propiedades else 0,
            "total_normales": total_normales
        }
        return {"gangas": gangas, "sobrevaloradas": sobrevaloradas, "resumen": resumen}
    except Exception as e:
        print(f"🚨 ERROR en detectar_anomalias_precios: {e}")
        return default_return


def calcular_score_inversion(dataset_id: str, col_precio: str = 'precio', col_area: str = 'area_m2', col_ano: str = 'ano_construccion', col_zona: str = 'zona') -> dict:
    """ Calcula un score de inversión simple para cada propiedad. """
    print(f"-> calculando_score_inversion para dataset {dataset_id}")
    default_stats = {"propiedades": [], "estadisticas": {"score_promedio": 0.0, "total_propiedades": 0, "distribucion": {"excelente": 0, "muy_bueno": 0, "bueno": 0, "regular": 0, "bajo": 0}}}
    try:
        df = obtener_dataframe_crudo(dataset_id)
        if df.empty:
             print("   Dataset vacío, no se puede calcular score.")
             return default_stats
        df_scores = df.copy()

        # --- Funciones de cálculo de score (robustecidas) ---
        def score_precio_func(serie):
            serie = serie.replace([np.inf, -np.inf], np.nan).dropna() # Limpiar serie
            if serie.empty or serie.nunique() < 2: return pd.Series([50] * len(serie), index=serie.index) # Usar longitud original o la limpia? Usar limpia por ahora.
            mean, std = serie.mean(), serie.std()
            if std > 0 and pd.notna(std):
                z = (serie - mean) / std; z_clamped = z.clip(-2, 2)
                score = ((-z_clamped + 2) / 4) * 100
                return score.fillna(50)
            else: return pd.Series([50] * len(serie), index=serie.index)

        def score_antiguedad_func(serie):
            serie = serie.replace([np.inf, -np.inf], np.nan).dropna()
            if serie.empty or serie.nunique() < 2: return pd.Series([50] * len(serie), index=serie.index)
            min_val, max_val = serie.min(), serie.max()
            if max_val > min_val and pd.notna(min_val) and pd.notna(max_val):
                 score = ((serie - min_val) / (max_val - min_val)) * 100
                 return score.fillna(50)
            else: return pd.Series([50] * len(serie), index=serie.index)

        def score_zona_func(serie): # Asume zona numérica, ID bajo = mejor
             serie = serie.replace([np.inf, -np.inf], np.nan).dropna()
             if serie.empty or serie.nunique() < 2: return pd.Series([50] * len(serie), index=serie.index)
             min_val, max_val = serie.min(), serie.max()
             if max_val > min_val and pd.notna(min_val) and pd.notna(max_val):
                  score = (1 - (serie - min_val) / (max_val - min_val)) * 100
                  return score.fillna(50)
             else: return pd.Series([50] * len(serie), index=serie.index)

        # --- Calcular scores individuales ---
        score_p = pd.Series([50.0] * len(df_scores), index=df_scores.index) # Inicializar con defecto
        if col_precio in df_scores.columns and pd.api.types.is_numeric_dtype(df_scores[col_precio]):
             score_p_calc = score_precio_func(df_scores[col_precio])
             score_p = score_p_calc.reindex(df_scores.index).fillna(50) # Reindexar y rellenar NaNs
        else: print(f"⚠️ Columna '{col_precio}' no válida para score_precio.")

        score_a = pd.Series([50.0] * len(df_scores), index=df_scores.index)
        if col_ano in df_scores.columns and pd.api.types.is_numeric_dtype(df_scores[col_ano]):
             score_a_calc = score_antiguedad_func(df_scores[col_ano])
             score_a = score_a_calc.reindex(df_scores.index).fillna(50)
        else: print(f"⚠️ Columna '{col_ano}' no válida para score_antiguedad.")

        score_z = pd.Series([50.0] * len(df_scores), index=df_scores.index)
        if col_zona in df_scores.columns and pd.api.types.is_numeric_dtype(df_scores[col_zona]):
             score_z_calc = score_zona_func(df_scores[col_zona])
             score_z = score_z_calc.reindex(df_scores.index).fillna(50)
        else: print(f"⚠️ Columna '{col_zona}' no válida para score_zona.")

        df_scores['score_precio'] = score_p
        df_scores['score_antiguedad'] = score_a
        df_scores['score_zona'] = score_z

        # Score Combinado (Ponderado)
        df_scores['score_inversion_float'] = (0.5 * df_scores['score_precio'] + 0.3 * df_scores['score_antiguedad'] + 0.2 * df_scores['score_zona'])
        df_scores['score_inversion'] = df_scores['score_inversion_float'].fillna(50).round().astype(int).clip(0, 100)

        # Asignar Potencial
        def asignar_potencial(score):
            if pd.isna(score): return 'Bajo'
            score = int(score) # Asegurar que sea int
            if score >= 80: return 'Excelente';
            if score >= 65: return 'Muy Bueno';
            if score >= 50: return 'Bueno';
            if score >= 35: return 'Regular';
            return 'Bajo'
        df_scores['potencial'] = df_scores['score_inversion'].apply(asignar_potencial)

        # Formatear Salida
        propiedades = []
        def safe_get(row_dict, key, default=None, round_digits=None, convert_int=False):
            val = row_dict.get(key)
            if pd.isna(val) or (isinstance(val, float) and not np.isfinite(val)): return default # Chequear NaN y Inf
            try:
                num_val = float(val)
                if round_digits is not None: num_val = round(num_val, round_digits)
                if convert_int: return int(num_val)
                return num_val
            except (ValueError, TypeError): return val if val is not None else default

        for index, row in df.iterrows(): # Iterar sobre df ORIGINAL para tener todas las filas
             score_row = df_scores.loc[index] if index in df_scores.index else None
             if score_row is None or pd.isna(score_row.get('score_inversion')): continue # Saltar si no hay score o es NaN

             prop = {
                 "indice": int(index),
                 "precio": safe_get(row, col_precio, default=0.0, round_digits=2),
                 "score_inversion": int(score_row['score_inversion']),
                 "potencial": score_row.get('potencial', 'Bajo')
             }
             prop['area_m2'] = safe_get(row, col_area, default=0.0, round_digits=2)
             # CORRECCIÓN: Devolver zona original (puede ser str), manejar NaN
             zona_val = row.get(col_zona)
             prop['zona'] = zona_val if pd.notna(zona_val) else None
             prop['ano_construccion'] = safe_get(row, col_ano, default=0, convert_int=True)

             # Limpiar diccionario final de NaNs flotantes (opcional, pero buena práctica)
             prop_clean = {}
             for k, v in prop.items():
                  if isinstance(v, float) and (pd.isna(v) or not np.isfinite(v)):
                       prop_clean[k] = None # O 0 según prefieras
                  else:
                       prop_clean[k] = v
             propiedades.append(prop_clean)


        propiedades.sort(key=lambda x: x.get('score_inversion', 0), reverse=True)

        # Estadísticas
        score_promedio_calc = df_scores['score_inversion'].mean() # Ya maneja NaN internamente
        score_promedio_final = round(score_promedio_calc, 1) if pd.notna(score_promedio_calc) else 0.0
        distribucion_potencial = df_scores['potencial'].value_counts().to_dict()
        estadisticas = {
            "score_promedio": score_promedio_final, "total_propiedades": len(propiedades), # Usar len(propiedades) filtradas
            "distribucion": {"excelente": distribucion_potencial.get('Excelente', 0), "muy_bueno": distribucion_potencial.get('Muy Bueno', 0), "bueno": distribucion_potencial.get('Bueno', 0), "regular": distribucion_potencial.get('Regular', 0), "bajo": distribucion_potencial.get('Bajo', 0)}
        }

        if not propiedades: print("⚠️ La lista final de propiedades para score de inversión está vacía.")

        return {"propiedades": propiedades, "estadisticas": estadisticas}
    except Exception as e:
        print(f"🚨 ERROR en calcular_score_inversion: {e}")
        return default_stats

