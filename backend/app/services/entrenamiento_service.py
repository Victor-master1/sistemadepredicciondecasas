# --- Importaciones ---
import torch
import torch.nn as nn
from app.services.supabase_service import supabase
from app.services.analisis_service import obtener_dataframe_crudo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, StandardScaler as TargetScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, mean_squared_error, r2_score,
    recall_score, f1_score, roc_curve, roc_auc_score,
    confusion_matrix,
    precision_score
)
from sklearn.inspection import permutation_importance
from sklearn.base import BaseEstimator
from datetime import datetime
import json
import pandas as pd
import numpy as np
import uuid
import time
# NUEVO: Importaciones para guardar/cargar artefactos
import joblib
import io

# NUEVO: Nombre del bucket de Supabase Storage para los artefactos
ARTEFACTOS_BUCKET_NAME = "artefactos_modelos"

# --- Clase PyTorchSklearnWrapper (Sin Cambios) ---
class PyTorchSklearnWrapper(BaseEstimator):
    # ... (Código del Wrapper sin cambios) ...
    """
    Envuelve un modelo PyTorch para hacerlo compatible con funciones de Sklearn
    que solo necesitan el método predict() o predict_proba(), como permutation_importance.
    """
    def __init__(self, model, scaler_y=None, is_classification=False, num_classes=None):
        self.model = model
        self.scaler_y = scaler_y # Para desescalar predicciones de regresión
        self.is_classification = is_classification
        self.num_classes = num_classes
        self.model.eval() # Asegurarse de que el modelo esté en modo evaluación

    def fit(self, X, y):
        # Permutation importance puede llamar a fit internamente, pero no lo necesitamos.
        return self

    def predict(self, X):
        """Hace predicciones y las devuelve en el formato esperado (numpy array)."""
        tensor_X = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            outputs = self.model(tensor_X)

            if self.is_classification:
                if self.num_classes == 2:
                    preds = (torch.sigmoid(outputs) > 0.5).long().flatten().numpy()
                else:
                    _, preds = torch.max(outputs.data, 1)
                    preds = preds.numpy()
                return preds
            else: # Regresión
                preds_scaled = outputs.numpy().flatten()
                # Desescalar si se proporcionó un scaler para 'y'
                if self.scaler_y:
                    preds = self.scaler_y.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
                else:
                    preds = preds_scaled
                return preds

    def predict_proba(self, X):
        """Devuelve probabilidades para clasificación (si aplica)."""
        if not self.is_classification:
            raise AttributeError("predict_proba solo está disponible para modelos de clasificación.")

        tensor_X = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            outputs = self.model(tensor_X)
            if self.num_classes == 2:
                probs = torch.sigmoid(outputs).numpy()
                # Sklearn espera [prob_clase_0, prob_clase_1]
                return np.hstack((1 - probs, probs))
            else:
                # Usar Softmax para obtener probabilidades multiclase
                probs = torch.softmax(outputs, dim=1).numpy()
                return probs

    # Métodos necesarios para evitar errores de clonación en Sklearn
    def get_params(self, deep=True):
         return {'model': self.model, 'scaler_y': self.scaler_y, 'is_classification': self.is_classification, 'num_classes': self.num_classes}

    def set_params(self, **parameters):
         for parameter, value in parameters.items():
              setattr(self, parameter, value)
         return self


# --- Clase NeuralNet (Con BatchNorm1d - Sin Cambios) ---
class NeuralNet(nn.Module):
    # ... (Código de NeuralNet sin cambios) ...
    def __init__(self, input_size, num_classes, is_regression=False):
        super(NeuralNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        if is_regression:
            self.output_layer = nn.Linear(32, 1)
        else:
            self.output_layer = nn.Linear(32, num_classes if num_classes > 2 else 1)

    def forward(self, x):
        x = self.network(x)
        return self.output_layer(x)

def iniciar_nuevo_entrenamiento(config: dict):
    # ... (Validación inicial, variables, preparación datos, entrenamiento... todo sin cambios hasta el final) ...
    if config.get('columna_objetivo') in config.get('columnas_entrada', []):
        raise ValueError("La columna objetivo no puede estar incluida en las columnas de entrada.")

    experimento_id = str(uuid.uuid4())
    estado_experimento = 'iniciando'
    tipo_problema_detectado = 'indefinido'
    scaler_y = None
    le = None # NUEVO: Guardaremos el LabelEncoder si existe

    try:
        # --- 1. Preparación de Datos ---
        start_time = time.time()
        dataset_id = config.get('dataset_id')
        tipo_modelo_usuario = config.get('tipo_modelo')
        columna_objetivo = config.get('columna_objetivo')
        print(f"🚀 Iniciando entrenamiento para: {dataset_id} con {tipo_modelo_usuario}")

        df = obtener_dataframe_crudo(dataset_id)
        for col in df.select_dtypes(include=np.number).columns:
            if df[col].isnull().sum() > 0: df[col].fillna(df[col].mean(), inplace=True)

        columnas_categoricas = [col for col in df.columns if df[col].dtype == 'object' and col != columna_objetivo]
        if columnas_categoricas: df = pd.get_dummies(df, columns=columnas_categoricas, drop_first=True)

        columnas_disponibles = [col for col in config['columnas_entrada'] if col in df.columns]
        X = df[columnas_disponibles].apply(pd.to_numeric, errors='coerce').fillna(0)
        y_raw = df[columna_objetivo]


        # --- 2. Detección y Procesamiento de 'y' ---
        es_clasificacion = (pd.api.types.is_string_dtype(y_raw) or
                            pd.api.types.is_categorical_dtype(y_raw) or
                           (pd.api.types.is_integer_dtype(y_raw) and y_raw.nunique() <= 30))

        tipo_problema_detectado = "clasificacion" if es_clasificacion else "regresion"
        print(f"🧠 Tipo de problema detectado: {tipo_problema_detectado.upper()}")

        if tipo_modelo_usuario == 'regresion' and es_clasificacion and y_raw.nunique() > 30:
            raise ValueError(f"Conflicto de tipos. El modelo 'Random Forest' no soporta clasificación de alta cardinalidad ({y_raw.nunique()} clases). Intente con 'Red Neuronal'.")

        num_classes_detected = 1
        if es_clasificacion:
            le = LabelEncoder() # NUEVO: Asignar a la variable global
            y = pd.Series(le.fit_transform(y_raw.fillna(y_raw.mode()[0])), name=columna_objetivo)
            num_classes_detected = y.nunique()
            if num_classes_detected < 2:
                raise ValueError(f"La columna objetivo '{columna_objetivo}' debe tener al menos 2 clases para clasificar.")
        else:
            y = pd.to_numeric(y_raw, errors='coerce').fillna(y_raw.mean())
            scaler_y = TargetScaler()
            y = pd.Series(scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten(), name=columna_objetivo)
            print("-> Variable objetivo 'y' escalada (regresión).")

        # --- División y Escalado de 'X' ---
        try:
            print("-> Intentando división estratificada...")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config.get('validacion_split', 0.2), random_state=42, stratify=y if es_clasificacion else None)
        except ValueError:
            print("⚠️ Advertencia: La estratificación falló. Usando división normal.")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config.get('validacion_split', 0.2), random_state=42, stratify=None)

        scaler_x = StandardScaler(); X_train_scaled = scaler_x.fit_transform(X_train); X_test_scaled = scaler_x.transform(X_test)

        # --- 3. Entrenamiento del Modelo ---
        modelo_entrenado, predicciones, train_predicciones, metricas_por_epoca, tiempos_por_epoca = None, None, None, [], []
        y_test_original = y_test
        y_train_original = y_train

        # ... (Toda la lógica de entrenamiento de NN y RF sin cambios) ...
        if tipo_modelo_usuario == 'red_neuronal':
            # ... (Entrenamiento PyTorch) ...
            print("-> Entrenando Red Neuronal (PyTorch)...")
            X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32); X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)

            y_train_torch = torch.tensor(y_train_original.values, dtype=torch.long)
            y_test_torch = torch.tensor(y_test_original.values, dtype=torch.long)

            if es_clasificacion:
                if num_classes_detected == 2: # Clasificación binaria
                    y_train_t = y_train_torch.float().unsqueeze(1)
                    y_test_t = y_test_torch.float().unsqueeze(1)
                    criterion = nn.BCEWithLogitsLoss()
                else: # Clasificación multiclase
                    y_train_t = y_train_torch
                    y_test_t = y_test_torch
                    criterion = nn.CrossEntropyLoss()
            else: # Regresión
                y_train_t = torch.tensor(y_train_original.values, dtype=torch.float32).unsqueeze(1)
                y_test_t = torch.tensor(y_test_original.values, dtype=torch.float32).unsqueeze(1)
                criterion = nn.MSELoss()

            modelo_entrenado = NeuralNet(X_train_t.shape[1], num_classes_detected, is_regression=not es_clasificacion)
            optimizer = torch.optim.Adam(modelo_entrenado.parameters(), lr=config.get('tasa_aprendizaje', 0.001))

            for epoch in range(config.get('epocas', 100)):
                 epoch_start_time = time.time()
                 modelo_entrenado.train()
                 outputs = modelo_entrenado(X_train_t);
                 loss = criterion(outputs, y_train_t)
                 optimizer.zero_grad(); loss.backward(); optimizer.step()

                 modelo_entrenado.eval()
                 with torch.no_grad():
                     train_outputs = modelo_entrenado(X_train_t)
                     train_loss = criterion(train_outputs, y_train_t)
                     val_outputs = modelo_entrenado(X_test_t)
                     val_loss = criterion(val_outputs, y_test_t)

                     epoca_metrics = {
                         'epoca': epoch + 1,
                         'perdida_validacion': float(val_loss.item()),
                         'perdida_entrenamiento': float(train_loss.item())
                     }

                     if es_clasificacion:
                          train_labels, val_labels = None, None
                          if num_classes_detected == 2:
                              train_labels = (torch.sigmoid(train_outputs) > 0.5).long().flatten()
                              val_labels = (torch.sigmoid(val_outputs) > 0.5).long().flatten()
                          else:
                              _, train_labels = torch.max(train_outputs.data, 1)
                              _, val_labels = torch.max(val_outputs.data, 1)

                          epoca_metrics['precision_entrenamiento'] = precision_score(y_train_original, train_labels.numpy(), average='weighted', zero_division=0)
                          epoca_metrics['precision_validacion'] = precision_score(y_test_original, val_labels.numpy(), average='weighted', zero_division=0)
                     else: # Regresión
                         if scaler_y:
                             y_train_true_orig = scaler_y.inverse_transform(y_train_t.detach().numpy())
                             train_preds_orig = scaler_y.inverse_transform(train_outputs.detach().numpy())
                             y_test_true_orig = scaler_y.inverse_transform(y_test_t.detach().numpy())
                             val_preds_orig = scaler_y.inverse_transform(val_outputs.detach().numpy())

                             epoca_metrics['r2_entrenamiento'] = r2_score(y_train_true_orig, train_preds_orig)
                             epoca_metrics['r2_validacion'] = r2_score(y_test_true_orig, val_preds_orig)
                         else:
                              epoca_metrics['r2_entrenamiento'] = r2_score(y_train_t.detach().numpy(), train_outputs.detach().numpy())
                              epoca_metrics['r2_validacion'] = r2_score(y_test_t.detach().numpy(), val_outputs.detach().numpy())

                     metricas_por_epoca.append(epoca_metrics)

                 epoch_end_time = time.time()
                 tiempos_por_epoca.append(epoch_end_time - epoch_start_time)

            with torch.no_grad():
                 final_outputs = modelo_entrenado(X_test_t)
                 final_train_outputs = modelo_entrenado(X_train_t)

                 if es_clasificacion:
                      if num_classes_detected == 2:
                          predicciones = (torch.sigmoid(final_outputs) > 0.5).long().flatten().numpy()
                          train_predicciones = (torch.sigmoid(final_train_outputs) > 0.5).long().flatten().numpy()
                      else:
                          _, predicciones = torch.max(final_outputs.data, 1); predicciones = predicciones.numpy()
                          _, train_predicciones = torch.max(final_train_outputs.data, 1); train_predicciones = train_predicciones.numpy()
                 else: # Regresión
                     predicciones_scaled = final_outputs.numpy().flatten()
                     train_predicciones_scaled = final_train_outputs.numpy().flatten()
                     if scaler_y:
                         predicciones = scaler_y.inverse_transform(predicciones_scaled.reshape(-1, 1)).flatten()
                         train_predicciones = scaler_y.inverse_transform(train_predicciones_scaled.reshape(-1, 1)).flatten()
                         print("-> Predicciones desescaladas.")
                     else:
                         predicciones = predicciones_scaled
                         train_predicciones = train_predicciones_scaled

        elif tipo_modelo_usuario == 'regresion':
              y_test_orig_rf = y_test
              y_train_orig_rf = y_train

              if not es_clasificacion and scaler_y:
                   y_train = pd.Series(scaler_y.inverse_transform(y_train.values.reshape(-1, 1)).flatten(), name=columna_objetivo)
                   y_test = pd.Series(scaler_y.inverse_transform(y_test.values.reshape(-1, 1)).flatten(), name=columna_objetivo)
                   print("-> 'y' desescalada para entrenamiento Sklearn.")

              if es_clasificacion:
                  print(f"-> Entrenando Clasificación (Random Forest Classifier)...")
                  modelo_entrenado = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
              else:
                  print(f"-> Entrenando Regresión (Random Forest Regressor)...")
                  modelo_entrenado = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

              modelo_entrenado.fit(X_train_scaled, y_train)
              predicciones = modelo_entrenado.predict(X_test_scaled)
              train_predicciones = modelo_entrenado.predict(X_train_scaled)

              y_test = y_test_orig_rf
              y_train = y_train_orig_rf
        else:
             raise ValueError(f"Tipo de modelo '{tipo_modelo_usuario}' no reconocido.")


        end_time = time.time()

        # --- 4. CÁLCULO DE MÉTRICAS ---
        print("-> Calculando métricas finales...")
        metricas = {}
        matriz_confusion, curva_roc, importancia_features, distribucion_errores, predicciones_vs_reales = None, None, None, None, None

        y_test_eval = y_test
        y_train_eval = y_train
        if not es_clasificacion and scaler_y:
             y_test_eval = pd.Series(scaler_y.inverse_transform(y_test.values.reshape(-1, 1)).flatten(), name=columna_objetivo)
             y_train_eval = pd.Series(scaler_y.inverse_transform(y_train.values.reshape(-1, 1)).flatten(), name=columna_objetivo)

        if predicciones is not None:
             # ... (Cálculo de métricas sin cambios) ...
              if es_clasificacion:
                  metricas = {
                      'accuracy': accuracy_score(y_test_eval, predicciones),
                      'precision': precision_score(y_test_eval, predicciones, average='weighted', zero_division=0),
                      'recall': recall_score(y_test_eval, predicciones, average='weighted', zero_division=0),
                      'f1_score': f1_score(y_test_eval, predicciones, average='weighted', zero_division=0),
                  }
                  if train_predicciones is not None:
                       metricas['precision_entrenamiento'] = precision_score(y_train_eval, train_predicciones, average='weighted', zero_division=0)
                  metricas['precision_validacion'] = metricas['precision']
                  matriz_confusion = confusion_matrix(y_test_eval, predicciones).tolist()

                  if num_classes_detected == 2:
                       print("-> Calculando curva ROC...")
                       pred_prob = None
                       if hasattr(modelo_entrenado, 'predict_proba'):
                            pred_prob = modelo_entrenado.predict_proba(X_test_scaled)[:, 1]
                       elif tipo_modelo_usuario == 'red_neuronal':
                           wrapped_model = PyTorchSklearnWrapper(modelo_entrenado, scaler_y=scaler_y, is_classification=True, num_classes=num_classes_detected)
                           pred_prob = wrapped_model.predict_proba(X_test_scaled)[:, 1]

                       if pred_prob is not None:
                            fpr, tpr, _ = roc_curve(y_test_eval, pred_prob)
                            curva_roc = {'auc': roc_auc_score(y_test_eval, pred_prob), 'fpr': fpr.tolist(), 'tpr': tpr.tolist()}

              else: # REGRESSION
                  mse_validacion = mean_squared_error(y_test_eval, predicciones)
                  r2 = r2_score(y_test_eval, predicciones)
                  mse_entrenamiento = mean_squared_error(y_train_eval, train_predicciones) if train_predicciones is not None else None

                  metricas = {
                      'mse': mse_validacion,
                      'mse_validacion': mse_validacion,
                      'mse_entrenamiento': mse_entrenamiento,
                      'perdida_final': mse_validacion,
                      'r2_score': r2,
                  }

                  distribucion_errores = (y_test_eval.values - predicciones).tolist()
                  predicciones_vs_reales = [{'real': float(r), 'prediccion': float(p)} for r, p in zip(y_test_eval.values, predicciones)]


        if metricas_por_epoca:
             # ... (Sobrescritura métricas NN sin cambios) ...
              last_epoch_metrics = metricas_por_epoca[-1]
              if es_clasificacion:
                  if 'precision_validacion' in last_epoch_metrics:
                      metricas['precision'] = last_epoch_metrics['precision_validacion']
                      metricas['precision_validacion'] = last_epoch_metrics['precision_validacion']
                  if 'precision_entrenamiento' in last_epoch_metrics:
                      metricas['precision_entrenamiento'] = last_epoch_metrics['precision_entrenamiento']
              else: # REGRESSION (Neural Net)
                  if 'perdida_entrenamiento' in last_epoch_metrics:
                      metricas['nn_loss_train_scaled'] = last_epoch_metrics['perdida_entrenamiento']
                  if 'perdida_validacion' in last_epoch_metrics:
                       metricas['nn_loss_val_scaled'] = last_epoch_metrics['perdida_validacion']
                  if 'r2_validacion' in last_epoch_metrics:
                      metricas['r2_score'] = last_epoch_metrics['r2_validacion']


        # --- Cálculo de Importancia de Features (sin cambios) ---
        print("-> Calculando importancia de features...")
        # ... (código existente con el wrapper) ...
        n_jobs_val = 1 if tipo_modelo_usuario == 'red_neuronal' else -1
        print(f"-> Usando n_jobs={n_jobs_val} para permutation_importance")
        try:
            if tipo_modelo_usuario == 'red_neuronal':
                estimator_for_perm = PyTorchSklearnWrapper(
                    model=modelo_entrenado,
                    scaler_y=scaler_y,
                    is_classification=es_clasificacion,
                    num_classes=num_classes_detected
                )
                scoring_metric = 'accuracy' if es_clasificacion else 'neg_mean_squared_error'

                imps = permutation_importance(
                    estimator_for_perm,
                    X_test_scaled,
                    y_test_eval.values,
                    scoring=scoring_metric,
                    n_repeats=5,
                    random_state=42,
                    n_jobs=n_jobs_val
                )
                importancia_features = [{'feature': f, 'importancia': float(imp)} for f, imp in zip(X.columns, imps.importances_mean)]

            else: # Para modelos Sklearn
                imps = permutation_importance(modelo_entrenado, X_test_scaled, y_test_eval, n_repeats=10, random_state=42, n_jobs=n_jobs_val)
                importancia_features = [{'feature': f, 'importancia': float(imp)} for f, imp in zip(X.columns, imps.importances_mean)]

            importancia_features.sort(key=lambda x: x['importancia'], reverse=True)
            print("-> Importancia de features calculada.")

        except Exception as imp_err:
            print(f"⚠️ No se pudo calcular la importancia de features: {imp_err}")
            importancia_features = None


        # --- 5. NUEVO: Guardar artefactos en Supabase Storage ---
        print("-> Guardando artefactos del modelo...")
        artefactos_info = {}
        artefactos_guardados = {} # Guardaremos buffers en memoria aquí

        try:
            # Definir nombres de archivo únicos
            base_path = f"experimento_{experimento_id}"
            artefactos_info['model_type'] = tipo_modelo_usuario
            artefactos_info['columns'] = X.columns.tolist() # Guardar columnas usadas

            # Guardar scaler_x (siempre existe)
            scaler_x_buffer = io.BytesIO()
            joblib.dump(scaler_x, scaler_x_buffer)
            scaler_x_buffer.seek(0)
            artefactos_guardados['scaler_x.joblib'] = scaler_x_buffer
            artefactos_info['scaler_x_path'] = f"{base_path}/scaler_x.joblib"

            # Guardar scaler_y (si es regresión)
            if scaler_y:
                scaler_y_buffer = io.BytesIO()
                joblib.dump(scaler_y, scaler_y_buffer)
                scaler_y_buffer.seek(0)
                artefactos_guardados['scaler_y.joblib'] = scaler_y_buffer
                artefactos_info['scaler_y_path'] = f"{base_path}/scaler_y.joblib"

            # Guardar LabelEncoder (si es clasificación)
            if le:
                 le_buffer = io.BytesIO()
                 joblib.dump(le, le_buffer)
                 le_buffer.seek(0)
                 artefactos_guardados['label_encoder.joblib'] = le_buffer
                 artefactos_info['label_encoder_path'] = f"{base_path}/label_encoder.joblib"


            # Guardar modelo específico
            if tipo_modelo_usuario == 'red_neuronal':
                model_buffer = io.BytesIO()
                # Guardar SOLO el state_dict (los pesos)
                torch.save(modelo_entrenado.state_dict(), model_buffer)
                model_buffer.seek(0)
                artefactos_guardados['model_statedict.pth'] = model_buffer
                artefactos_info['model_path'] = f"{base_path}/model_statedict.pth"
                # NUEVO: Guardar info necesaria para reconstruir la NN
                artefactos_info['nn_input_size'] = X_train_t.shape[1]
                artefactos_info['nn_num_classes'] = num_classes_detected
                artefactos_info['nn_is_regression'] = not es_clasificacion

            elif tipo_modelo_usuario == 'regresion': # Random Forest (Clasificador o Regresor)
                model_buffer = io.BytesIO()
                joblib.dump(modelo_entrenado, model_buffer)
                model_buffer.seek(0)
                artefactos_guardados['model.joblib'] = model_buffer
                artefactos_info['model_path'] = f"{base_path}/model.joblib"

            # Subir todos los artefactos guardados a Supabase
            print(f"-> Subiendo {len(artefactos_guardados)} artefactos a Supabase Storage/{ARTEFACTOS_BUCKET_NAME}...")
            storage_client = supabase.storage.from_(ARTEFACTOS_BUCKET_NAME)
            urls_artefactos = {}

            for filename, buffer in artefactos_guardados.items():
                storage_path = f"{base_path}/{filename}"
                print(f"   Subiendo {filename} a {storage_path}...")
                # Asegurarse de que el mimetype sea genérico para archivos binarios
                storage_client.upload(storage_path, buffer.getvalue(), {"content-type": "application/octet-stream"})
                # Obtener la URL pública
                public_url = storage_client.get_public_url(storage_path)
                urls_artefactos[filename] = public_url
                print(f"   -> URL: {public_url}")

            # Añadir las URLs al diccionario de info (esto es lo que irá a la DB)
            artefactos_info['urls'] = urls_artefactos
            print("-> Artefactos subidos exitosamente.")

        except Exception as save_err:
            print(f"🔥🔥🔥 ERROR CRÍTICO al guardar/subir artefactos: {save_err}")
            # Decidir si fallar todo el experimento o solo continuar sin artefactos
            artefactos_info = {"error": f"Fallo al guardar artefactos: {str(save_err)}"}
            # Podrías querer cambiar el estado del experimento aquí también


        # --- 6. Guardar el experimento completo (AHORA CON artefactos_info) ---
        estado_experimento = 'completado' if 'error' not in artefactos_info else 'completado_sin_artefactos'
        nuevo_experimento = {
            'id': experimento_id,
            'nombre': f"Experimento_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'dataset_id': dataset_id,
            'configuracion': json.dumps(config),
            'estado': estado_experimento,
            'fecha_creacion': datetime.utcnow().isoformat(),
            'columnas_entrada': json.dumps(config.get('columnas_entrada', [])),
            'columna_objetivo': config.get('columna_objetivo'),
            # NUEVO: Convertir artefactos_info a JSON string para guardarlo
            'artefactos_info': json.dumps(artefactos_info),
            'metricas': json.dumps({k: float(v) if v is not None and np.isfinite(v) else None for k, v in metricas.items()}),
            'metricas_por_epoca': json.dumps(metricas_por_epoca),
            'tiempo_por_epoca': json.dumps(tiempos_por_epoca),
            'matriz_confusion': json.dumps(matriz_confusion),
            'curva_roc': json.dumps(curva_roc),
            'distribucion_errores': json.dumps(distribucion_errores),
            'predicciones_vs_reales': json.dumps(predicciones_vs_reales),
            'importancia_features': json.dumps(importancia_features),
        }

        result = supabase.table('experimentos').insert(nuevo_experimento).execute()
        # NUEVO: Mejor manejo de errores de Supabase
        if not result.data or len(result.data) == 0:
             # Intenta obtener el error de la respuesta si existe
             db_error = result.get('error') if hasattr(result, 'get') else 'Error desconocido de Supabase al insertar.'
             raise Exception(f"No se pudo guardar el experimento en la base de datos: {db_error}")

        print("🎉 Entrenamiento completado y guardado (con artefactos).")
        return result.data[0]

    except Exception as e:
        # ... (Manejo de errores sin cambios) ...
        print(f"🔥🔥🔥 Error detallado en el servicio de entrenamiento: {e}")
        estado_experimento = 'error'
        experimento_fallido = {
            'id': experimento_id,
            'nombre': f"Experimento Fallido - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            'dataset_id': config.get('dataset_id'),
            'configuracion': json.dumps(config),
            'estado': estado_experimento,
            'fecha_creacion': datetime.utcnow().isoformat(),
            'columnas_entrada': json.dumps(config.get('columnas_entrada', [])),
            'columna_objetivo': config.get('columna_objetivo'),
            # NUEVO: Guardar info de artefactos aunque falle (puede ser útil)
            'artefactos_info': json.dumps({"error": f"Entrenamiento falló: {str(e)}"}),
            'metricas': json.dumps({'error': str(e)})
        }
        supabase.table('experimentos').insert(experimento_fallido).execute()
        raise e

