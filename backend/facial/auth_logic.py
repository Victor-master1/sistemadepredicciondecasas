import numpy as np
import math
import mediapipe as mp
import cv2
import base64
from typing import List, Optional, Tuple

# Usamos el motor de Face Mesh
mp_face_mesh = mp.solutions.face_mesh

# ----------------- UMBRALES Y CONFIGURACIÓN -----------------
# Tolerancia para el reconocimiento (ajustable: 0.05 es estricto, 0.10 es más flexible)
RECOGNITION_THRESHOLD = 0.07 

# ----------------- FUNCIONES DE BIOMETRÍA -----------------

def normalize_vector(v: List[float]) -> List[float]:
    """Normalizes the vector to make comparison robust to size/position."""
    magnitude = math.sqrt(sum(c**2 for c in v))
    if magnitude == 0:
        return v
    
    # Returns the normalized vector
    return [c / magnitude for c in v]

def euclidean_distance(v1: List[float], v2: List[float]) -> float:
    """Calculates the Euclidean distance between two vectors."""
    v1_np = np.array(v1)
    v2_np = np.array(v2)
    return float(np.linalg.norm(v1_np - v2_np))

def decode_and_process_image(base64_string: str) -> Tuple[Optional[List[float]], Optional[str]]:
    """Decodes Base64, detects the face, and extracts the normalized embedding."""
    try:
        # 1. Decodificar Base64
        # Eliminar el prefijo 'data:image/jpeg;base64,' si existe
        if ',' in base64_string:
            header, encoded = base64_string.split(',', 1)
        else:
            encoded = base64_string
        
        image_data = base64.b64decode(encoded)
        
        # 2. Convertir a formato OpenCV
        np_arr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        # 3. Procesar con MediaPipe Face Mesh
        with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
            # Convertir a RGB para MediaPipe
            results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            
            if not results.multi_face_landmarks:
                return None, "No se detectó un rostro claro en la imagen."

            # 4. Extraer Landmarks
            landmarks = results.multi_face_landmarks[0].landmark
            
            # Crear un vector plano de coordenadas (x, y, z) - 468 * 3 = 1404 dimensiones
            flat_vector = [coord for landmark in landmarks for coord in (landmark.x, landmark.y, landmark.z)]
            
            # Devolver el vector normalizado
            return normalize_vector(flat_vector), None

    except Exception as e:
        print(f"Error en el procesamiento de imagen: {e}")
        return None, "Error interno al procesar la imagen."
