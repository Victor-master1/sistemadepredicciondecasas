from flask import request, jsonify, Blueprint
from dotenv import load_dotenv
from supabase import create_client, Client
import os
import json
import uuid
import jwt
import time
from datetime import datetime, timedelta, timezone
import traceback

from .auth_logic import decode_and_process_image, euclidean_distance, RECOGNITION_THRESHOLD

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
TABLE_NAME = "facial_profiles"
SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET")

if not SUPABASE_JWT_SECRET:
    print("🚨 ERROR CRÍTICO: SUPABASE_JWT_SECRET no está configurado en el archivo .env")
else:
    print("✅ SUPABASE_JWT_SECRET cargado.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

facial_bp = Blueprint('facial_api', __name__)

def create_supabase_session_token(user_id: str, email: str = "dummy@facial.local") -> str | None:
    if not SUPABASE_JWT_SECRET:
        print("❌ Error: SUPABASE_JWT_SECRET no disponible para generar token.")
        return None

    payload = {
        'sub': user_id,
        'aud': 'authenticated',
        'role': 'authenticated',
        'email': email,
        'exp': datetime.now(timezone.utc) + timedelta(hours=1),
        'iat': time.time()
    }

    try:
        token = jwt.encode(payload, SUPABASE_JWT_SECRET, algorithm='HS256')
        print(f"✅ Token JWT generado para user_id: {user_id}")
        return token
    except Exception as e:
        print(f"❌ Error al generar JWT: {e}")
        traceback.print_exc()
        return None

@facial_bp.route('/api/facial_auth', methods=['POST'])
def facial_auth():
    data = request.json
    mode = data.get('mode')
    full_name = data.get('full_name')
    image_base64 = data.get('image_base64')

    if not full_name or not image_base64:
        return jsonify({"message": "Faltan campos (nombre o imagen)."}), 400

    current_embedding, error_msg = decode_and_process_image(image_base64)
    if error_msg:
        return jsonify({"message": error_msg}), 400

    if mode == 'register':
        print(f"\n--- Iniciando Registro SEGURO para: {full_name} ---")
        try:
            print(f"Verificando si '{full_name}' ya existe...")
            existing_profile = supabase.table(TABLE_NAME).select("id").eq("full_name", full_name).maybe_single().execute()

            if existing_profile and existing_profile.data:
                print(f"'{full_name}' ya está registrado.")
                return jsonify({"message": f"El nombre '{full_name}' ya está registrado."}), 409

            print("El nombre está disponible.")

            dummy_email = f"{uuid.uuid4()}@example.com"
            random_password = str(uuid.uuid4())
            print(f"Creando usuario Auth con email dummy: {dummy_email}")

            auth_response = supabase.auth.admin.create_user({
                "email": dummy_email,
                "password": random_password,
                "user_metadata": {"full_name": full_name},
                "email_confirm": True
            })

            if not auth_response.user:
                print(f"Error al crear usuario Auth: {auth_response}")
                error_detail = getattr(auth_response, 'error', {}).get('message', 'No se pudo crear el usuario Auth.')
                return jsonify({"message": f"Error al crear usuario de autenticación: {error_detail}"}), 500

            auth_user_id = auth_response.user.id
            print(f"Usuario Auth creado con ID: {auth_user_id}")

            insert_data = {
                "full_name": full_name,
                "facial_embedding": json.dumps(current_embedding),
                "auth_user_id": auth_user_id
            }

            print(f"Insertando perfil facial vinculado: {insert_data}")
            profile_response = supabase.table(TABLE_NAME).insert(insert_data).execute()
            print(f"Respuesta de Supabase (Insertar Perfil): {profile_response}")

            if hasattr(profile_response, 'data') and profile_response.data:
                print("Perfil facial registrado y vinculado exitosamente.")
                return jsonify({
                    "message": f"Registro facial y de usuario exitoso para {full_name}.",
                    "action": "registered"
                }), 201
            else:
                print("Error: No se pudo guardar el perfil facial después de crear el usuario Auth.")
                try:
                    print(f"Intentando borrar usuario Auth huérfano: {auth_user_id}")
                    supabase.auth.admin.delete_user(auth_user_id)
                    print("Usuario Auth huérfano borrado.")
                except Exception as admin_error:
                    print(f"ADVERTENCIA: No se pudo borrar el usuario Auth huérfano {auth_user_id}: {admin_error}")

                return jsonify({"message": "Error al guardar perfil facial."}), 500

        except Exception as e:
            print(f"❌ ERROR EXCEPCIÓN en Registro: {e}")
            traceback.print_exc()
            return jsonify({"message": f"Error interno del servidor durante el registro."}), 500

    elif mode == 'login':
        print(f"\n--- Iniciando Login SEGURO para: {full_name} ---")
        try:
            print(f"Buscando perfil para: {full_name}")
            result = supabase.table(TABLE_NAME).select("facial_embedding, auth_user_id, full_name").eq("full_name", full_name).maybe_single().execute()

            if not result or not result.data:
                print(f"Perfil no encontrado para: {full_name}")
                return jsonify({"message": f"Usuario '{full_name}' no encontrado."}), 404

            print(f"Perfil encontrado. Datos: {result.data}")

            stored_embedding_json = result.data.get('facial_embedding')
            auth_user_id = result.data.get('auth_user_id')
            user_full_name = result.data.get('full_name')

            if not auth_user_id:
                print("Error: auth_user_id nulo en el perfil. (Registro antiguo)")
                return jsonify({"message": "Perfil facial no vinculado. Por favor, regístrese de nuevo."}), 500

            if not stored_embedding_json:
                print("Error: facial_embedding nulo en el perfil.")
                return jsonify({"message": "Embedding facial no encontrado en el perfil."}), 500

            stored_embedding = json.loads(stored_embedding_json) if isinstance(stored_embedding_json, str) else stored_embedding_json

            if not isinstance(current_embedding, list) or not isinstance(stored_embedding, list):
                print("Error: Formato de embedding inválido.")
                return jsonify({"message": "Formato de embedding inválido."}), 500

            distance = euclidean_distance(current_embedding, stored_embedding)
            print(f"Distancia calculada: {distance:.4f} (Umbral: {RECOGNITION_THRESHOLD})")

            if distance < RECOGNITION_THRESHOLD:
                print(f"Reconocimiento facial exitoso para {full_name}. Generando token...")
                session_token = create_supabase_session_token(auth_user_id)

                if not session_token:
                    print("Fallo al generar el token JWT.")
                    return jsonify({"message": "Error interno al generar token de sesión."}), 500

                return jsonify({
                    "message": f"Acceso concedido a {full_name}!",
                    "token": session_token,
                    "user_id": auth_user_id,
                    "full_name": user_full_name,
                    "distance": distance
                }), 200
            else:
                print("Rostro no coincide con el perfil registrado.")
                return jsonify({"message": f"Rostro no coincide. Distancia: {distance:.4f}"}), 401

        except Exception as e:
            print(f"❌ ERROR INESPERADO en Login: {e}")
            traceback.print_exc()
            return jsonify({"message": f"Error interno del servidor durante el login."}), 500

    return jsonify({"message": "Modo no soportado."}), 400