from supabase import create_client, Client
import os
from dotenv import load_dotenv

# Carga las variables del archivo .env
load_dotenv()

# Obtiene las credenciales desde las variables de entorno
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Crea el cliente Supabase
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
