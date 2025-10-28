import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")
    PORT = int(os.getenv("PORT", 5000))
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
