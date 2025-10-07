import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv, find_dotenv
from .deps.weaviate_client import get_wv_client, ensure_collection
from .deps.firebase import get_firebase

load_dotenv(find_dotenv(), override=False)

REQUIRED_ENV = ["WEAVIATE_URL", "WEAVIATE_API_KEY", "OPENAI_API_KEY", "GROQ_API_KEY"]

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup ---
    missing = [k for k in REQUIRED_ENV if not os.getenv(k)]
    if missing:
        raise RuntimeError(f"Faltan variables de entorno: {', '.join(missing)}")

    # inicializar dependencias
    get_firebase()
    ensure_collection()
    wv_client = get_wv_client()
    yield

    # --- Shutdown ---
    try:
        wv_client.close()
    except Exception as e:
        print("⚠️ Error al cerrar conexión Weaviate:", e)

app = FastAPI(title="Pages Chatbots API", lifespan=lifespan)

# Configurar CORS
origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.state.session_cookie_name = os.getenv("SESSION_COOKIE_NAME", "__session")

# Rutas
from .routes import chatbot, chat
app.include_router(chatbot.router)
app.include_router(chat.router)

@app.get("/health")
def health():
    return {"ok": True}
