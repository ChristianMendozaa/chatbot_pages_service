# app/deps/firebase.py
import os
from functools import lru_cache
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, auth, firestore

def get_firestore():
    """Devuelve una instancia del cliente Firestore."""
    return firestore.client()

def _service_account_from_env():
    pk = os.getenv("FIREBASE_PRIVATE_KEY", "").replace("\\n", "\n")
    if not pk.strip().startswith("-----BEGIN"):
        # ayuda para detectar clave mal formateada
        raise RuntimeError("FIREBASE_PRIVATE_KEY ausente o sin saltos de línea '\\n'. Revísalo en .env")
    return {
        "type": os.getenv("FIREBASE_TYPE", "service_account"),
        "project_id": os.getenv("FIREBASE_PROJECT_ID"),
        "private_key_id": os.getenv("FIREBASE_PRIVATE_KEY_ID"),
        "private_key": pk,
        "client_email": os.getenv("FIREBASE_CLIENT_EMAIL"),
        "client_id": os.getenv("FIREBASE_CLIENT_ID"),
        "auth_uri": os.getenv("FIREBASE_AUTH_URI"),
        "token_uri": os.getenv("FIREBASE_TOKEN_URI"),
        "auth_provider_x509_cert_url": os.getenv("FIREBASE_AUTH_PROVIDER_X509_CERT_URL"),
        "client_x509_cert_url": os.getenv("FIREBASE_CLIENT_X509_CERT_URL"),
    }

def _init_firebase_if_needed():
    """Inicializa la app por defecto si aún no está inicializada."""
    if not firebase_admin._apps:
        cred = credentials.Certificate(_service_account_from_env())
        firebase_admin.initialize_app(cred, {"projectId": os.getenv("FIREBASE_PROJECT_ID")})

@lru_cache
def get_firebase():
    _init_firebase_if_needed()
    return firestore.client()

def verify_session_cookie(session_cookie: str) -> str:
    """Devuelve uid si el cookie es válido; lanza excepción si no."""
    _init_firebase_if_needed()  # <-- 💡 asegúrate de estar inicializado
    decoded = auth.verify_session_cookie(session_cookie, check_revoked=True)
    return decoded["uid"]

def get_pages_collection():
    db = get_firebase()
    return db.collection("pages")

def find_page_by_nickname(nickname: str):
    pages = get_pages_collection().where("nickname", "==", nickname).limit(1).stream()
    for doc in pages:
        data = doc.to_dict()
        return doc.reference, data
    return None, None

def set_chatbot_active(doc_ref, active: bool, extra: dict | None = None):
    payload = {"chatbotActive": active, "updatedAt": datetime.utcnow()}
    if extra:
        payload.update(extra)
    doc_ref.set(payload, merge=True)
