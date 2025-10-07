import os
from functools import lru_cache
import weaviate
from weaviate.classes.config import Configure, Property, DataType, VectorDistances
from weaviate.classes.tenants import Tenant
from weaviate.exceptions import WeaviateBaseError

COLLECTION_NAME = "DocChunk"

def _is_local_url(url: str) -> bool:
    return url.startswith("http://localhost") or url.startswith("http://127.0.0.1") or url.startswith("http://0.0.0.0")

@lru_cache
def get_wv_client():
    url = os.getenv("WEAVIATE_URL")
    key = os.getenv("WEAVIATE_API_KEY")
    if not url:
        raise RuntimeError("WEAVIATE_URL no configurada")

    # Conexión correcta según destino
    if _is_local_url(url):
        # Local/self-hosted (sin Auth por API Key)
        client = weaviate.connect_to_local(
            http_host=url.replace("http://", "").replace("https://", ""),  # p.ej. "localhost:8080"
            grpc_host=url.replace("http://", "").replace("https://", "").split(":")[0] + ":50051",
        )
    else:
        # Weaviate Cloud / cluster con API Key
        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=url,
            auth_credentials=weaviate.auth.AuthApiKey(key) if key else None,
        )
    return client

def _collection_exists(name: str) -> bool:
    client = get_wv_client()
    try:
        # v4: exists helper
        return client.collections.exists(name)
    except AttributeError:
        # fallback: intentar get -> si falla, no existe
        try:
            client.collections.get(name)
            return True
        except Exception:
            return False
        
def ensure_collection():
    client = get_wv_client()
    if _collection_exists(COLLECTION_NAME):
        return

    try:
        client.collections.create(
            name=COLLECTION_NAME,
            properties=[
                Property(name="text", data_type=DataType.TEXT),
                Property(name="source", data_type=DataType.TEXT),
                Property(name="chunk_index", data_type=DataType.INT),
            ],
            vector_config=Configure.Vector(
                distance_metric=VectorDistances.COSINE,
                vectorizer=Configure.Vectorizer.none(),  # ya no se usa vectorizer_config separado
                vector_index_type=Configure.VectorIndex.hnsw(),
            ),
            multi_tenancy_config=Configure.multi_tenancy(enabled=True),
        )
        client.collections.get(COLLECTION_NAME)
    except WeaviateBaseError as e:
        raise RuntimeError(f"No se pudo crear la colección '{COLLECTION_NAME}': {e}")

def ensure_tenant(nickname: str):
    """
    Upsert de tenant:
    - Asegura que la colección existe.
    - Intenta crear el tenant; si ya existe, continúa sin fallar.
    """
    ensure_collection()
    client = get_wv_client()
    col = client.collections.get(COLLECTION_NAME)

    try:
        # Crear directamente; si ya existe, Weaviate lanza error -> lo ignoramos
        col.tenants.create(Tenant(name=nickname))
    except WeaviateBaseError as e:
        msg = str(e).lower()
        if "already exists" in msg or "conflict" in msg:
            # Tenant ya creado: OK
            return
        # Si el error es “class not found” aquí, es que la colección no quedó bien creada.
        if "class not found" in msg:
            raise RuntimeError("La colección DocChunk no existe (class not found). Revisa ensure_collection().")
        # Otros errores sí los propagamos
        raise

def delete_tenant(nickname: str):
    ensure_collection()
    client = get_wv_client()
    col = client.collections.get(COLLECTION_NAME)

    # si no existe, idempotente
    try:
        existing = [t.name for t in (col.tenants.get() or [])]
        if nickname not in existing:
            return
    except Exception:
        # si falla get(), intentamos borrar igual
        pass

    # métodos posibles: delete / remove
    if hasattr(col.tenants, "delete"):
        col.tenants.delete(nickname)
        return
    if hasattr(col.tenants, "remove"):
        col.tenants.remove(nickname)
        return

    # último recurso: método alterno en el objeto colección
    if hasattr(col, "delete_tenant"):
        col.delete_tenant(nickname)
        return

    raise RuntimeError("Tu SDK de Weaviate no expone delete/remove para tenants. Actualiza a weaviate-client >= 4.9.")