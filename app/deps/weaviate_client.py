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

    if _is_local_url(url):
        host_port = url.replace("http://", "").replace("https://", "")
        host, port = (host_port.split(":") + ["8080"])[:2]
        grpc_port = 50051
        client = weaviate.connect_to_local(
            host=host, port=int(port), grpc_port=grpc_port
        )
    else:
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
            # ❌ vector_config=Configure.Vector(...)  ->  ✅ usar estos dos:
            vectorizer_config=Configure.Vectorizer.none(),
            vector_index_config=Configure.VectorIndex.hnsw(
                distance_metric=VectorDistances.COSINE
            ),
            multi_tenancy_config=Configure.multi_tenancy(enabled=True),
        )
        client.collections.get(COLLECTION_NAME)  # fuerza lazy init
    except WeaviateBaseError as e:
        raise RuntimeError(f"No se pudo crear la colección '{COLLECTION_NAME}': {e}")

def ensure_tenant(nickname: str):
    ensure_collection()
    client = get_wv_client()
    col = client.collections.get(COLLECTION_NAME)

    try:
        if hasattr(col.tenants, "create"):
            col.tenants.create(Tenant(name=nickname))
        elif hasattr(col.tenants, "add"):
            col.tenants.add([Tenant(name=nickname)])
        else:
            # fallback raro
            raise RuntimeError("El SDK de Weaviate no expone create/add para tenants.")
    except WeaviateBaseError as e:
        msg = str(e).lower()
        if "already exists" in msg or "conflict" in msg:
            return
        if "class not found" in msg:
            raise RuntimeError("La colección DocChunk no existe (class not found). Revisa ensure_collection().")
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