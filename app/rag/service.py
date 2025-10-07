# app/rag/service.py
import os
import inspect
from typing import List, Tuple
from openai import OpenAI
from .chunker import chunk_text
from ..deps.weaviate_client import get_wv_client, COLLECTION_NAME, ensure_tenant
from weaviate.classes.query import MetadataQuery

def _get_openai():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY no está configurada.")
    return OpenAI(api_key=api_key)

def _embed_model():
    # Asegúrate de que el modelo concuerde en dimension con tu colección (p.ej. ada-002 => 768)
    return os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")

def embed(texts: List[str]) -> List[List[float]]:
    client = _get_openai()
    resp = client.embeddings.create(model=_embed_model(), input=texts)
    return [d.embedding for d in resp.data]

def _set_tenant_param(kwargs: dict, func, nickname: str):
    sig = inspect.signature(func)
    if "tenant_name" in sig.parameters:
        kwargs["tenant_name"] = nickname
        return True
    if "tenant" in sig.parameters:
        kwargs["tenant"] = nickname
        return True
    return False

def ingest_text(nickname: str, raw_text: str) -> int:
    ensure_tenant(nickname)
    
    size_tokens = int(os.getenv("CHUNK_TOKENS", "400"))
    overlap_tokens = int(os.getenv("CHUNK_OVERLAP_TOKENS", "100"))
    chunks = chunk_text(
        raw_text,
        size_tokens=size_tokens,
        overlap_tokens=overlap_tokens,
        model_hint=_embed_model(),
    )

    if not chunks:
        return 0

    vectors = embed(chunks)
    client = get_wv_client()
    col = client.collections.get(COLLECTION_NAME)

    # 1) batch con tenant(_name)
    batch_kwargs = {}
    if _set_tenant_param(batch_kwargs, col.batch.dynamic, nickname):
        with col.batch.dynamic(**batch_kwargs) as batch:
            for idx, (c, v) in enumerate(zip(chunks, vectors)):
                batch.add_object(
                    properties={"text": c, "source": "upload", "chunk_index": idx},
                    vector=v,
                )
        return len(chunks)

    # 2) with_tenant (algunos SDK)
    scoped_col = None
    if hasattr(col, "with_tenant"):
        try:
            scoped_col = col.with_tenant(nickname)
            try:
                with scoped_col.batch.dynamic() as batch:
                    for idx, (c, v) in enumerate(zip(chunks, vectors)):
                        batch.add_object(
                            properties={"text": c, "source": "upload", "chunk_index": idx},
                            vector=v,
                        )
                return len(chunks)
            except TypeError:
                scoped_col = None
        except Exception:
            scoped_col = None

    # 3) insert_many o insert (fallback)
    objs = [
        {"properties": {"text": c, "source": "upload", "chunk_index": idx}, "vector": v}
        for idx, (c, v) in enumerate(zip(chunks, vectors))
    ]
    if hasattr(col.data, "insert_many"):
        insert_kwargs = {"objects": objs}
        if not _set_tenant_param(insert_kwargs, col.data.insert_many, nickname):
            if scoped_col and hasattr(scoped_col.data, "insert_many"):
                scoped_col.data.insert_many(objects=objs)
                return len(chunks)
        else:
            col.data.insert_many(**insert_kwargs)
            return len(chunks)

    inserted = 0
    if hasattr(col.data, "insert"):
        for o in objs:
            insert_kwargs = dict(o)
            if not _set_tenant_param(insert_kwargs, col.data.insert, nickname):
                if scoped_col and hasattr(scoped_col.data, "insert"):
                    scoped_col.data.insert(**o)
                    inserted += 1
                    continue
                raise RuntimeError(
                    "Tu versión del cliente Weaviate no permite especificar tenant en insert."
                )
            col.data.insert(**insert_kwargs)
            inserted += 1
        return inserted

    raise RuntimeError("Cliente Weaviate sin soporte multi-tenant en batch/insert.")

def retrieve(nickname: str, question: str, k: int | None = None) -> List[Tuple[str, int]]:
    limit_val = int(os.getenv("RAG_MAX_CHUNKS", "5")) if k is None else k

    q_vec = embed([question])[0]
    col = get_wv_client().collections.get(COLLECTION_NAME)

    query_kwargs = {
        "near_vector": q_vec,
        "limit": limit_val,
        "return_metadata": MetadataQuery(distance=True),
    }

    if _set_tenant_param(query_kwargs, col.query.near_vector, nickname):
        res = col.query.near_vector(**query_kwargs)
    else:
        if hasattr(col, "with_tenant"):
            sc = col.with_tenant(nickname)
            res = sc.query.near_vector(
                near_vector=q_vec,
                limit=limit_val,                    # 👈 aquí también
                return_metadata=MetadataQuery(distance=True),
            )
        else:
            raise RuntimeError("Tu cliente Weaviate no permite tenant en query.")

    out: List[Tuple[str, int]] = []
    for o in (res.objects or [])[:limit_val]:       # 👈 corte defensivo
        props = o.properties or {}
        out.append((props.get("text", ""), props.get("chunk_index", 0)))
    return out
