from typing import List, Optional
import os

try:
    import tiktoken
    _HAS_TIKTOKEN = True
except Exception:
    _HAS_TIKTOKEN = False

def _get_encoding_for_model(model: str):
    """
    Usa 'cl100k_base' (la misma de text-embedding-ada-002 y text-embedding-3*).
    """
    enc_name = "cl100k_base"
    try:
        return tiktoken.get_encoding(enc_name)
    except Exception:
        return tiktoken.get_encoding("cl100k_base")

def _encode(text: str, model_hint: Optional[str] = None):
    if not _HAS_TIKTOKEN:
        return None  # sin tiktoken, se hará fallback por caracteres
    enc = _get_encoding_for_model(model_hint or "")
    return enc.encode(text)

def _decode(tokens: List[int], model_hint: Optional[str] = None) -> str:
    enc = _get_encoding_for_model(model_hint or "")
    return enc.decode(tokens)

def chunk_text(
    text: str,
    size_tokens: int = 400,
    overlap_tokens: int = 100,
    model_hint: Optional[str] = None,
) -> List[str]:
    """
    Divide `text` en chunks por TOKENS. Si no hay tiktoken, hace fallback por caracteres
    (aprox 4 chars ~ 1 token).
    """
    text = text.strip()
    if not text:
        return []

    if _HAS_TIKTOKEN:
        toks = _encode(text, model_hint=model_hint)
        if toks is None:
            toks = []
        chunks: List[str] = []
        i = 0
        n = len(toks)
        step = max(1, size_tokens - overlap_tokens)
        while i < n:
            end = min(i + size_tokens, n)
            piece_tokens = toks[i:end]
            chunks.append(_decode(piece_tokens, model_hint=model_hint))
            i += step
        return chunks

    # 🔁 Fallback por caracteres (aprox)
    approx_chars_per_token = 4
    size_chars = size_tokens * approx_chars_per_token
    overlap_chars = overlap_tokens * approx_chars_per_token
    chunks: List[str] = []
    i = 0
    n = len(text)
    step = max(1, size_chars - overlap_chars)
    while i < n:
        end = min(i + size_chars, n)
        chunks.append(text[i:end])
        i += step
    return chunks
