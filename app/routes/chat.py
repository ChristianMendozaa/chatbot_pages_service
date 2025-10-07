import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from groq import Groq
from ..deps.firebase import find_page_by_nickname
from ..rag.service import retrieve

# Usaremos tiktoken para contar tokens de manera precisa (instálalo con pip install tiktoken)
import tiktoken

router = APIRouter(tags=["chat"])

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

# 🔢 Límite de tokens para la pregunta del usuario
MAX_INPUT_TOKENS = int(os.getenv("MAX_INPUT_TOKENS", "300"))

class ChatBody(BaseModel):
    nickname: str = Field(..., min_length=3)
    question: str = Field(..., min_length=2)

SYSTEM_PROMPT = (
    "Eres un asistente de una página. Responde con claridad y concisión usando SOLO el contexto. "
    "Si el contexto no contiene la respuesta, di que no está disponible."
)

def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """Cuenta tokens usando tiktoken (o una estimación si falla)."""
    try:
        enc = tiktoken.encoding_for_model(model)
        return len(enc.encode(text))
    except Exception:
        return len(text.split())

def truncate_to_tokens(text: str, max_tokens: int, model: str = "gpt-3.5-turbo") -> str:
    """Trunca el texto para no superar cierta cantidad de tokens."""
    try:
        enc = tiktoken.encoding_for_model(model)
        tokens = enc.encode(text)
        if len(tokens) <= max_tokens:
            return text
        truncated = enc.decode(tokens[:max_tokens])
        return truncated
    except Exception:
        # fallback simple si falla tiktoken
        return text[:max_tokens * 4]  # aprox 4 chars/token

@router.post("/chat")
def chat(body: ChatBody):
    # 1️⃣ Validar chatbot activo
    doc_ref, data = find_page_by_nickname(body.nickname)
    if not data or not data.get("chatbotActive", False):
        raise HTTPException(status_code=404, detail="Chatbot no activo para este nickname")

    # 2️⃣ Limitar tokens de pregunta
    token_count = count_tokens(body.question, GROQ_MODEL)
    if token_count > MAX_INPUT_TOKENS:
        body.question = truncate_to_tokens(body.question, MAX_INPUT_TOKENS, GROQ_MODEL)

    # 3️⃣ Recuperar contexto limitado
    docs = retrieve(body.nickname, body.question, k=3)
    context = "\n\n".join([f"[{i}] {t}" for (t, i) in docs]) or "(sin contexto)"

    # 4️⃣ Construir prompt
    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"### CONTEXTO\n{context}\n\n"
        f"### PREGUNTA\n{body.question}\n\n"
        f"### INSTRUCCIONES\n"
        f"Si no hay respuesta en el contexto, dilo claramente."
    )

    # 5️⃣ Llamar al modelo Groq
    resp = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
        max_tokens=500,
    )

    # 6️⃣ Extraer respuesta y métricas
    answer = resp.choices[0].message.content
    usage = getattr(resp, "usage", None)

    return {
        "answer": answer,
        "sources": [{"index": i} for (_, i) in docs],
        "tokens": {
            "input_tokens": getattr(usage, "prompt_tokens", None),
            "output_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
            "question_tokens": token_count,
        },
    }
