# app/routes/chat.py
import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from groq import Groq
from ..deps.firebase import find_page_by_nickname
from ..rag.service import retrieve
import tiktoken
from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0  # detección consistente

router = APIRouter(tags=["chat"])

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

MAX_INPUT_TOKENS = int(os.getenv("MAX_INPUT_TOKENS", "300"))

class ChatBody(BaseModel):
    nickname: str = Field(..., min_length=3)
    question: str = Field(..., min_length=2)

def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    try:
        enc = tiktoken.encoding_for_model(model)
        return len(enc.encode(text))
    except Exception:
        return len(text.split())

def truncate_to_tokens(text: str, max_tokens: int, model: str = "gpt-3.5-turbo") -> str:
    try:
        enc = tiktoken.encoding_for_model(model)
        toks = enc.encode(text)
        if len(toks) <= max_tokens:
            return text
        return enc.decode(toks[:max_tokens])
    except Exception:
        return text[:max_tokens * 4]

def detect_language(text: str) -> str:
    try:
        return detect(text)
    except Exception:
        return "es"

@router.post("/chat")
def chat(body: ChatBody):
    # 1️⃣ Validar chatbot activo
    doc_ref, data = find_page_by_nickname(body.nickname)
    if not data or not data.get("chatbotActive", False):
        raise HTTPException(status_code=404, detail="Chatbot no activo para este nickname")

    # 2️⃣ Limitar tokens de input
    token_count = count_tokens(body.question, GROQ_MODEL)
    if token_count > MAX_INPUT_TOKENS:
        body.question = truncate_to_tokens(body.question, MAX_INPUT_TOKENS, GROQ_MODEL)

    # 3️⃣ Detectar idioma del usuario
    lang = detect_language(body.question)
    lang_name = {
        "es": "Spanish",
        "en": "English",
        "fr": "French",
        "pt": "Portuguese",
        "de": "German",
        "it": "Italian"
    }.get(lang, "Spanish")

    # 4️⃣ Obtener contexto (sin etiquetas ni índices)
    docs = retrieve(body.nickname, body.question, k=3)
    context = "\n".join([t for (t, _) in docs]) or "(no context found)"

    # 5️⃣ Prompt optimizado
    system_prompt = (
        f"You are a helpful assistant that always responds in {lang_name}. "
        "You must summarize information ONLY from the provided context. "
        "Do not copy the sentences verbatim — instead, explain them naturally in the target language. "
        "Do not include any Spanish text if the answer is in English, and do not mention the context explicitly. "
        "If there is no relevant information in the context, reply exactly: 'Not available.'"
    )

    user_prompt = (
        f"Context:\n{context}\n\n"
        f"Question: {body.question}\n\n"
        f"Answer naturally in {lang_name}. Avoid parentheses, citations, or translation notes."
    )

    # 6️⃣ Llamar al modelo
    resp = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.4,
        max_tokens=400,
    )

    answer = resp.choices[0].message.content.strip()
    usage = getattr(resp, "usage", None)

    return {
        "answer": answer,
        "language": lang_name,
        "sources": [{"index": i} for (_, i) in docs],
        "tokens": {
            "input_tokens": getattr(usage, "prompt_tokens", None),
            "output_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
            "question_tokens": token_count,
        },
    }
