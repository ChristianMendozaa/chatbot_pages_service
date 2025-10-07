# app/routes/chatbot.py
from fastapi import APIRouter, HTTPException, Request, UploadFile, File, Form
from pydantic import BaseModel, Field
from ..deps.firebase import verify_session_cookie, find_page_by_nickname, set_chatbot_active
from ..deps.weaviate_client import ensure_collection, ensure_tenant, delete_tenant
from ..rag.service import ingest_text
import fitz  # PyMuPDF

router = APIRouter(prefix="/chatbot", tags=["chatbot"])

class ActivateBody(BaseModel):
    nickname: str = Field(..., min_length=3, max_length=50)
    text: str = Field(..., min_length=20)
    clear_existing: bool = False

class DeactivateBody(BaseModel):
    nickname: str

def _require_owner(req: Request, nickname: str):
    """
    Verifica que el request tenga cookie válida y que el usuario sea dueño de la página.
    Si la cookie no existe o es inválida, lanza HTTP 401.
    """
    cookie_name = req.app.state.session_cookie_name  # ej: "__session"
    sess = req.cookies.get(cookie_name)

    if not sess:
        raise HTTPException(status_code=401, detail="Falta la cookie de sesión.")

    try:
        uid = verify_session_cookie(sess)
    except Exception as e:
        raise HTTPException(
            status_code=401,
            detail=f"Cookie de sesión inválida o expirada: {str(e)}"
        )

    doc_ref, data = find_page_by_nickname(nickname)
    if not data:
        raise HTTPException(status_code=404, detail="Página no encontrada.")
    if data.get("uid") != uid:
        raise HTTPException(status_code=403, detail="No eres el dueño de esta página.")

    return doc_ref, data, uid

@router.post("/activate")
def activate_chatbot(
    req: Request,
    nickname: str = Form(...),
    text: str = Form(""),                 # opcional
    file: UploadFile | None = File(None)  # opcional (PDF)
):
    """
    Activa el chatbot para una página y permite subir texto y/o un PDF.
    La cookie de sesión se toma del header Cookie (req.cookies).
    """
    # 1) Validar cookie y dueño usando el helper que lee req.cookies
    doc_ref, data, uid = _require_owner(req, nickname)

    # 2) Armar contenido desde form-data (texto + PDF)
    content = (text or "").strip()
    if file:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Solo se aceptan archivos PDF.")
        pdf_text = ""
        with fitz.open(stream=file.file.read(), filetype="pdf") as pdf:
            for page in pdf:
                pdf_text += page.get_text("text") + "\n"
        content = (content + "\n" + pdf_text).strip()

    if len(content) < 100:
        raise HTTPException(status_code=400, detail="No se encontró suficiente texto válido.")

    # 3) Ingesta RAG
    ensure_tenant(nickname)
    n_chunks = ingest_text(nickname, content)

    # 4) Marcar como activo (usando el mismo doc_ref de la página)
    set_chatbot_active(doc_ref, True, extra={
        "chatbot": {"active": True, "tenant": nickname, "chunks": n_chunks}
    })

    return {
        "ok": True,
        "nickname": nickname,
        "chunks": n_chunks,
        "message": f"Chatbot activado con éxito para '{nickname}' ({n_chunks} fragmentos almacenados)."
    }

@router.post("/deactivate")
def deactivate(req: Request, body: DeactivateBody):
    ensure_collection()
    doc_ref, data, uid = _require_owner(req, body.nickname)
    delete_tenant(body.nickname)
    set_chatbot_active(doc_ref, False, extra={"chatbot": {"active": False}})
    return {"ok": True, "nickname": body.nickname, "deleted": "tenant"}
