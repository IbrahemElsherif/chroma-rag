from __future__ import annotations
import os, time, json, threading
from typing import List, Dict, Any, Optional, Generator, Tuple

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from src.app.settings import get_settings, Settings
from src.app.rag_chain import RAGChain
from src.ingest.loader import load_documents
from src.ingest.splitter import split_documents
from src.ingest.index import build_or_update_index

# =======================
# Basic API setup
# =======================
app = FastAPI(title="RAG API", version="0.1.0")
settings: Settings = get_settings()
rag = RAGChain(settings)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ضيّقه لدومين ووردبرس لاحقًا
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY = os.getenv("API_KEY", "CHANGE_ME")

def require_api_key(x_api_key: str = Header(None)):
    if not API_KEY or x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return True

# =======================
# Minimal in-memory TTL cache (thread-safe)
# =======================
CACHE_TTL = int(os.getenv("CACHE_TTL", "300"))      # ثواني، افتراضي 5 دقائق
CACHE_MAX = int(os.getenv("CACHE_MAX", "512"))      # أقصى عناصر
_cache_lock = threading.Lock()
_cache_store: Dict[str, Tuple[float, str]] = {}     # key -> (expires_at, answer)

def _norm(s: str) -> str:
    # تبسيط للمفتاح (اسئلة متماثلة → نفس الكاش)
    return " ".join(s.strip().lower().split())

def _cache_get(key: str) -> Optional[str]:
    now = time.time()
    with _cache_lock:
        item = _cache_store.get(key)
        if not item:
            return None
        exp, val = item
        if now > exp:
            _cache_store.pop(key, None)
            return None
        return val

def _cache_set(key: str, value: str):
    exp = time.time() + CACHE_TTL
    with _cache_lock:
        if len(_cache_store) >= CACHE_MAX:
            # evict أقدم عنصر تقريبًا (بسيطة)
            oldest = min(_cache_store.items(), key=lambda kv: kv[1][0])[0]
            _cache_store.pop(oldest, None)
        _cache_store[key] = (exp, value)

# =======================
# Schemas
# =======================
class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    history: Optional[List[Dict[str, Any]]] = None
    k: Optional[int] = None
    cache_bypass: Optional[bool] = False  # لو true يتجاهل الكاش

class ChatResponse(BaseModel):
    answer: str

class IngestResponse(BaseModel):
    added_chunks: int

# =======================
# Endpoints
# =======================
@app.get("/health")
def health():
    return {
        "status": "ok",
        "vectorstore": settings.chroma_path,
        "collection": settings.collection_name,
        "cache_ttl": CACHE_TTL,
        "cache_items": len(_cache_store),
    }

@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(require_api_key)])
def chat(req: ChatRequest):
    # override k لو حابب
    if req.k:
        rag.retriever.search_kwargs["k"] = req.k

    # مفتاح الكاش: السؤال + k + (هاش بسيط من history)
    hist_sig = ""
    if req.history:
        try:
            hist_sig = str(hash(json.dumps(req.history, ensure_ascii=False)))
        except Exception:
            hist_sig = str(hash(str(req.history)))
    key = f"q:{_norm(req.question)}|k:{rag.retriever.search_kwargs.get('k')}|h:{hist_sig}"

    # محاولة من الكاش
    if not req.cache_bypass:
        cached = _cache_get(key)
        if cached is not None:
            return {"answer": cached}

    # إنتاج الإجابة
    partial = ""
    for chunk in rag.stream_answer(req.question, history_messages=req.history or []):
        partial = chunk or partial

    # حفظ في الكاش
    if partial:
        _cache_set(key, partial)

    return {"answer": partial}

@app.post("/chat/stream", dependencies=[Depends(require_api_key)])
def chat_stream(req: ChatRequest):
    if req.k:
        rag.retriever.search_kwargs["k"] = req.k

    hist_sig = ""
    if req.history:
        try:
            hist_sig = str(hash(json.dumps(req.history, ensure_ascii=False)))
        except Exception:
            hist_sig = str(hash(str(req.history)))
    key = f"q:{_norm(req.question)}|k:{rag.retriever.search_kwargs.get('k')}|h:{hist_sig}"

    # لو في كاش، هنرجعه كـ stream سريع (سطر واحد)
    cached = None if req.cache_bypass else _cache_get(key)

    def gen() -> Generator[bytes, None, None]:
        try:
            if cached is not None:
                data = {"role": "assistant", "content": cached, "cached": True}
                yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n".encode("utf-8")
                return

            last = ""
            for chunk in rag.stream_answer(req.question, history_messages=req.history or []):
                last = chunk or last
                data = {"role": "assistant", "content": chunk}
                yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n".encode("utf-8")

            if last:
                _cache_set(key, last)
        except Exception as e:
            err = {"error": str(e)}
            yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n".encode("utf-8")

    return StreamingResponse(gen(), media_type="text/event-stream")

@app.post("/ingest", response_model=IngestResponse, dependencies=[Depends(require_api_key)])
async def ingest(file: Optional[UploadFile] = File(None), directory: Optional[str] = Form(None)):
    """
    - لو أرسلت 'file': نحفظه داخل DATA_RAW_DIR ونفهرس المجلد كله.
    - أو مرّر 'directory' (نسبي) داخل DATA_RAW_DIR.
    """
    from pathlib import Path
    target_dir = Path(settings.data_raw_dir)

    if file:
        target_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = target_dir / file.filename
        with open(tmp_path, "wb") as f:
            f.write(await file.read())
        data_dir = str(target_dir)
    elif directory:
        data_dir = str(Path(directory))
    else:
        raise HTTPException(status_code=400, detail="Provide either a file or a directory")

    raw_docs = load_documents(data_dir)
    chunks = split_documents(raw_docs, settings)
    added = build_or_update_index(chunks, settings)
    # تفريغ الكاش (المعرفة تغيّرت)
    with _cache_lock:
        _cache_store.clear()
    return {"added_chunks": added}
