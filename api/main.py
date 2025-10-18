from __future__ import annotations
import os
import time
import json
import threading
from typing import List, Dict, Any, Optional, Generator, Tuple
import uvicorn

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
# API INITIALIZATION
# =======================
# FastAPI application instance for the RAG API
# Sets up middlewares and loads main global objects.
app = FastAPI(title="RAG API", version="0.1.0")

# Global application settings loaded from environment or config file.
settings: Settings = get_settings()

# Main chat and retrieval chain using LangChain + Chroma.
rag = RAGChain(settings)

# CORS middleware to allow frontends to connect. WARNING: allow_origins=["*"] enables public access.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request authentication - API key system
API_KEY = os.getenv("API_KEY", "CHANGE_ME")


def require_api_key(x_api_key: str = Header(None)) -> bool:
    """
    Dependency for protected endpoints.
    Verifies an incoming request's x-api-key header versus the configured API_KEY.

    Args:
        x_api_key (str): The authentication header sent by the client.
    Returns:
        bool: True if authentication succeeds. Otherwise, raises HTTPException.

    Example:
        curl -X POST -H 'x-api-key: mykey' ...
    """
    if not API_KEY or x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return True

# =======================
# IN-MEMORY TTL CACHE (thread-safe)
# =======================
CACHE_TTL = int(os.getenv("CACHE_TTL", "300"))      # Cache time-to-live in seconds (default 5 min)
CACHE_MAX = int(os.getenv("CACHE_MAX", "512"))      # Cache size limit

_cache_lock = threading.Lock()
_cache_store: Dict[str, Tuple[float, str]] = {}      # In-memory: key -> (expires_at, answer)


def _norm(s: str) -> str:
    """
    Normalize the cache key by stripping whitespace, converting to lowercase,
    and compressing all intermediate whitespace.
    This ensures that similar questions/calls will map to a single cache entry.

    Args:
        s (str): The string key to normalize.

    Returns:
        str: A simplified, lowercased, single-space cache key.

    Example:
        "   Hello   WORLD?   " -> "hello world?"

    """
    return " ".join(s.strip().lower().split())


def _cache_get(key: str) -> Optional[str]:
    """
    Thread-safe cache retrieval.
    Checks if key exists and is not expired. Otherwise, returns None.

    Args:
        key (str): Normalized cache key.

    Returns:
        Optional[str]: Cached string value or None if missing/expired.

    Example:
        _cache_get("hello") -> "Hi there" or None

    """
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
    """
    Thread-safe cache setter. Stores a value with expiration.
    If the cache exceeds its max size, removes the (approximately) oldest item.

    Args:
        key (str): Normalized cache key.
        value (str): Value to store.

    Returns: None

    Example:
        _cache_set("q:hello", "Hi!")

    """
    exp = time.time() + CACHE_TTL
    with _cache_lock:
        if len(_cache_store) >= CACHE_MAX:
            # Evict the oldest item (by earliest expiration)
            oldest = min(_cache_store.items(), key=lambda kv: kv[1][0])[0]
            _cache_store.pop(oldest, None)
        _cache_store[key] = (exp, value)

# =======================
# SCHEMA DEFINITIONS
# =======================

class ChatRequest(BaseModel):
    """
    Schema for chat endpoint requests.

    Args:
        question (str): The user query. Required.
        history (Optional[List[Dict[str, Any]]]): Chat history as a list of dicts.
        k (Optional[int]): Number of top documents to retrieve.
        cache_bypass (Optional[bool]): Forces bypass of cache if set to true.

    Example:
        {
            "question": "What is the schedule?",
            "history": [{"role":"user","content":"Hi"},{"role":"assistant","content":"Hello"}],
            "k": 5,
            "cache_bypass": false
        }
    """
    question: str = Field(..., min_length=1)
    history: Optional[List[Dict[str, Any]]] = None
    k: Optional[int] = None
    cache_bypass: Optional[bool] = False  # If true, force bypass of cache

class ChatResponse(BaseModel):
    """
    Schema for chat responses.

    Args:
        answer (str): The generated or cached answer.
    """
    answer: str

class IngestResponse(BaseModel):
    """
    Schema for ingestion responses.

    Args:
        added_chunks (int): Number of indexed document chunks.
    """
    added_chunks: int

# =======================
# ENDPOINT DEFINITIONS
# =======================

@app.get("/health")
def health() -> dict:
    """
    Simple health & status endpoint.
    Reports whether the server is up and information about the vector database/cache.

    Returns:
        dict: Service and status fields

    Example:
        curl http://localhost:8000/health
    """
    return {
        "status": "ok",
        "vectorstore": settings.chroma_path,
        "collection": settings.collection_name,
        "cache_ttl": CACHE_TTL,
        "cache_items": len(_cache_store),
    }


@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(require_api_key)])
def chat(req: ChatRequest) -> dict:
    """
    Chat endpoint.
    Answers a user's question using the retrieval-augmented generation (RAG) chain.
    Optionally uses cache for repeated questions.

    Args:
        req (ChatRequest): The input request containing question, history, k, and cache_bypass.

    Returns:
        dict: A dictionary with the generated answer.

    Example:
        POST /chat
        Body: {"question":"hello", "cache_bypass": false}
        Result: {"answer":"Hi there!"}
    """

    # Optionally override the number of top results (k) for retrieval.
    if req.k:
        rag.retriever.search_kwargs["k"] = req.k

    # Build a cache key using normalized question, k, and a hash of the history.
    hist_sig = ""
    if req.history:
        try:
            hist_sig = str(hash(json.dumps(req.history, ensure_ascii=False)))
        except Exception:
            hist_sig = str(hash(str(req.history)))
    key = f"q:{_norm(req.question)}|k:{rag.retriever.search_kwargs.get('k')}|h:{hist_sig}"

    # Try to answer from cache unless cache_bypass is true.
    if not req.cache_bypass:
        cached = _cache_get(key)
        if cached is not None:
            return {"answer": cached}

    # Generate the answer via the RAG chain.
    partial = ""
    for chunk in rag.stream_answer(req.question, history_messages=req.history or []):
        # The last chunk is the complete answer.
        partial = chunk or partial

    # Cache the new answer.
    if partial:
        _cache_set(key, partial)

    return {"answer": partial}


@app.post("/chat/stream", dependencies=[Depends(require_api_key)])
def chat_stream(req: ChatRequest):
    """
    Streaming chat endpoint.
    Streams the answer in real-time using server-sent events (SSE).
    If an answer is cached, returns it in a single SSE event.

    Args:
        req (ChatRequest): The input request with question, history, and options.

    Returns:
        StreamingResponse: FastAPI streaming response suitable for SSE (text/event-stream).

    Example:
        POST /chat/stream
        Body: {"question": "hello"}

    Yields:
        bytes: SSE events following the "data: ..." protocol.
    """
    # Optionally override k if provided.
    if req.k:
        rag.retriever.search_kwargs["k"] = req.k

    # Build cache key as in /chat endpoint.
    hist_sig = ""
    if req.history:
        try:
            hist_sig = str(hash(json.dumps(req.history, ensure_ascii=False)))
        except Exception:
            hist_sig = str(hash(str(req.history)))
    key = f"q:{_norm(req.question)}|k:{rag.retriever.search_kwargs.get('k')}|h:{hist_sig}"

    # Check cache unless bypassed.
    cached = None if req.cache_bypass else _cache_get(key)

    def gen() -> Generator[bytes, None, None]:
        """
        Streaming generator to yield chat responses or error messages as SSE events.

        Yields:
            bytes: JSON-encoded SSE lines.
        """
        try:
            if cached is not None:
                # Return cached answer as single SSE event.
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
async def ingest(
    file: Optional[UploadFile] = File(None),
    directory: Optional[str] = Form(None)
) -> dict:
    """
    Ingest endpoint for (re-)indexing document files.

    Accepts either:
      - A file upload (saved and indexed)
      - Or a directory name (relative to DATA_RAW_DIR)
    On ingestion, clears the answer cache as the knowledge base may have changed.

    Args:
        file (Optional[UploadFile]): The uploaded file. Saved under data_raw_dir.
        directory (Optional[str]): Directory under data_raw_dir to index.

    Returns:
        dict: {"added_chunks": N} where N is the number of indexed chunks.

    Example:
        POST /ingest
        Form Fields: file=<UploadFile>
        OR
        Form Fields: directory="mydata"
    """
    from pathlib import Path
    target_dir = Path(settings.data_raw_dir)

    if file:
        # Save uploaded file under data_raw_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = target_dir / file.filename
        with open(tmp_path, "wb") as f:
            f.write(await file.read())
        data_dir = str(target_dir)
    elif directory:
        data_dir = str(Path(directory))
    else:
        raise HTTPException(status_code=400, detail="Provide either a file or a directory")

    # Document loading and chunking
    raw_docs = load_documents(data_dir)
    chunks = split_documents(raw_docs, settings)
    added = build_or_update_index(chunks, settings)

    # Clear all cached answers (knowledge has changed)
    with _cache_lock:
        _cache_store.clear()

    return {"added_chunks": added}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))