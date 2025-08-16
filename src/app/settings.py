from __future__ import annotations

import os
from dataclasses import dataclass
from dotenv import load_dotenv


def _get_bool(val: str | None, default: bool) -> bool:
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class Settings:
    # OpenAI
    openai_api_key: str
    openai_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    openai_timeout: int = 60

    # Paths
    data_raw_dir: str = "./data/raw"
    data_processed_dir: str = "./data/processed"
    chroma_path: str = "./vectorstore"
    collection_name: str = "academy_docs"

    # Retrieval
    rag_search_type: str = "mmr"  # "similarity" | "mmr"
    rag_search_k: int = 4
    rag_mmr_lambda: float = 0.7

    # Splitting
    chunk_size: int = 400
    chunk_overlap: int = 80
    splitter_encoder: str = "tiktoken"  # "tiktoken" | "char"

    # UI
    gradio_share: bool = False
    gradio_username: str | None = None
    gradio_password: str | None = None

    # App
    default_language: str = "auto"  # "auto" | "ar" | "en"
    temperature: float = 0.3
    log_level: str = "INFO"


def get_settings() -> Settings:
    """
    Load environment variables into a strongly-typed Settings object.
    """
    load_dotenv()  # load from .env if present

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing. Set it in your .env file.")

    return Settings(
        openai_api_key=api_key,
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        openai_embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        openai_timeout=int(os.getenv("OPENAI_TIMEOUT", "60")),
        data_raw_dir=os.getenv("DATA_RAW_DIR", "./data/raw"),
        data_processed_dir=os.getenv("DATA_PROCESSED_DIR", "./data/processed"),
        chroma_path=os.getenv("CHROMA_PATH", "./vectorstore"),
        collection_name=os.getenv("COLLECTION_NAME", "academy_docs"),
        rag_search_type=os.getenv("RAG_SEARCH_TYPE", "mmr"),
        rag_search_k=int(os.getenv("RAG_SEARCH_K", "4")),
        rag_mmr_lambda=float(os.getenv("RAG_MMR_LAMBDA", "0.7")),
        chunk_size=int(os.getenv("CHUNK_SIZE", "400")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "80")),
        splitter_encoder=os.getenv("SPLITTER_ENCODER", "tiktoken"),
        gradio_share=_get_bool(os.getenv("GRADIO_SHARE"), False),
        gradio_username=os.getenv("GRADIO_USERNAME") or None,
        gradio_password=os.getenv("GRADIO_PASSWORD") or None,
        default_language=os.getenv("DEFAULT_LANGUAGE", "auto"),
        temperature=float(os.getenv("TEMPERATURE", "0.3")),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
    )
