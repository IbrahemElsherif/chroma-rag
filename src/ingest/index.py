from __future__ import annotations

from hashlib import md5
from typing import List

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# استخدم عميل Chroma مباشر للتحكم في التليمترى/المسار
import chromadb
from chromadb.config import Settings as ChromaSettings

from src.app.settings import Settings


def _stable_id(doc: Document) -> str:
    src = str(doc.metadata.get("source", ""))
    page = str(doc.metadata.get("page", ""))
    payload = (src + "|" + page + "|" + doc.page_content).encode("utf-8")
    return md5(payload).hexdigest()


def build_or_update_index(chunks: List[Document], settings: Settings) -> int:
    """
    Add chunks to Chroma collection with content-hash IDs to avoid duplicates.
    With langchain-chroma, persistence is handled by the underlying client when using a persistent path.
    """
    # عميل دائم مع تعطيل التليمترى (يوقف رسائل Failed to send telemetry ...)
    client = chromadb.PersistentClient(
        path=settings.chroma_path,
        settings=ChromaSettings(anonymized_telemetry=False)
    )

    embeddings = OpenAIEmbeddings(
        model=settings.openai_embedding_model,
        api_key=settings.openai_api_key,
    )

    vector_store = Chroma(
        collection_name=settings.collection_name,
        embedding_function=embeddings,
        client=client,  # بدل persist_directory
    )

    # Normalize metadata
    for d in chunks:
        d.metadata["doc_id"] = d.metadata.get("source", "")
        d.metadata["page_num"] = d.metadata.get("page")

    ids = [_stable_id(d) for d in chunks]
    vector_store.add_documents(chunks, ids=ids)

    # لا حاجة لـ persist()؛ لكن لو موجودة في إصدارك مش غلط ننده عليها
    if hasattr(vector_store, "persist"):
        try:
            vector_store.persist()  # بعض الإصدارات القديمة تدعمها
        except Exception:
            pass

    return len(chunks)
