from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from typing import List

from src.app.settings import Settings


def make_splitter(settings: Settings) -> RecursiveCharacterTextSplitter:
    """
    Create a text splitter.
    If SPLITTER_ENCODER == 'tiktoken', chunk sizes are token-based.
    """
    if settings.splitter_encoder.lower() == "tiktoken":
        return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
    # Fallback: char-based
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )


def split_documents(docs: List[Document], settings: Settings) -> List[Document]:
    splitter = make_splitter(settings)
    return splitter.split_documents(docs)
