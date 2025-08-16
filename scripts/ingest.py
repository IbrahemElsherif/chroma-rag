from __future__ import annotations

from src.app.settings import get_settings
from src.ingest.loader import load_documents
from src.ingest.splitter import split_documents
from src.ingest.index import build_or_update_index


def main() -> None:
    settings = get_settings()
    print(f"[ingest] Loading from: {settings.data_raw_dir}")

    raw_docs = load_documents(settings.data_raw_dir)
    print(f"[ingest] Loaded {len(raw_docs)} documents/pages.")

    chunks = split_documents(raw_docs, settings)
    print(f"[ingest] Split into {len(chunks)} chunks (size={settings.chunk_size}, overlap={settings.chunk_overlap}).")

    added = build_or_update_index(chunks, settings)
    print(f"[ingest] Added {added} chunks to collection '{settings.collection_name}' at {settings.chroma_path}")


if __name__ == "__main__":
    main()
