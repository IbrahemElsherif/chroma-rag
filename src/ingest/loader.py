from __future__ import annotations

from pathlib import Path
from typing import List

from langchain_community.document_loaders import (
    PyPDFDirectoryLoader,
    DirectoryLoader,
    TextLoader,
)
from langchain_core.documents import Document


def load_documents(data_dir: str) -> List[Document]:
    """
    Load PDF, TXT, and MD files from a directory tree.

    - PDFs are parsed page-by-page with metadata (source/page).
    - TXT/MD are loaded as whole files.
    """
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"Data directory not found: {root}")

    documents: List[Document] = []

    # PDFs
    pdf_loader = PyPDFDirectoryLoader(str(root))
    documents.extend(pdf_loader.load())

    # TXT
    txt_loader = DirectoryLoader(str(root), glob="**/*.txt", loader_cls=TextLoader, show_progress=True)
    documents.extend(txt_loader.load())

    # MD
    md_loader = DirectoryLoader(str(root), glob="**/*.md", loader_cls=TextLoader, show_progress=True)
    documents.extend(md_loader.load())

    return documents
