from __future__ import annotations

import os
import pytest

from src.app.settings import get_settings
from src.app.rag_chain import RAGChain


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
def test_chain_initialization():
    settings = get_settings()
    chain = RAGChain(settings)
    # retriever should be available even if empty
    assert chain.retriever is not None
