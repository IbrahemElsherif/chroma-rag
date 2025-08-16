from __future__ import annotations

import gradio as gr
from .settings import get_settings, Settings
from .rag_chain import RAGChain


def launch() -> None:
    settings: Settings = get_settings()
    rag = RAGChain(settings)

    def _extract_user_text(message) -> str:
        if message is None:
            return ""
        if isinstance(message, str):
            return message
        if isinstance(message, dict):
            return str(message.get("content", ""))
        if isinstance(message, list) and message:
            last = message[-1]
            if isinstance(last, dict):
                return str(last.get("content", ""))
            return str(last)
        return str(message)

    # Gradio سيمرّر (message, history) في type="messages"
    def _stream_messages(message, history):
        user_text = _extract_user_text(message)
        try:
            # مرّر التاريخ كما هو ليتم تلخيصه داخل السلسلة
            for chunk in rag.stream_answer(user_text, history_messages=history):
                yield {"role": "assistant", "content": chunk}
        except Exception as e:
            yield {"role": "assistant", "content": f"حدث خطأ أثناء التوليد:\n{e}"}

    auth = None
    if settings.gradio_username and settings.gradio_password:
        auth = (settings.gradio_username, settings.gradio_password)

    demo = gr.ChatInterface(
        fn=_stream_messages,
        type="messages",  # الشكل الحديث
        title="RAG Chat (Chroma + OpenAI)",
        description="Ask questions based on your indexed documents.",
        cache_examples=False,
    )

    demo.launch(
        share=settings.gradio_share,
        auth=auth,
        inbrowser=False,
        server_name="0.0.0.0",
    )
