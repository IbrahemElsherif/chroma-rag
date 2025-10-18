from __future__ import annotations
import os 
import gradio as gr
from .settings import get_settings, Settings
from .rag_chain import RAGChain


def launch() -> None:
    """
    Launches the Gradio chat interface for the RAG-based assistant.

    This function initializes settings, creates the retrieval-augmented generation chain,
    and defines the Gradio chat UI. Handles user authentication if configured.

    Returns:
        None

    Example:
        >>> launch()  # Starts the chat UI
    """
    settings: Settings = get_settings()
    rag = RAGChain(settings)

    def _extract_user_text(message) -> str:
        """
        Extracts the user's text from a Gradio message object. Handles multiple input formats,
        including string, dict (expects "content" key), or list (uses last message).

        Args:
            message (Any): The message object as received from Gradio's ChatInterface.

        Returns:
            str: The extracted user message as a string.

        Example:
            >>> _extract_user_text("Hello")
            'Hello'
            >>> _extract_user_text({"content": "Hi"})
            'Hi'
            >>> _extract_user_text([{"role": "user", "content": "What is this?"}])
            'What is this?'
        """
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

    def _stream_messages(message, history):
        """
        Handles a chat round for the Gradio ChatInterface. Receives the user's message and history,
        streams assistant replies back to the UI iterator-style.

        Args:
            message (Any): The latest user message (varied structure).
            history (Any): The accumulated chat history as passed by Gradio.

        Yields:
            dict: A response in the form {"role": "assistant", "content": chunk}, streamed.

        Example:
            for msg in _stream_messages("Hello", []):
                print(msg)
        """
        user_text = _extract_user_text(message)
        try:
            # Passes full chat history to the RAG chain, which summarizes if needed.
            for chunk in rag.stream_answer(user_text, history_messages=history):
                yield {"role": "assistant", "content": chunk}
        except Exception as e:
            # In case of any errors, return a friendly error message to the user.
            yield {"role": "assistant", "content": f"An error occurred during generation:\n{e}"}

    # Authentication setup for Gradio: enables if both username & password are set in config.
    auth = None
    if settings.gradio_username and settings.gradio_password:
        auth = (settings.gradio_username, settings.gradio_password)

    # Define the Gradio ChatInterface with configuration and descriptive UI parameters.
    demo = gr.ChatInterface(
        fn=_stream_messages,
        type="messages",  # Modern format where Gradio passes messages as list[dict]
        title="مساعد طلاب المعهد السعودي العالي المتخصص للتدريب",
        description="Ask questions based on your indexed documents.",
        cache_examples=False,
    )
    demo.launch(
        share=False,  # Disable share for production
        auth=auth,
        inbrowser=False,
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", 7860)),  # Use Render's PORT
        show_error=True
    )
    # # Launch the Gradio app with optional share, authentication, and network parameters.
    # demo.launch(
    #     share=settings.gradio_share,
    #     auth=auth,
    #     inbrowser=False,           # Set to True to auto-open browser tab
    #     server_name="0.0.0.0",     # Listen on all interfaces
    # )
