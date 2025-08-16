from __future__ import annotations

from typing import Generator, List, Tuple, Sequence, Mapping, Any

from .guardrails import load_catalog, looks_like_diploma_query, answer_diplomas

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

from .settings import Settings


SYSTEM_PROMPT = """
# هويتك وقدراتك
- أنت مساعد طلاب ومتدربين المعهد السعودي العالي المتخصص للتدريب
- مهمتك الرئيسية هي تقديم معلومات دقيقة عن برامج المعهد ودوراته ودبلوماته
- عليك توليد الرد بنفس لغة استفسار المستخدم

The question: {question}

Conversation history:
{history}

The knowledge:
{knowledge}
"""

def _build_prompt() -> ChatPromptTemplate:
    # سيملأ placeholders الثلاثة من format_messages
    return ChatPromptTemplate.from_template(SYSTEM_PROMPT)


def _init_llm(settings: Settings) -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.openai_model,
        temperature=settings.temperature,
        timeout=settings.openai_timeout,
        api_key=settings.openai_api_key,
    )


def _init_embeddings(settings: Settings) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=settings.openai_embedding_model,
        api_key=settings.openai_api_key,
    )


def _init_vectorstore(settings: Settings) -> Chroma:
    embeddings = _init_embeddings(settings)
    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=embeddings,
        persist_directory=settings.chroma_path,  # أو client=... لو استخدمت PersistentClient
    )


def _build_context_from_docs(docs) -> str:
    """سياق نصّي فقط بدون إظهار المصادر للمستخدم."""
    blocks = []
    for d in docs:
        blocks.append(d.page_content)
    return "\n\n---\n\n".join(blocks)


def _format_history(messages: Sequence[Mapping[str, Any]] | None, max_turns: int = 6) -> str:
    """
    Gradio (type='messages') يمرّر history كـ list[{'role','content'}].
    نُحوّله لنص مختصر: 'User: ...\\nAssistant: ...'
    """
    if not messages:
        return ""
    # خذ آخر max_turns فقط
    msgs = messages[-max_turns:]
    lines: List[str] = []
    role_map = {"user": "User", "assistant": "Assistant", "system": "System"}
    for m in msgs:
        role = role_map.get(str(m.get("role", "")).lower(), str(m.get("role", "User")))
        content = str(m.get("content", "")).strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


class RAGChain:
    def __init__(self, settings: Settings):
        self.settings = settings # .env settings
        self.prompt = _build_prompt() # prepare prompt
        self.llm = _init_llm(settings) # prepare LLM
        self.vectorstore = _init_vectorstore(settings) # Open the vectorstore
        self.catalog = load_catalog()
        self.retriever = self.vectorstore.as_retriever(
            search_type=settings.rag_search_type,
            search_kwargs={
                "k": settings.rag_search_k,
                **({"lambda_mult": settings.rag_mmr_lambda} if settings.rag_search_type == "mmr" else {}),
            },
        )

    def stream_answer(
        self,
        question: str,
        history_messages: Sequence[Mapping[str, Any]] | None = None,
    ) -> Generator[str, None, None]:
        """يرجع نص متدفّق بدون إرفاق المصادر في النهاية."""
        # أسئلة “الدبلومات” نجاوبها من الكتالوج فقط (مصدر مُعتَمد)، لتجنّب ضجيج/OCR
        if looks_like_diploma_query(question):
            yield answer_diplomas(self.catalog)
            return
        docs = self.retriever.invoke(question)
        if not docs:
            yield "لا توجد معرفة مفهرسة بعد. رجاءً شغّل أمر الفهرسة ثم أعد المحاولة."
            return

        knowledge = _build_context_from_docs(docs)
        history_str = _format_history(history_messages)

        messages = self.prompt.format_messages(
            question=question,
            history=history_str,
            knowledge=knowledge,
        )

        partial = ""
        for chunk in self.llm.stream(messages):
            partial += chunk.content or ""
            yield partial

        # لا نضيف المصادر في النهاية بناءً على طلبك
        yield partial
