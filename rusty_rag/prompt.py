from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rusty_rag.config import AppConfig

_SYSTEM_PROMPT = (
    "You are a question-answering assistant. "
    "Answer ONLY using the information provided in the context. "
    "If the context does not contain enough information to answer the question, "
    "respond with: 'I don't have information about this in the knowledge base.' "
    "Do NOT use outside knowledge or training data."
)


def build_context_prompt(question: str, chunks: list[dict]) -> str:
    context = "\n\n".join(
        f"[{i + 1}] ({c['source_path']})\n{c['text']}"
        for i, c in enumerate(chunks)
    )
    return f"Context:\n\n{context}\n\nQuestion: {question}"


def ask_llm(prompt: str, config: AppConfig) -> str:
    import openai
    response = openai.OpenAI().chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content
