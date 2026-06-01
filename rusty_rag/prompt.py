from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rusty_rag.config import AppConfig


def build_context_prompt(question: str, chunks: list[dict]) -> str:
    context = "\n\n".join(
        f"[{i + 1}] ({c['source_path']})\n{c['text']}"
        for i, c in enumerate(chunks)
    )
    return f"Answer based on the following context:\n\n{context}\n\nQuestion: {question}"


def ask_llm(prompt: str, config: AppConfig) -> str:
    import openai
    response = openai.OpenAI().chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content
