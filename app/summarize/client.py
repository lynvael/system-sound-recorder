"""OpenAI-compatible client factory and a thin chat-completion helper.

The `openai` import is local to this module (kept out of package import) so
that importing `app.summarize` stays cheap and Qt-free; the dependency is only
pulled when a summarization actually runs.

Every `chat.completions.create` MUST pass the reasoning kwargs below — the
configured proxy (FrankAI) is unusably slow without `reasoning_effort`
(MEMORY.md). Keep them in this single place so no call can forget them.
"""

from __future__ import annotations

from app.config import LLMSettings
from app.log import get_logger

logger = get_logger("summarize.client")

# Single source of truth for the reasoning kwargs (see module docstring).
_REASONING_KWARGS = {
    "reasoning_effort": "medium",
    "extra_body": {"allowed_openai_params": ["reasoning_effort"]},
}


def build_client(llm: LLMSettings):
    """Construct an OpenAI client pointed at the configured endpoint.

    `llm.url` is used verbatim as `base_url`, so it must already include any
    `/v1` suffix the server expects.
    """
    from openai import OpenAI

    return OpenAI(
        base_url=llm.url,
        api_key=llm.api_key,
        timeout=llm.request_timeout,
    )


def chat(client, llm: LLMSettings, system: str, user: str) -> str:
    """Single chat completion; returns the assistant text (Markdown).

    Raises on transport/API errors and on an empty response — the caller must
    surface failures, not swallow them.
    """
    resp = client.chat.completions.create(
        model=llm.model,
        temperature=llm.temperature,
        max_tokens=llm.max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        **_REASONING_KWARGS,
    )
    if not resp.choices:
        raise RuntimeError("LLM вернул пустой ответ (нет choices).")
    content = resp.choices[0].message.content
    if not content or not content.strip():
        raise RuntimeError("LLM вернул пустой текст.")
    return content.strip()
