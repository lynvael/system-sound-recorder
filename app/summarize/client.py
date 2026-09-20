"""OpenAI-compatible client factory and a thin chat-completion helper.

The `openai` import is local to this module (kept out of package import) so that
importing `app.summarize` stays cheap and Qt-free; the dependency is only pulled
when a summarization actually runs.
"""

from __future__ import annotations

import json
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from app.config import LLMSettings
from app.log import get_logger
from app.summarize.schema import strict_json_schema

TModel = TypeVar("TModel", bound=BaseModel)

logger = get_logger("summarize.client")


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
    """Single chat completion; returns the assistant text.

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
    )
    if not resp.choices:
        raise RuntimeError("LLM вернул пустой ответ (нет choices).")
    content = resp.choices[0].message.content
    if not content or not content.strip():
        raise RuntimeError("LLM вернул пустой текст.")
    return content.strip()


def chat_structured(
    client,
    llm: LLMSettings,
    system: str,
    user: str,
    schema_model: type[TModel],
) -> TModel:
    """Chat completion that returns a validated `schema_model` instance.

    Uses the OpenAI-compatible native structured output: the server is asked to
    emit JSON conforming to `schema_model`'s (strict-friendly) JSON schema, so we
    parse and validate the object instead of scraping Markdown from free text.

    Raises RuntimeError (mirroring `chat()`'s empty-response handling) when the
    response is missing, when the server rejects `response_format`, or when the
    payload is not valid JSON / does not satisfy the schema. The pipeline wraps
    these into a Russian SummarizationError.
    """
    schema = strict_json_schema(schema_model)
    try:
        resp = client.chat.completions.create(
            model=llm.model,
            temperature=llm.temperature,
            max_tokens=llm.max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": schema_model.__name__,
                    "schema": schema,
                    "strict": True,
                },
            },
        )
    except Exception as exc:  # noqa: BLE001 - map to a clear Russian error
        # Only an actual bad request (HTTP 400) means the server rejected our
        # response_format=json_schema; treat everything else (timeout, network,
        # auth, 5xx) as a neutral request failure so the message isn't
        # misleading. openai is imported lazily to keep module import cheap.
        from openai import BadRequestError

        if isinstance(exc, BadRequestError):
            raise RuntimeError(
                "Сервер LLM отклонил запрос структурированного вывода "
                f"(response_format=json_schema): {exc}"
            ) from exc
        raise RuntimeError(
            f"Ошибка запроса структурированного вывода к LLM: {exc}"
        ) from exc

    if not resp.choices:
        raise RuntimeError("LLM вернул пустой ответ (нет choices).")
    content = resp.choices[0].message.content
    if not content or not content.strip():
        raise RuntimeError("LLM вернул пустой структурированный ответ.")
    try:
        print(f"result: ${content}")
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"LLM вернул не-JSON в структурированном ответе: {exc}"
        ) from exc
    try:
        return schema_model.model_validate(data)
    except ValidationError as exc:
        raise RuntimeError(
            f"Структурированный ответ LLM не прошёл валидацию схемы: {exc}"
        ) from exc
