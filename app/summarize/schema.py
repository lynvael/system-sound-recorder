"""Structured schema for the final meeting protocol.

This is the contract that replaces the old "emit `## headers`, then scrape the
Markdown" flow: the LLM fills these fields via native structured output
(`response_format` = json_schema), and docx_export renders straight from the
validated object. The field set mirrors 1:1 the sections we have always
rendered; the Russian docx headings for each live in docx_export._SECTIONS.

Kept as a standalone module (not inside docx_export/pipeline) so both the
client helper — which needs the JSON schema — and the renderer can import it
without pulling in openai/docx. pydantic is already a project-wide dependency,
so importing it at module scope here is consistent with the rest of the app.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Task(BaseModel):
    """A single action item. `assignee` is null when nobody was named."""

    text: str = Field(description="Формулировка задачи.")
    # Nullable rather than optional-absent: strict json_schema requires every
    # property to be present, so the model must emit `assignee` (possibly null).
    # The default lets Python-side construction omit it too.
    assignee: str | None = Field(
        default=None,
        description="Ответственный, если он назван в тексте; иначе null.",
    )


class MeetingSummary(BaseModel):
    """The whole protocol. One field per rendered docx section."""

    tldr: str = Field(description="2–4 предложения о сути встречи.")
    key_decisions: list[str] = Field(description="Принятые решения.")
    tasks: list[Task] = Field(description="Поставленные задачи.")
    topics: list[str] = Field(description="Обсуждённые темы и важные детали.")
    open_questions: list[str] = Field(description="Нерешённые вопросы.")


def strict_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Return a json_schema for `model` that satisfies OpenAI strict mode.

    Strict structured output requires, for every object node (including nested
    `$defs` like Task), `additionalProperties: false` and *all* properties
    listed in `required`. pydantic's `model_json_schema()` does neither by
    default, so we walk the tree and enforce both. Idempotent and side-effect
    free on the caller's model — it operates on a freshly generated dict.
    """
    schema = model.model_json_schema()
    _strictify(schema)
    return schema


def _strictify(node: Any) -> None:
    """Recursively force additionalProperties=false + required=all on objects.

    Also strips every `default` key: OpenAI strict mode rejects schemas that
    carry `default` (real endpoint 400s with "'default' is not permitted"), and
    pydantic emits `"default": null` for `assignee` because of its `= None`.
    Since strict mode lists every property in `required`, defaults are
    meaningless here anyway. `title` is left in place — OpenAI tolerates it.
    """
    if isinstance(node, dict):
        node.pop("default", None)  # strict mode rejects `default`
        if isinstance(node.get("properties"), dict):
            node["additionalProperties"] = False
            node["required"] = list(node["properties"].keys())
        for value in node.values():
            _strictify(value)
    elif isinstance(node, list):
        for item in node:
            _strictify(item)
