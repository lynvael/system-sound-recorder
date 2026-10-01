"""Shared Russian prompts for summarization.

All prompts and the produced report are in Russian: meetings are Russian and
the report must be too. The final step ALWAYS produces Markdown (plain
`chat()`, no structured output); the Markdown→docx renderer
(`docx_export.write_markdown_docx`) understands a fixed subset, so the output
rules below stay in this FIXED part — the user-editable prompt never
constrains syntax.

Strategy-owned prompts (map instructions, per-method framings) live in the
strategy modules, not here.
"""

from __future__ import annotations

# System prompt shared by every LLM call: sets role and language.
SYSTEM = (
    "Ты — ассистент, который составляет деловые протоколы встреч на русском "
    "языке. Пиши кратко, по делу, только на русском. Не выдумывай факты: "
    "опирайся исключительно на предоставленный текст. Реплики размечены "
    "по говорящим (например «Я» и «Собеседники»)."
)

# Fixed Markdown output rules — appended to SYSTEM for the final call. The
# subset must stay in sync with docx_export.write_markdown_docx: anything
# outside it degrades to plain text and must never crash the renderer.
MARKDOWN_OUTPUT_RULES = (
    "Формат ответа: Markdown. Разрешённая разметка: заголовки `#`, `##`, "
    "`###`; маркированные списки `-`; нумерованные списки `1.`; жирный "
    "`**текст**`; обычные абзацы. Не используй таблицы, блоки кода и другую "
    "разметку. Отвечай только на русском языке. Если требования к формату "
    "отчёта противоречат этим правилам, по содержанию приоритет у требований, "
    "но ответ должен оставаться в перечисленной разметке. Не выдумывай факты."
)

# Default report prompt: reproduces the classic protocol sections as
# Markdown. Used when the user does not supply a custom prompt.
DEFAULT_REPORT_PROMPT = (
    "Составь итоговый протокол встречи на русском языке в формате Markdown "
    "со следующими разделами (заголовки `##`), в этом порядке:\n"
    "## TL;DR — 2–4 предложения о сути встречи;\n"
    "## Ключевые решения — принятые решения (маркированный список);\n"
    "## Задачи — поставленные задачи (маркированный список); для каждой "
    "укажи ответственного, если он назван в тексте (например «Подготовить "
    "отчёт — Иван»);\n"
    "## Обсуждённые темы — обсуждённые темы и важные детали;\n"
    "## Открытые вопросы — нерешённые вопросы.\n"
    "Если по разделу нет данных — напиши в нём «—».\n"
    "Не придумывай фактов: опирайся исключительно на предоставленный текст."
)
