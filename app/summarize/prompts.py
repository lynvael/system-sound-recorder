"""Russian prompts for the map-reduce summarization.

All prompts and the produced summary are in Russian: meetings are Russian and
the summary must be too. The MAP pass stays free-text (chunk notes). The REDUCE
and SINGLE passes no longer describe a Markdown layout: the shape is enforced by
the json_schema of schema.MeetingSummary (native structured output), so these
prompts only describe what belongs in each field.
"""

from __future__ import annotations

# System prompt shared by both passes: sets role and language.
SYSTEM = (
    "Ты — ассистент, который составляет деловые протоколы встреч на русском "
    "языке. Пиши кратко, по делу, только на русском. Не выдумывай факты: "
    "опирайся исключительно на предоставленный текст. Реплики размечены "
    "по говорящим (например «Я» и «Собеседники»)."
)

# --- map: summarize one chunk -----------------------------------------------
MAP_INSTRUCTIONS = (
    "Ниже — фрагмент стенограммы встречи (часть {index} из {total}). "
    "Кратко законспектируй ЭТОТ фрагмент на русском языке, выделив:\n"
    "- принятые решения;\n"
    "- поставленные задачи (с ответственными, если они названы);\n"
    "- обсуждённые темы и важные детали;\n"
    "- открытые вопросы.\n"
    "Не придумывай то, чего нет в тексте. Если какой-то категории нет — "
    "просто пропусти её. Ответ дай простым маркированным списком.\n\n"
    "Фрагмент стенограммы:\n{chunk}"
)

# --- reduce: consolidate chunk summaries into the final structured object ---
# The output shape is enforced by schema.MeetingSummary (json_schema), so this
# prompt only says what to put where — no Markdown headers to match.
REDUCE_INSTRUCTIONS = (
    "Ниже — последовательные конспекты фрагментов одной встречи. "
    "Объедини их в ОДИН связный итоговый протокол на русском языке, убрав "
    "повторы и противоречия, и заполни поля структуры:\n"
    "- tldr: 2–4 предложения о сути встречи;\n"
    "- key_decisions: принятые решения;\n"
    "- tasks: поставленные задачи; для каждой укажи assignee, если "
    "ответственный назван в тексте, иначе оставь assignee пустым (null);\n"
    "- topics: обсуждённые темы и важные детали;\n"
    "- open_questions: нерешённые вопросы.\n"
    "Пиши только на русском. Не придумывай фактов: опирайся исключительно на "
    "конспекты. Если для списка нет данных — оставь его пустым.\n\n"
    "Конспекты фрагментов:\n{summaries}"
)

# When there is a single chunk we skip the reduce round but still fill the same
# structure directly from the transcript.
SINGLE_INSTRUCTIONS = (
    "Ниже — стенограмма встречи. Составь итоговый протокол на русском языке и "
    "заполни поля структуры:\n"
    "- tldr: 2–4 предложения о сути встречи;\n"
    "- key_decisions: принятые решения;\n"
    "- tasks: поставленные задачи; для каждой укажи assignee, если "
    "ответственный назван в тексте, иначе оставь assignee пустым (null);\n"
    "- topics: обсуждённые темы и важные детали;\n"
    "- open_questions: нерешённые вопросы.\n"
    "Пиши только на русском. Не придумывай фактов: опирайся исключительно на "
    "текст. Если для списка нет данных — оставь его пустым.\n\n"
    "Стенограмма:\n{chunk}"
)
