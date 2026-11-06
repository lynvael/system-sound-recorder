# AGENTS.md

## Python / uv

- Все операции с Python — только через **uv**: зависимости (`uv sync`,
  `uv lock`), запуск кода и тестов (`uv run python ...`, `uv run pytest`).
- Не использовать `pip`, `python -m venv` и ручное управление `.venv/`.
