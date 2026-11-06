# Эксперименты по суммаризации длинных транскриптов

Сравнение трёх стратегий суммаризации длинных текстов (транскрипты встреч из
`recordings/`). Каждая стратегия — self-contained папка со своим скриптом,
README и unit-тестами; кросс-импортов между папками нет.

## Папки

| Папка | Стратегия | Источники | Нужен embeddings-эндпоинт |
|---|---|---|---|
| [`map_reduce/`](map_reduce/) | Классический Map-Reduce: чанки → параллельные LLM-саммари → рекурсивный merge | AWS ML Blog, LangChain `map_reduce` | Нет |
| [`eacss/`](eacss/) | EACSS (Extractive-Abstractive): extractive-фаза (эмбеддинги предложений + k-means → отбор у центроидов) → abstractive-фаза (LLM) | AWS ML Blog, дек. 2023 | **Да** |
| [`hierarchical_context/`](hierarchical_context/) | Context-Aware Hierarchical Merging, вариант **Extract-Support**: иерархический merge, где каждому merge-шагу передаются извлечённые из исходника passages как фактологическая опора | Ou & Lapata, Findings of ACL 2025 | **Да** |

## Запуск

```bash
uv run python summary_tests/map_reduce/run.py recordings/20260924_161938/transcript.json
uv run python summary_tests/eacss/run.py recordings/vtb_broken/transcript.txt
uv run python summary_tests/hierarchical_context/run.py recordings/vtb_broken/transcript.txt
```

Вход: `.txt` (обычный текст) или `.json` (список сегментов
`{"start", "end", "speaker", "text"}` — формат `recordings/*/transcript.json`).
Результат: саммари в stdout + файл `output/<имя-входа>.md` в папке стратегии.

Общие параметры: `--chunk-size` (по умолчанию 32000 символов ≈ 8K токенов,
как в статье ACL 2025), `--overlap` (500).

## Переменные окружения (`.env`)

| Переменная | Для чего |
|---|---|
| `LLM_URL`, `LLM_API_KEY`, `LLM_MODEL`, `LLM_REQUEST_TIMEOUT` | Основная LLM (OpenAI-совместимый эндпоинт) — все три стратегии |
| `EMBED_URL`, `EMBED_API_KEY`, `EMBED_MODEL` | OpenAI-совместимый `/v1/embeddings` — только `eacss/` и `hierarchical_context/` (extractive-фаза) |

**Важно про LLM:** все вызовы идут с `reasoning_effort="medium"` и
`extra_body={"allowed_openai_params": ["reasoning_effort"]}` — без этого
модель отвечает неприемлемо долго (~60–90+ с на вызов против ~25–35 с).

## Что НЕ используется из оригинальных статей (ограничения)

- **Нет специализированных моделей.** В оригиналах:
  - ACL 2025 (Extract-Support) использует **MemSum** — RL-обученный
    extractive-суммаризатор, дообученный на домене. Здесь — generic
    embedding-модель + k-means (тот же подход, что в AWS EACSS).
  - AWS EACSS использует `bert-extractive-summarizer` (локальный
    multilingual BERT). Здесь — remote embeddings-эндпоинт.
- **Нет rerank-модели** (варианты IC-модуля Retrieve/Cite не реализованы —
  только Extract).
- **Нет динамического выбора контекста** (future work авторов ACL 2025 —
  context augmentation как tool агента).
- `eacss/`: abstractive-фаза при переполнении — один уровень map-reduce,
  map-шаг последовательный.

## Критерии выбора стратегии

- **Полнота охвата** (LLM читает весь текст): Map-Reduce — самый надёжный.
- **Экономия LLM-токенов**: EACSS — извлечение дешёвым embeddings-эндпоинтом
  сжимает текст до обращения к LLM.
- **Фактологичность длинных текстов**: Hierarchical Extract-Support —
  контекст из исходника подавляет накопление галлюцинаций при рекурсивном
  merge (в статье: +10 п. к доле верных claims против базлайнов).
- **Точечный вопрос, а не саммари**: ни одна из трёх — это сценарий
  Map-ReRank/RAG (не реализован).

## Тесты

```bash
uv run pytest summary_tests/
```
