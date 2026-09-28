# EACSS — Extractive-Abstractive Content Summarization Strategy

Экспериментальная реализация стратегии суммаризации длинных документов из
статьи AWS Machine Learning Blog
[«Simplify summarization of long documents with LLMs» (дек. 2023)](https://aws.amazon.com/blogs/machine-learning/simplify-summarization-of-long-documents-with-llms/):
сначала **extractive**-фаза (выбор ключевых предложений по кластеризации
эмбеддингов), затем **abstractive**-фаза (LLM-саммари по извлечённому
контенту). Папка self-contained: нет кросс-импортов с `app/` и другими
папками `summary_tests/`.

## Схема пайплайна

```
transcript (.txt / .json)
        │
        ▼
разбиение на предложения ──► чанки по 32 000 символов (по границам
        │                    предложений, overlap 500)
        ▼
┌── Extractive-фаза (на каждый чанк, параллельно) ──────────────────┐
│  эмбеддинги предложений (батчи по 64, concurrency 4, timeout 60с) │
│  k-means: k = min(10, max(3, n_sentences // 10)), numpy, seed=42  │
│  из каждого кластера — предложение, ближайшее к центроиду         │
│  (косинусная близость); дедупликация, исходный порядок           │
└────────────────────────────────────────────────────────────────────┘
        │  extractive-саммари каждого чанка
        ▼
скачивание (concat)
        │
        ▼
┌── Abstractive-фаза (последовательно) ─────────────────────────────┐
│  если объём ≤ 32 000 символов — один LLM-вызов (stuff)            │
│  иначе — один уровень map-reduce: батчи ≤ 32 000 → саммари        │
│  каждого → финальный merge-вызов                                   │
└────────────────────────────────────────────────────────────────────┘
        │
        ▼
stdout + summary_tests/eacss/output/<имя-входа>.md
```

Промпт abstractive-фазы — структурный, с антигаллюцинационным паттерном
**«cite-then-summarize»**: LLM получает только извлечённые k-means'ом
предложения (не весь текст) и обязана оперировать исключительно ими,
явно отмечая отсутствующее/неоднозначное, без внешних знаний.

- **Источник:** executive-summary prompt с [bestprompts.sh](https://www.bestprompts.sh/),
  адаптированный под саммари извлечённых предложений.
- **Stuff / Reduce** (объём ≤ chunk_size, либо финальный merge map-reduce):
  формат TL;DR → Ключевые моменты (точные цифры/даты/имена, без округления)
  → Решения и действия (или «Не заявлено») → Чего не покрыто.
  В reduce-варианте ввод — «саммари ключевых частей документа».
- **Map** (map-reduce, когда извлечённый контент > chunk_size): простой
  промпт «Составь краткое саммари… Сохрани все ключевые факты: имена,
  цифры, даты, сроки, решения».

## Как запустить

Все команды через **uv** (см. AGENTS.md):

```bash
# Полная прогонка (нужны LLM_* и EMBED_* в .env)
uv run python summary_tests/eacss/run.py recordings/20260924_161938/transcript.json

# Свои размеры чанка/overlap
uv run python summary_tests/eacss/run.py recordings/vtb_broken/transcript.txt --chunk-size 32000 --overlap 500

# Dry-run: без сети (синтетические эмбеддинги, abstractive-фаза — заглушка)
uv run python summary_tests/eacss/run.py some.txt --dry-run

# Только abstractive-фаза на уже подготовленном извлечённом контенте
uv run python summary_tests/eacss/run.py some.txt --extracted extracted.txt

# Unit-тесты (чанки, k-means)
uv run pytest summary_tests/eacss/test_eacss.py -v
```

## Требуемые env-переменные

| Переменная | Обязательна | Описание |
|---|---|---|
| `LLM_URL` | да (кроме `--dry-run`) | base_url OpenAI-совместимого LLM (в `.env`: `https://ai.frankrg.com/litellm/v1`) |
| `LLM_API_KEY` | да | ключ LLM |
| `LLM_MODEL` | да (кроме `--dry-run`) | имя модели |
| `LLM_REQUEST_TIMEOUT` | нет (default 600) | таймаут LLM-вызовов, сек |
| `EMBED_URL` | да (кроме `--dry-run`/`--extracted`) | base_url embeddings-эндпоинта, включая `/v1` |
| `EMBED_API_KEY` | нет (default `not-needed`) | ключ embeddings |
| `EMBED_MODEL` | да (кроме `--dry-run`/`--extracted`) | имя embedding-модели |

Таймаут embeddings фиксирован: 60 с.

**Требование к LLM-эндпоинту:** каждый `chat.completions.create` отправляет
`reasoning_effort="medium"` через
`extra_body={"allowed_openai_params": ["reasoning_effort"]}` — провайдер
(LiteLLM-прокси) должен пропускать этот параметр модели. Без ограничения
reasoning модель отвечает неприемлемо долго.

## Число вызовов для N чанков

Пусть в чанках суммарно `S` предложений, `k_i` — число кластеров i-го чанка
(≤ 10), извлечённый контент — `E` символов.

- **Embeddings-вызовы:** `⌈S / 64⌉` (батчи по 64 предложения; батчи чанков
  исполняются параллельно с concurrency 4).
- **LLM-вызовы:**
  - `E ≤ chunk_size` → **1** (stuff);
  - `E > chunk_size` → **⌈E / chunk_size⌉ + 1** (map + финальный merge).
- **Выбранных предложений:** `Σ k_i ≤ 10·N` (по одному на кластер).

## Ограничения

- Используется **generic embedding-модель** вместо fine-tuned
  extractive-суммаризатора из оригинальной статьи AWS: кластеризация по
  общим семантическим эмбеддингам хуже находит «представительные»
  предложения, чем специализированная модель.
- **Риск потери «атипичных» фрагментов:** k-means выбирает по одному
  предложению на кластер, поэтому редкие, но важные детали (один раз
  упомянутый факт, исключение из общего тона) могут не попасть ни в один
  крупный кластер и быть отброшенными.
- **Abstractive-фаза не параллелится:** map-вызовы и merge идут
  последовательно (осознанное ограничение, см. схему).
- Overlap 500 символов по границам предложений: реальный overlap может быть
  меньше 500 (если предложение начинается позже), а при очень длинном
  предложении overlap может оказаться нулевым.
