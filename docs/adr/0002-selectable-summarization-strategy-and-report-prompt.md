# ADR-002: Выбор стратегии саммаризации и пользовательский промпт итогового отчёта

## Status

Accepted (2026-09-30). Реализовано (2026-09-30); отклонения от предложения,
утверждённые человеком, перечислены в разделе «Изменения по итогам
обсуждения / реализации» ниже. Открытые вопросы в конце документа.

- Дата: 2026-09-30
- Область: `app/summarize/`, `app/config.py`, `app/gui/{main_window,worker,prefs}.py`,
  новый `app/gui/prompt_dialog.py`
- Не затрагивается: аудио, VAD, STT, `Session`, `summary_tests/`

---

## Изменения по итогам обсуждения / реализации

Реализация выполнена со следующими отклонениями от исходного предложения
(каждое заменяет соответствующее место в теле ADR; в теле такие места
помечены «заменено, см. выше»):

1. **Структурированный вывод удалён.** `chat_structured()` /
   `json_schema` / `MeetingSummary` не используются: и промпт по умолчанию,
   и свой промпт дают **Markdown**, который встроенный рендерер
   `write_markdown_docx` превращает в .docx (заголовки `#`–`###`, списки,
   `**жирный**`; таблицы/код деградируют до обычного текста). Фактически
   реализован вариант (a) из «Ключевого напряжения», а не гибрид (c).
2. **Имя файла — на каждый запуск.** Не фиксированный `summary.docx`, а
   `summary_<метод>_<YYYYMMDD_HHMMSS>.docx` (новый файл на каждый запуск;
   D8 заменено). «Открыть отчёт» открывает самый свежий отчёт по mtime
   (legacy-`summary.docx` тоже учитывается).
3. **Параллелизм.** Fan-out-фазы (map, экстрактивные) идут через пул
   daemon-потоков с лимитом `LLM_CONCURRENCY` (по умолчанию 3) —
   `app/summarize/parallel.py` (D3/Q6 закрыты; путь по умолчанию тоже
   изменился: map теперь параллельный). Реальный максимум задаёт
   LiteLLM-прокси.
4. **Отмена добавлена** (A6 заменено). GUI создаёт `threading.Event` в
   момент клика и передаёт его через сигнал
   `request_summarize(dir, options, cancel_event)`; отдельного cancel-слота
   у воркера нет. При отмене: новые LLM/embedding-вызовы не отправляются,
   отчёт не записывается, статус «Саммаризация отменена.». Уже ушедшие
   HTTP-запросы **не прерываются** — они завершаются в фоновых
   daemon-потоках, их результаты отбрасываются. Закрытие окна во время
   саммаризации отменяет запуск.
5. **`LLM_REQUEST_TIMEOUT` по умолчанию 300 с** (Q5: поднят с 120 ради
   32k-промптов).
6. **`CUSTOM_FORMAT_TEMPLATE` упразднён.** Если сохранённый текст пуст,
   диалог «Промпт отчёта…» предзаполняется `DEFAULT_REPORT_PROMPT`.
7. **Сигнатура `request_summarize`** — `(session_dir, options, cancel_event)`;
   `run_summarization(session_dir, config, on_status, options, cancel_event)`.

---

## Context

### Что просит пользователь

1. **Выбор стратегии саммаризации в GUI.** В `summary_tests/` есть три
   эксперимента: `map_reduce/`, `eacss/` (extractive: эмбеддинги + k-means,
   затем abstractive LLM), `hierarchical_context/` (Context-Aware Hierarchical
   Merging, Extract-Support). **По умолчанию** остаётся map-reduce, который уже
   есть в приложении (`app/summarize/pipeline.py`).
2. **Пользовательский промпт финальной саммаризации.** Через него пользователь
   задаёт **формат** итогового отчёта. Нужен чекбокс «Использовать промпт по
   умолчанию». Его включение **не удаляет** текст пользователя: текст
   сохраняется, но не используется.
3. **Разумная архитектура без over-engineering.** Текущий код не в лучшей
   форме, это надо учесть.

### Текущее состояние (на 2026-09-30, ветка `feature/sounddevice-capture`)

- `app/summarize/pipeline.py`: `run_summarization(session_dir, config, on_status)` —
  единственная публичная точка входа. В одной функции смешаны:
  I/O (загрузка `transcript.json`), подготовка текста (`merge_consecutive` →
  `build_transcript_text` → `chunk_text`), **логика метода** (map по чанкам через
  `chat()`), **финальный шаг** (`chat_structured()` → `MeetingSummary`) и экспорт
  (`write_docx`). Любая ошибка оборачивается в `SummarizationError` (класс определён
  здесь же).
- `app/summarize/prompts.py`: `SYSTEM`, `MAP_INSTRUCTIONS`, `REDUCE_INSTRUCTIONS`,
  `SINGLE_INSTRUCTIONS`. В REDUCE и SINGLE склеены три разные вещи: описание
  входа («ниже — конспекты фрагментов» или «ниже — стенограмма»), **формат**
  (список полей `tldr/key_decisions/tasks/topics/open_questions`) и сам материал.
- `app/summarize/client.py`: `chat()` передаёт `reasoning_effort="medium"` и
  `extra_body={"allowed_openai_params": ["reasoning_effort"]}`.
  **`chat_structured()` этого НЕ делает.** Это нарушает правило из MEMORY.md
  («любой LLM-вызов обязан передавать reasoning_effort»), а финальный вызов — самый
  тяжёлый. Там же остался отладочный `print(f"result: ${content}")` (строка ~125):
  он печатает в консоль весь ответ LLM.
- `app/summarize/docx_export.py`: `write_docx(MeetingSummary, SummaryMeta, path)`,
  фиксированные русские заголовки разделов.
- GUI: кнопка «Саммаризация» на правой панели «Сеанс» (фиксированная ширина 300 px)
  вызывает `request_summarize(session_dir)` → `SessionWorker.summarize(session_dir)`
  → daemon-поток → `run_summarization`. Сигналы `summarize_status/done/failed`.
- `app/gui/prefs.py` хранит выбор устройств в QSettings. Особенности QSettings
  описаны в MEMORY.md: кэш пути на Linux, `sync()` возвращает `None`, INI-секции.
- `app/config.py`: настроек `EMBED_*` нет. Эксперименты читают
  `EMBED_URL/EMBED_API_KEY/EMBED_MODEL` прямо из env.
- Зависимости: `numpy`, `openai`, `python-docx`, `pydantic-settings`, `PySide6` —
  уже базовые. `markdown-it-py` есть в `uv.lock` только транзитивно; полагаться
  на него нельзя.

### Чем эксперименты отличаются от приложения

| Аспект | Приложение | `summary_tests/*` |
|---|---|---|
| Клиент | sync `OpenAI`, `chat()` / `chat_structured()` | `AsyncOpenAI`, у каждого эксперимента свой helper, свой retry-цикл |
| Чанкинг | `chunk_text` (по строкам, с защитой от деградации, покрыт тестами) | три разных sentence-чанкера |
| Вход | `transcript.json` → merged segments → строки `[mm:ss] Спикер: текст` | `.txt`/`.json` → плоский текст **без спикеров** |
| Размер чанка | `LLM_CHUNK_CHARS=8000` | 32000 (по ACL 2025) |
| k-means | — | две почти одинаковые numpy-реализации |
| Выход | `MeetingSummary` → docx | Markdown в `output/<name>.md`, у каждого свой формат |
| Параллелизм | нет, map идёт последовательно | `asyncio.gather` + семафор 4 |

`summary_tests/map_reduce` — **другой** map-reduce: рекурсивный reduce и
collapse, параллельный map, 32k-чанки, без спикеров. Пользователь сказал:
«по умолчанию map-reduce, **который уже есть в приложении**». Поэтому пункт
«Map-Reduce» в GUI — это существующий алгоритм приложения. Эксперимент
`map_reduce/` не портируется (см. открытый вопрос Q1).

### Инварианты, которые нельзя сломать

1. Все ошибки саммаризации приходят в GUI как `SummarizationError` с русским
   текстом. Они не терминальны.
2. Файлы `transcript.*` никогда не перезаписываются и не удаляются.
3. `on_status` — best-effort прогресс, его может вызвать любой поток.
4. Пакет `app.summarize` не импортирует Qt. CLI не импортирует PySide6.
5. Каждый `chat.completions.create` передаёт `reasoning_effort="medium"` и
   `extra_body={"allowed_openai_params": ["reasoning_effort"]}`.
6. Путь по умолчанию (map-reduce + промпт по умолчанию) ведёт себя как сейчас:
   те же вызовы LLM, тот же структурированный `MeetingSummary`, тот же вид docx.
   (Заменено, см. выше: структурированный вывод удалён, map параллельный.)
7. Без новых внешних зависимостей, если на это нет явного решения.

---

## Ключевое напряжение: пользовательский «формат» против фиксированной схемы

Сейчас финальный шаг возвращает **структурированный объект** `MeetingSummary`
(native `json_schema`, strict). Он рендерится в docx с фиксированными
заголовками. Если пользователь описывает формат отчёта своим промптом, это
противоречит фиксированной схеме.

### Варианты

| # | Вариант | Выполняет просьбу («свой формат»)? | Путь по умолчанию | Сложность | Вердикт |
|---|---|:---:|:---:|:---:|---|
| a | **Всегда свободный текст.** Все промпты, включая дефолтный, выдают Markdown → свой md→docx-рендерер. Схему убираем | да | **регресс**: теряем strict-схему, «—» для пустых разделов и стабильный docx | 1 путь, 1 рендерер | Реализован при реализации (см. «Изменения», п.1) |
| b | **Схема остаётся, пользователь правит только инструкции** к полям | **нет**: разделы и их порядок не меняются | без изменений | минимальная | Отклонён |
| c | **Гибрид.** Промпт по умолчанию → структурированный `MeetingSummary` (как сейчас). Свой промпт → свободный Markdown → минимальный md→docx-рендерер | да | **без изменений** | 2 ветки финального шага (~15 строк) + рендерер (~80 строк) | Выбран → заменено, см. выше (п.1) |
| d | Схему задаёт пользователь (свой JSON Schema) | да | без изменений | высокая, UX для не-программиста плохой | Отклонён (астронавтика) |
| e | Свой промпт → **общая документная схема** `{sections: [{heading, paragraphs[], bullets[]}]}` через strict json_schema | частично | без изменений | средняя | Отклонён |

**Почему (c).**

- Путь по умолчанию не меняется (инвариант 6). Решение про native structured
  output в MEMORY.md принималось осознанно и остаётся в силе.
- Свой промпт даёт полную свободу формата: разделы, порядок, детализация, стиль.
  Для этого у LLM естественный выход — Markdown.
- Формат и стратегия становятся **ортогональны**: стратегия решает, *как* сжать
  стенограмму, а финальный шаг — *в каком виде* отдать отчёт. Эти две оси
  выбираются независимо. Собственные Markdown-форматы экспериментов (например,
  «TL;DR / Ключевые моменты / Чего не покрыто» у EACSS) не переносятся. При
  желании пользователь вставит их в свой промпт.

**Почему не (e).** Появляются два источника правды о формате: схема и промпт.
Они конфликтуют: пользователь просит нумерованный список или таблицу задач, а
схема этого не выражает. Длинная проза внутри JSON-строк у LLM обычно выходит
хуже (format tax). Выигрыш — не нужен Markdown-парсер — не окупает потерю
свободы формата.

**Почему не (a) по умолчанию.** Это проще всего (один путь), но ломает текущий
надёжный путь и противоречит свежему архитектурному решению. Если пользователю
важнее прозрачность («промпт по умолчанию — это видимый текст, который можно
скопировать и поправить»), (a) — законная альтернатива. См. Q2.

### Как рендерить Markdown в docx без новой зависимости

Нужна своя минимальная функция `write_markdown_docx(markdown, meta, path)` в
`docx_export.py`. Она поддерживает подмножество, которое мы сами просим у LLM
в системном промпте (`MARKDOWN_OUTPUT_RULES`):

- `#`, `##`, `###+` → `add_heading(level=1..3)`;
- `-`/`*`/`+` → стиль `List Bullet` (отступ ≥ 2 пробела → `List Bullet 2`);
- `1.` / `1)` → `List Number`;
- `**жирный**` → жирный run (курсив не обязателен);
- `---`/`***` пропускаем; строки code fence пропускаем, содержимое выводим
  обычными абзацами;
- строки таблиц `|…|` выводим как обычный текст (осознанная деградация);
- любая другая непустая строка → абзац; пустой ответ → «—».

Рендерер **никогда не падает** на странном Markdown: худший исход — видимые
символы разметки. Шапка документа (название сеанса, дата, участники,
длительность) общая с `write_docx`, её выносим в приватный helper.

Отвергнутые альтернативы: `pypandoc` (внешний бинарник pandoc), `markdown` +
HTML→docx (две новые зависимости), `markdown-it-py` напрямую (сейчас
транзитивная зависимость, её пришлось бы объявить явно, а подмножество из
6 правил её не оправдывает).

---

## Как пользовательский промпт встраивается во все три стратегии

### Вариант с плейсхолдером (отклонён)

Пользователь пишет промпт с `{summaries}`/`{content}`, а код делает `.format()`.

- Пользователь забывает плейсхолдер → LLM не получает материал.
- Фигурные скобки в тексте пользователя (пример JSON, «{имя}») → `KeyError`/
  `IndexError` в `str.format`.
- У стратегий разный материал (конспекты; извлечённые предложения; саммари +
  поддерживающие контексты), и одного имени плейсхолдера не хватает.

### Выбранный вариант: код собирает промпт, пользователь пишет только формат

Каждая стратегия **сжимает** стенограмму и возвращает `FinalInput`:

- `framing` — принадлежит стратегии. Описывает, что за материал и какие у метода
  правила. Пример для hierarchical: «контексты используй только для вычитки».
- `material` — сам материал со своей подписью, например «Конспекты
  фрагментов:\n…».

**Общий финальный шаг** (`_finalize` в `pipeline.py`) собирает запрос:

```
по умолчанию (structured, как сейчас):
  system = prompts.SYSTEM
  user   = framing + "\n\n" + prompts.DEFAULT_FORMAT_INSTRUCTIONS + "\n\n" + material
  → chat_structured(..., MeetingSummary)

свой промпт (free-text Markdown):
  system = prompts.SYSTEM + "\n\n" + prompts.MARKDOWN_OUTPUT_RULES
  user   = framing + "\n\n" + material + "\n\n"
           + "Требования к формату итогового отчёта:\n" + custom_prompt
  → chat(...)  → str (Markdown)
```

- Плейсхолдеров нет, забыть нечего. Текст пользователя всегда подставляется
  как **значение**, никогда не как шаблон. Правило для реализации: **никогда
  не вызывать `.format()` на строке, в которой есть текст пользователя.**
- В пути со своим промптом требования к формату стоят **после** материала. На
  длинном контексте инструкции в конце соблюдаются лучше. Путь по умолчанию
  сохраняет текущий порядок (инвариант 6).
- `MARKDOWN_OUTPUT_RULES` (код, пользователь не редактирует) задаёт синтаксис,
  который понимает рендерер: заголовки `#`, списки `-`/`1.`, `**жирный**`, без
  таблиц и блоков кода. Там же сказано: при конфликте с общими указаниями о
  стиле («кратко») побеждают требования пользователя, но выдумывать факты
  по-прежнему нельзя.

### Что считается «финальным шагом» у каждой стратегии

| Стратегия | Промежуточные шаги (не редактируются) | Финальный шаг = `_finalize(FinalInput)` |
|---|---|---|
| `map_reduce` (приложение) | MAP по чанкам `LLM_CHUNK_CHARS` (`MAP_INSTRUCTIONS`) | 1 чанк → framing «стенограмма встречи», material = чанк (сейчас это SINGLE). >1 чанка → framing «последовательные конспекты фрагментов, объедини без повторов», material = конспекты (сейчас это REDUCE). **Число вызовов не меняется.** |
| `eacss` | extractive: чанки 32000 → предложения → эмбеддинги → k-means → по 1 предложению на кластер. При переполнении (извлечённое > 32000) MAP по частям | Извлечённое помещается → framing «ключевые предложения, извлечённые из стенограммы; только на их основе; отсутствующее так и помечай», material = предложения. Переполнение → framing «саммари частей», material = «Часть i из n: …». **Как в эксперименте:** stuff- или reduce-вызов и есть финальный. |
| `hierarchical` | Уровень 1: саммари каждого чанка (LEVEL1) + extractive-контекст. Merge-уровни (MERGE, Table 8), пока узлы не влезут в **одну** группу | Последний merge (единственная группа или принудительный: `MAX_MERGE_LEVELS` / ничего не группируется) → framing = семантика Table 8 («объедини саммари; контексты только для вычитки»), material = «Саммари 1..k» + «Поддерживающие контексты». 1 чанк → framing «стенограмма», material = чанк: сливать нечего, это эквивалент LEVEL1 с нужным форматом. |

Промежуточные промпты — часть метода (перенесены дословно из статей и
экспериментов). Пользователь их не редактирует: иначе можно сломать метод, и
UX становится сложнее.

---

## Решения по остальным вопросам

### D1. Где живут стратегии и какой у них интерфейс

| Вариант | За | Против | Вердикт |
|---|---|---|---|
| **Подпакет `app/summarize/strategies/`**: 3 модуля-функции + registry-dict | Явная граница «метод сжатия». Реестр из 3 записей | +1 уровень пакета | **Выбран** |
| Плоские модули `strategy_*.py` | Меньше вложенности | Реестр некуда положить без цикла импорта | — |
| Иерархия классов (ABC `Strategy`) | «Классика» | У стратегий нет состояния, класс — церемония | Отклонён |
| Plugin discovery / entry points | — | Спекулятивная гибкость | Отклонён |

Интерфейс — **чистая функция**. Протоколов и базовых классов нет:

```python
# app/summarize/strategies/base.py
@dataclass(frozen=True)
class FinalInput:
    framing: str   # strategy-owned description of `material` + method rules (RU)
    material: str  # condensed content incl. its own caption

@dataclass(frozen=True)
class StrategyContext:
    client: Any                      # sync OpenAI client from client.build_client
    llm: LLMSettings
    embedder: "Embedder | None"      # only for requires_embeddings strategies
    notify: Callable[[str], None]    # best-effort progress (pipeline._notify bound)

Condense = Callable[[str, StrategyContext], FinalInput]   # (transcript_text, ctx)

# app/summarize/strategies/__init__.py
@dataclass(frozen=True)
class StrategyInfo:
    id: str
    label: str                  # Russian label for the GUI combo
    requires_embeddings: bool
    condense: Condense

STRATEGIES: dict[str, StrategyInfo]   # insertion order = combo order
DEFAULT_STRATEGY_ID = "map_reduce"
```

Идентификаторы и подписи: `map_reduce` — «Map-Reduce (по умолчанию)»,
`eacss` — «EACSS: извлечение + LLM», `hierarchical` — «Иерархическое слияние
с контекстом».

**Вход стратегии** — уже подготовленный текст `[mm:ss] Спикер: текст` (после
`merge_consecutive(max_chars=llm.chunk_chars)` и `build_transcript_text`).
Эксперименты отбрасывали спикеров. Мы их сохраняем: без них LLM не отличит «Я»
от «Собеседников», а это нужно для задач и ответственных. Это осознанное
отличие от экспериментов (допущение A3).

### D2. Что переиспользуется, а что портируется

| Компонент | Решение |
|---|---|
| Загрузка стенограммы, merge, рендер строк | Остаются в `pipeline.py`/`chunking.py`, стратегии их не дублируют |
| Чанкинг | **Один** `chunking.chunk_text` для всех стратегий. eacss/hierarchical делят чанк на предложения уже после чанкинга (`split_sentences(chunk)`). Три sentence-чанкера из экспериментов не переносятся |
| LLM-вызовы | Только `client.chat` / `client.chat_structured`. Свои retry-циклы экспериментов не переносятся: `OpenAI` SDK по умолчанию сам делает 2 ретрая на connection errors / 408 / 409 / 429 / 5xx |
| Эмбеддинги, предложения, k-means | Новый `app/summarize/extractive.py`: `Embedder` (sync `OpenAI`, батчи по 64, порядок по `.index`, ошибки → `SummarizationError` «Ошибка сервиса эмбеддингов…»), `split_sentences`, **один** `kmeans(X, k, *, iters, seed=42)` |
| Правила отбора предложений | Остаются в модуле стратегии. У eacss по 1 предложению на кластер, `k=min(10,max(3,n//10))`, нормированные векторы. У hierarchical round-robin до 20 предложений; если предложений ≤ 20, эмбеддинги не запрашиваются |
| Константы методов (32000 / 500 / 20 / `MAX_MERGE_LEVELS=10`) | Модульные константы стратегий, **без** новых env-настроек (Q5) |
| Промпты | `prompts.py` = **общие** промпты (`SYSTEM`, `DEFAULT_FORMAT_INSTRUCTIONS`, `MARKDOWN_OUTPUT_RULES`, `CUSTOM_FORMAT_TEMPLATE`). Промпты метода (MAP, LEVEL1, MERGE-framing и т.д.) лежат в модуле своей стратегии. `REDUCE_INSTRUCTIONS`/`SINGLE_INSTRUCTIONS` раскладываются на framing (в `map_reduce.py`) + `DEFAULT_FORMAT_INSTRUCTIONS` |
| `SummarizationError` | Переезжает в `app/summarize/errors.py`, чтобы стратегии могли её бросать без цикла импорта с `pipeline.py`. `pipeline.py` реэкспортирует её, старый импорт продолжает работать |
| `summary_tests/` | **Остаются как есть**: замороженная лаборатория и эталон для сравнения. Кросс-импортов нет в обе стороны: приложение не импортирует `summary_tests`, эксперименты не импортируют `app` (контракт их README) |

### D3. Sync или async

| Вариант | За | Против | Вердикт |
|---|---|---|---|
| **Sync, последовательно** | Совпадает с текущим `chat()`, daemon-поток воркера. Минимум кода. Прогресс «фрагмент i/n» сохраняется | Долгие встречи в eacss/hierarchical медленнее экспериментов (~30 с на вызов × N) | Заменено, см. выше (п.3: параллелизм через `LLM_CONCURRENCY`) |
| Sync + пул потоков на fan-out фазах | Скорость как в экспериментах. Sync-клиент `OpenAI` (httpx) потокобезопасен | Чуть сложнее прогресс и обработка ошибок. Меняет поведение пути по умолчанию | Реализовано при реализации (п.3; `ThreadPoolExecutor` заменён пулом daemon-потоков в `parallel.py`) |
| Портировать asyncio как есть (`asyncio.run` в daemon-потоке) | Ближе к оригиналу | Второй клиентский стек, дублирование helper'ов, event loop внутри потока | Отклонён |

Код стратегий пишем так, чтобы fan-out был одним циклом `for` по списку.
Тогда замена на `pool.map` — локальная правка.

### D4. Эмбеддинги: конфиг и UX, когда они не настроены

Конфиг: в `app/config.py` добавляется `EmbedSettings(_Base)` с `env_prefix="EMBED_"`:
`url: str = ""`, `api_key: str = "not-needed"`, `model: str = ""`,
`request_timeout: float = 60.0` и свойство `is_configured = bool(url.strip() and
model.strip())`. `Config.embed = EmbedSettings()`. Секция `EMBED_*` добавляется в
`.env.example`. Имена переменных совпадают с экспериментами, так что один `.env`
подходит обоим.

| Вариант UX | За | Против | Вердикт |
|---|---|---|---|
| **Пункты видны, но недоступны** (disabled) + tooltip «Нужны EMBED_URL и EMBED_MODEL в .env» + **backend-guard** | Пользователь видит, что методы есть и что для них нужно | Состояние читается при старте GUI (после правки `.env` нужен перезапуск) | **Выбран** |
| Пункты доступны, ошибка при запуске | Проще в GUI | Ошибка обнаруживается поздно | Остаётся как guard |
| Пункты скрыты | — | Пользователь не узнает о методах | Отклонён |

Backend-guard в `run_summarization`: если `requires_embeddings` и
`not config.embed.is_configured`, то `SummarizationError` **до** любого сетевого
вызова. Этот же guard ловит случай «сохранённый выбор = eacss, а `EMBED_*`
потом убрали».

### D5. Хранение выбора и промпта

| Вариант | За | Против | Вердикт |
|---|---|---|---|
| **QSettings через `app/gui/prefs.py`** | Уже есть прецедент (устройства), per-user. Многострочный текст нормально хранится (INI экранирует `\n`, в реестре REG_SZ). Не смешивается с секретами | Особенности QSettings (см. MEMORY) | **Выбран** |
| `.env` / `app/config.py` | — | Это конфиг развёртывания (эндпоинты, ключи). Из GUI его не отредактировать, многострочный текст в env неудобен | Отклонён |
| Отдельный файл промпта (`report_prompt.md`) | Можно править во внешнем редакторе | Новый механизм, пути, кодировки | Отклонён |

Ключи: `summarize/strategy` (str), `summarize/use_default_prompt` (bool),
`summarize/custom_prompt` (str). API в стиле существующих функций —
**тотальные, никогда не бросают исключений**:

```python
@dataclass
class SummarizePrefs:
    strategy: str | None          # None = not saved / corrupt
    use_default_prompt: bool      # default True
    custom_prompt: str            # default "" (preserved even when unused)

def load_summarize_prefs() -> SummarizePrefs
def save_summarize_prefs(p: SummarizePrefs) -> None
```

Bool читается терпимо: `v is True or (isinstance(v, str) and v.lower() == "true")`.
Отсутствующий или повреждённый ключ → `True`. В INI и в реестре Windows QSettings
отдаёт bool как строку `"true"`/`"false"`. Проверку `id in STRATEGIES` делает
GUI, а не prefs: prefs ничего не знает о домене.

### D6. Передача опций GUI → worker → pipeline

| Вариант | Вердикт |
|---|---|
| **`SummarizationOptions` (frozen dataclass, `app/summarize/pipeline.py`, реэкспорт в `app.summarize`)** | **Выбран**: один объект в Qt-сигнале, без Qt, у каждого поля есть значение по умолчанию |
| kwargs `run_summarization(strategy=..., custom_prompt=...)` | Равноценно, но сигнал несёт два поля вместо одного объекта |
| Мутировать `config` (как `start_session` делает с `stt.engine`) | Отклонён: `config` — это env-настройки развёртывания, опции — выбор пользователя на один запуск |

```python
@dataclass(frozen=True)
class SummarizationOptions:
    strategy: str = DEFAULT_STRATEGY_ID
    custom_prompt: str | None = None   # None → built-in structured protocol (MeetingSummary)

def run_summarization(
    session_dir: str | Path,
    config: Config,
    on_status: Callable[[str], None] | None = None,
    options: SummarizationOptions | None = None,   # None ≡ SummarizationOptions()
) -> Path
```

(Фактическая сигнатура добавляет `cancel_event: threading.Event | None = None`
— заменено, см. выше, п.7.)

**Состояние «использовать промпт по умолчанию, но текст сохранить» полностью
живёт в GUI.** Backend получает только **эффективный** промпт:
`custom_prompt = None if use_default else text`. У backend нет понятия
«сохранённый, но неиспользуемый» — это забота UI. Пустой или пробельный
`custom_prompt` (не `None`) → `SummarizationError` «Пользовательский промпт
пуст…». Неизвестная стратегия → `SummarizationError`.

### D7. UI

| Вариант | За | Против | Вердикт |
|---|---|---|---|
| **Комбо + чекбокс на панели «Сеанс», текст — в модальном диалоге «Промпт отчёта…»** | Текущий режим виден сразу. Переключение в один клик. У редактора есть место | +1 маленький модуль диалога | **Выбран** |
| Всё на панели, включая `QPlainTextEdit` | Нет диалога | Панель 300 px и так плотная (8 строк инфо + 6 кнопок). При высоте окна 700 px редактор не поместится | Отклонён |
| Отдельный диалог «Настройки саммаризации…» со всеми тремя контролами | Панель чище | Текущий режим не виден, смена метода — лишние клики | Отклонён |

Эскиз панели (новые элементы — над индикатором прогресса):

```
┌ Сеанс ─────────────────────────────┐
│ [20260925_101500_Планёрка ▾][Обновить]│
│ Готово                               │
│ Название: …   Папка: …   …           │
│ ─────────────────────────────────── │
│ Метод: [Map-Reduce (по умолчанию) ▾] │  ← QComboBox из STRATEGIES; eacss/hierarchical
│                                      │     disabled + tooltip, если EMBED_* не заданы
│ [x] Использовать промпт по умолчанию │  ← tooltip: «Встроенный формат: TL;DR,
│ [Промпт отчёта…]                     │     решения, задачи, темы, открытые вопросы»
│ ▒▒▒▒▒▒▒▒ (прогресс)  статус…         │
│ [Саммаризация]                       │
│ [Открыть отчёт] …                    │
└──────────────────────────────────────┘
```

Диалог (`app/gui/prompt_dialog.py`, функция
`edit_report_prompt(parent, text: str) -> str | None`, где `None` = «Отмена»):

```
┌ Промпт итогового отчёта ─────────────────────────────┐
│ Опишите, каким должен быть итоговый отчёт: разделы,    │
│ порядок, степень детализации. Стенограмма или          │
│ промежуточные конспекты подставляются автоматически.   │
│ ┌────────────────────────────────────────────────────┐ │
│ │ QPlainTextEdit (~15 строк)                          │ │
│ └────────────────────────────────────────────────────┘ │
│                                     [OK]  [Отмена]     │
└────────────────────────────────────────────────────────┘
```

Поведение:

- Чекбокс **не трогает** текст. Переключение сразу сохраняется в prefs.
- Кнопка «Промпт отчёта…» доступна **всегда**: сохранённый текст можно править и
  при включённом «по умолчанию». OK сохраняет текст в prefs. Чекбокс при этом
  **не переключается автоматически** (без скрытых побочных эффектов, см. Q9).
- Если сохранённый текст пуст, диалог открывается с
  `prompts.CUSTOM_FORMAT_TEMPLATE`. Это Markdown-аналог разделов по умолчанию,
  стартовая точка для правки. (Заменено, см. выше, п.6: `CUSTOM_FORMAT_TEMPLATE`
  упразднён, диалог предзаполняется `DEFAULT_REPORT_PROMPT`.)
- Смена метода в комбо сохраняется в prefs **только при действии пользователя**.
  Программный предвыбор идёт с `blockSignals`, как у устройств. Если сохранённый
  метод неизвестен или сейчас недоступен (нет `EMBED_*`), предвыбирается
  `DEFAULT_STRATEGY_ID`, а сохранённое значение не перезаписывается.
- При нажатии «Саммаризация», если чекбокс снят и текст пуст:
  `QMessageBox.warning` («Свой промпт пуст: заполните «Промпт отчёта…» или
  включите промпт по умолчанию»), запуск не происходит. Backend-guard остаётся
  как вторая линия защиты.
- Опции снимаются в момент клика. Смена комбо или чекбокса во время
  выполнения на текущий запуск не влияет, поэтому блокировать контролы не нужно.

### D8. Имя выходного файла

| Вариант | Вердикт |
|---|---|
| **Всегда `summary.docx` (перезапись, как сейчас) + строка «Метод: …» в шапке** | Выбран → заменено, см. выше (п.2: timestamped-файл на каждый запуск) |
| `summary_<strategy>.docx` | Удобно сравнивать, но «Открыть отчёт» и `has_report` пришлось бы выбирать из нескольких файлов (Q4) |
| Версии с timestamp | Мусор в папке сеанса |

`SummaryMeta` получает поле `method: str | None = None` (label стратегии; для
своего промпта с пометкой «, свой формат»). Обе функции записи выводят его
в шапке.

---

## Decision (сводка)

### Раскладка модулей

```
app/summarize/
  __init__.py        # exports: run_summarization, SummarizationOptions, SummarizationError,
                     #          STRATEGIES, DEFAULT_STRATEGY_ID
  errors.py          # NEW  SummarizationError (moved; re-exported from pipeline)
  pipeline.py        # orchestration only: validate options → load → merge → text →
                     #   STRATEGIES[id].condense → _finalize → export; SummarizationOptions
  prompts.py         # SHARED prompts: SYSTEM, DEFAULT_FORMAT_INSTRUCTIONS,
                     #   MARKDOWN_OUTPUT_RULES, CUSTOM_FORMAT_TEMPLATE
  client.py          # chat / chat_structured (+reasoning kwargs, −debug print), build_client
  extractive.py      # NEW  Embedder (remote /v1/embeddings), split_sentences, kmeans
  chunking.py        # unchanged
  schema.py          # unchanged
  docx_export.py     # write_docx (unchanged API) + write_markdown_docx; shared header; meta.method
  strategies/
    __init__.py      # NEW  StrategyInfo, STRATEGIES, DEFAULT_STRATEGY_ID
    base.py          # NEW  FinalInput, StrategyContext, Condense
    map_reduce.py    # NEW  existing algorithm moved out of pipeline.py (+ MAP prompt, framings)
    eacss.py         # NEW  port of summary_tests/eacss (sync, shared chunker/client)
    hierarchical.py  # NEW  port of summary_tests/hierarchical_context (sync)
app/config.py        # + EmbedSettings (EMBED_*), Config.embed
app/gui/prefs.py     # + SummarizePrefs, load/save_summarize_prefs
app/gui/prompt_dialog.py  # NEW  edit_report_prompt(parent, text) -> str | None
app/gui/worker.py    # summarize(session_dir, options)
app/gui/main_window.py    # combo + checkbox + dialog button; request_summarize(dir, options)
```

Направление зависимостей (только внутрь, циклов нет):
`gui → app.summarize (public API) → pipeline → strategies → {client, chunking,
extractive, errors, prompts}`. Пакет `app.summarize` не знает о Qt.

### Поток данных

```
MainWindow._on_summarize_clicked
  ├─ prefs/виджеты → SummarizationOptions(strategy, custom_prompt | None)
  └─ request_summarize.emit(session_dir, options)            # Signal(object, object), queued
SessionWorker.summarize(session_dir, options)                 # worker QThread → daemon thread
  └─ run_summarization(session_dir, config, on_status, options)
       ├─ validate: strategy ∈ STRATEGIES; embeddings configured if required;
       │            custom_prompt is None or non-blank            → SummarizationError
       ├─ _load_segments → merge_consecutive(max_chars) → build_transcript_text
       ├─ ctx = StrategyContext(build_client(llm), llm, Embedder(cfg.embed)|None, notify)
       ├─ final_input = STRATEGIES[id].condense(text, ctx)     # FinalInput(framing, material)
       ├─ result = _finalize(ctx, final_input, custom_prompt)  # MeetingSummary | str
       └─ write_docx(result, meta) | write_markdown_docx(result, meta) → summary.docx
  → summarize_done(path) | summarize_failed(msg)
```

Фактический поток отличается в деталях (заменено, см. выше): `_finalize`
возвращает только Markdown; финальный вызов тоже идёт через `parallel_map`
(ради отмены); файл — `summary_<метод>_<YYYYMMDD_HHMMSS>.docx`; сигнал
`summarize_cancelled` добавлен.

Обёртка ошибок как сейчас: `SummarizationError` пробрасывается без изменений,
прочие исключения → «Ошибка обращения к LLM: …». Ошибки эмбеддингов
оборачиваются ещё внутри `Embedder`, со своим текстом. Ошибка записи docx →
«возможно, файл открыт в Word».

### Новые зависимости

**Нет.** `numpy`, `openai`, `python-docx`, `pydantic-settings`, `PySide6` уже
базовые.

---

## Consequences

### Что становится проще / лучше

- Пользователь выбирает метод и формат отчёта независимо друг от друга.
- `pipeline.py` занимается только оркестрацией. Метод отделён от финального шага
  и от экспорта, каждую часть можно тестировать без сети (fake client, fake embedder).
- Три чанкера, две реализации k-means и три LLM-клиента из экспериментов
  сводятся к одному чанкеру, одному k-means и одному клиенту.
- Исправляется `chat_structured`: добавляется `reasoning_effort`, убирается
  отладочный `print`.

### Что становится сложнее / хуже

- У финального шага две ветки (structured и free-text) и два рендерера.
  (Заменено, см. выше, п.1: один Markdown-путь, один рендерер.)
- Свой формат рендерится **подмножеством** Markdown: таблиц нет, странная
  разметка может остаться видимыми символами.
- eacss/hierarchical в v1 работают последовательно. На длинных встречах они
  медленнее экспериментов. (Заменено, см. выше, п.3: fan-out-фазы параллельны.)
- Появляется `EMBED_*`-конфиг. Доступность методов определяется при старте GUI.
- Порт немного отличается от экспериментов: спикеры в тексте, общий чанкер,
  `max_tokens=LLM_MAX_TOKENS` вместо 1500, таймаут `LLM_REQUEST_TIMEOUT` (120 с)
  вместо 600. Результаты не будут байт-в-байт как в `summary_tests/*/output`.
- Текст промпта по умолчанию **пересобирается** из framing и
  `DEFAULT_FORMAT_INSTRUCTIONS`. По смыслу он эквивалентен текущему REDUCE/SINGLE,
  но дословно может отличаться.

### Что НЕ меняется

- Аудио, VAD, STT, `Session`, CLI, файлы `transcript.*`.
- `schema.MeetingSummary`, `strict_json_schema`, `write_docx` (API),
  `chunk_text`, `merge_consecutive`. (Заменено, см. выше, п.1: структурированный
  вывод удалён.)
- Логика кнопок GUI (`has_report` по `summary.docx`, блокировки во время
  записи и саммаризации), `summarize_*`-сигналы. (Фактическая реализация:
  `has_report` по `find_latest_summary` — самый свежий отчёт, в т.ч. legacy-
  `summary.docx`; добавлен сигнал `summarize_cancelled`.)
- `summary_tests/` — остаётся как есть.

---

## File-level план

> Реализация не входит в данный ADR; ниже — карта правок по файлам.

| Файл | Что меняется |
|---|---|
| `app/summarize/errors.py` **(новый)** | `class SummarizationError(Exception)` |
| `app/summarize/pipeline.py` | Импорт `SummarizationError` из `errors` (реэкспорт). `SummarizationOptions`. `run_summarization(..., options=None)`: валидация, `StrategyContext`, вызов стратегии, `_finalize`, выбор рендерера, `meta.method`. Алгоритм map-reduce и его промпты уходят в `strategies/map_reduce.py` |
| `app/summarize/prompts.py` | Остаются `SYSTEM` и новые `DEFAULT_FORMAT_INSTRUCTIONS` (список полей из REDUCE/SINGLE), `MARKDOWN_OUTPUT_RULES`, `CUSTOM_FORMAT_TEMPLATE`. `MAP_INSTRUCTIONS` → `map_reduce.py`. `REDUCE/SINGLE_INSTRUCTIONS` удаляются (раскладываются на части) |
| `app/summarize/client.py` | `chat_structured`: + `reasoning_effort="medium"`, `extra_body={"allowed_openai_params": ["reasoning_effort"]}`. Лучше одна модульная константа `_REASONING_KWARGS` для `chat` и `chat_structured`. Удалить `print(f"result: ${content}")` |
| `app/summarize/extractive.py` **(новый)** | `Embedder(settings: EmbedSettings)` с `.embed(texts) -> np.ndarray` (батчи по 64, сортировка по `.index`, ошибки → `SummarizationError`). `split_sentences`. `kmeans` (numpy, k-means++, seed 42) |
| `app/summarize/strategies/__init__.py` **(новый)** | `StrategyInfo`, `STRATEGIES`, `DEFAULT_STRATEGY_ID` |
| `app/summarize/strategies/base.py` **(новый)** | `FinalInput`, `StrategyContext`, `Condense` |
| `app/summarize/strategies/map_reduce.py` **(новый)** | Существующий map-reduce: `chunk_text(llm.chunk_chars, llm.chunk_overlap)`, MAP, два framing'а |
| `app/summarize/strategies/eacss.py` **(новый)** | Порт eacss: константы 32000/500, отбор по 1 предложению на кластер, stuff/overflow (MAP по частям) → `FinalInput` |
| `app/summarize/strategies/hierarchical.py` **(новый)** | Порт hierarchical: `SYSTEM_PROMPT`/LEVEL1/MERGE дословно, `Node`, `group_nodes`, `extractive_context`, `MAX_MERGE_LEVELS`, принудительные слияния (через `logger.warning` вместо `print(stderr)`) → `FinalInput` |
| `app/summarize/docx_export.py` | Приватный `_add_header(document, meta)`. `SummaryMeta.method`. `write_markdown_docx(markdown, meta, out_path) -> Path` |
| `app/summarize/__init__.py` | Реэкспорт публичного API |
| `app/config.py` | `EmbedSettings` (`EMBED_URL/API_KEY/MODEL/REQUEST_TIMEOUT`), `is_configured`, `Config.embed` |
| `.env.example` | Секция `EMBED_*` с комментарием «нужно только для EACSS и иерархического метода» |
| `app/gui/prefs.py` | `SummarizePrefs`, `load_summarize_prefs`, `save_summarize_prefs`. Docstring модуля расширить: «GUI preferences» |
| `app/gui/prompt_dialog.py` **(новый)** | `edit_report_prompt(parent, text) -> str \| None` (`None` = «Отмена») |
| `app/gui/worker.py` | `@Slot(object, object) summarize(session_dir, options)` → `run_summarization(..., options=options)` |
| `app/gui/main_window.py` | `request_summarize = Signal(object, object)`. Виджеты метода, чекбокса и кнопки диалога в `_build_session_panel`. Предвыбор из prefs. Сохранение при действиях пользователя. Проверка пустого промпта в `_on_summarize_clicked` |
| `README.md`, `MEMORY.md` | Раздел «Саммаризация встреч»: методы, `EMBED_*`, свой промпт. MEMORY: новые решения и ссылка на ADR-002 |
| `tests/…` | См. план работ |

---

## План работ

Владельцы: **BE** — backend implementer, **FE** — frontend (GUI) implementer.
У каждого пакета работ один владелец. Все тесты запускаются через
`uv run pytest` и работают без сети: fake-клиент с
`.chat.completions.create(**kw)` / `.embeddings.create(**kw)`, который
записывает kwargs.

```
B1 ─┬─► B3 ─┬─► B5 ─┐
B2 ─┘       ├─► B6 ─┼─► B7
B1 ─► B4 ───┘       │
F1 ─────► F2 ◄── B3 ┘ (F2 can start against the interface in this ADR)
```

### B1. Конфиг эмбеддингов и гигиена клиента — BE

- Что: `errors.py` (перенос `SummarizationError` + реэкспорт); `EmbedSettings` +
  `Config.embed` + `.env.example`; `client.py`: reasoning-kwargs в
  `chat_structured` (общая константа) и удаление `print`.
- Зависимости: нет.
- DoD: `from app.summarize.pipeline import SummarizationError` работает; оба
  LLM-helper'а передают reasoning-kwargs; в `app/summarize/` нет `print(`.
- Тесты (`tests/test_summarize_client.py`, `tests/test_config_embed.py`):
  fake client фиксирует kwargs `chat` и `chat_structured` —
  `reasoning_effort == "medium"` и
  `extra_body == {"allowed_openai_params": ["reasoning_effort"]}`;
  `EmbedSettings.is_configured` для пустого, частичного и полного набора (через
  `monkeypatch.setenv` и `_env_file=None`).

### B2. Рендерер Markdown → docx — BE

- Что: `_add_header`, `SummaryMeta.method`, `write_markdown_docx` по подмножеству
  из этого ADR.
- Зависимости: нет (параллельно с B1).
- DoD: `write_docx` даёт прежний результат (кроме необязательной строки «Метод»);
  рендерер не бросает исключений на произвольном тексте.
- Тесты (`tests/test_summarize_docx_markdown.py`, чтение через
  `docx.Document(path)`): заголовки `#/##/###` → `Heading 1..3`; `-`/`*` →
  `List Bullet`; `1.` → `List Number`; `**x**` → run с `bold=True`; пустой
  Markdown → «—»; таблица и code fence → обычные абзацы без исключения; шапка
  содержит «Метод: …», когда `meta.method` задан; регресс `write_docx` на
  фиксированном `MeetingSummary`.

### B3. Каркас стратегий, перенос map-reduce и финальный шаг — BE

- Что: `strategies/{__init__,base,map_reduce}.py`; `SummarizationOptions`;
  `run_summarization(..., options)`; `_finalize`; рефакторинг `prompts.py`;
  реестр пока с одной записью `map_reduce`.
- Зависимости: B1, B2.
- DoD: путь по умолчанию (`options=None`) делает **то же число и тот же вид**
  вызовов, что и сейчас: N MAP через `chat` + 1 `chat_structured` (или 1
  `chat_structured` для одного чанка), и пишет тот же docx. Свой промпт даёт один
  финальный `chat` без `response_format` и docx через `write_markdown_docx`.
- Тесты (`tests/test_summarize_pipeline.py`; fake client, `tmp_path` с
  `transcript.json`):
  - default, 1 чанк → 1 вызов с `response_format.type == "json_schema"`;
  - default, 3 чанка → 3 вызова без `response_format` + 1 с ним;
  - custom → последний вызов без `response_format`; текст пользователя стоит в
    user-сообщении **после** материала; `MARKDOWN_OUTPUT_RULES` в system;
  - custom с фигурными скобками (`"{summaries} {0} {x}"`) → без исключений,
    текст передан дословно;
  - `custom_prompt="   "` → `SummarizationError`, ноль вызовов LLM;
  - неизвестная стратегия → `SummarizationError`;
  - исключение клиента → `SummarizationError` с «Ошибка обращения к LLM»;
  - `transcript.json`/`transcript.txt` побайтно не изменились;
  - `on_status` получил «чтение стенограммы…» и «завершена».

### B4. Общий extractive-модуль — BE

- Что: `extractive.py` (`Embedder`, `split_sentences`, `kmeans`).
- Зависимости: B1 (`EmbedSettings`, `errors`).
- DoD: API из этого ADR; ошибка эндпоинта → `SummarizationError` с русским
  текстом про сервис эмбеддингов.
- Тесты (`tests/test_summarize_extractive.py`; можно адаптировать тесты
  k-means и предложений из `summary_tests`): детерминизм `kmeans` при seed,
  `k > n` сводится к `n`; `split_sentences` по `.!?…` и `\n`; `Embedder`
  режет 130 текстов на батчи 64/64/2 и восстанавливает порядок по `.index`
  (fake client отдаёт `data` в перемешанном порядке); исключение клиента →
  `SummarizationError`.

### B5. Стратегия EACSS — BE

- Что: `strategies/eacss.py` + запись в `STRATEGIES` (`requires_embeddings=True`).
- Зависимости: B3, B4.
- DoD: stuff и overflow ветки → корректный `FinalInput`; прогресс через
  `ctx.notify`; guard «эмбеддинги не настроены» срабатывает в pipeline до
  сети.
- Тесты (`tests/test_strategy_eacss.py`; fake embedder на md5 и fake client):
  короткий текст → 0 вызовов LLM в `condense`, material = извлечённые
  предложения в исходном порядке, их число ≤ k; текст, у которого извлечённое
  > 32000 (уменьшить константу через monkeypatch) → MAP-вызовы по частям +
  framing reduce; `run_summarization(strategy="eacss")` без `EMBED_*` →
  `SummarizationError`, ноль вызовов.

### B6. Стратегия Hierarchical Extract-Support — BE

- Что: `strategies/hierarchical.py` + запись в `STRATEGIES`.
- Зависимости: B3, B4 (параллельно с B5).
- DoD: 1 чанк → `FinalInput` с сырым чанком без промежуточных вызовов;
  несколько чанков → N вызовов LEVEL1, merge-уровни до одной группы, финальная
  группа уходит в `FinalInput` (саммари + контексты); принудительные ветки
  (`MAX_MERGE_LEVELS`, «ничего не группируется») логируют warning и отдают все
  узлы в финал.
- Тесты (`tests/test_strategy_hierarchical.py`; можно адаптировать
  `summary_tests/hierarchical_context/test_pipeline.py`): `group_nodes`
  соблюдает лимит; `extractive_context` не зовёт embedder при ≤ 20 предложениях;
  число LLM-вызовов для N чанков при маленьком лимите через monkeypatch;
  material финала содержит «Поддерживающие контексты»; контекст следующего
  уровня извлекается из склеенных контекстов, а не из саммари.

### F1. Хранение настроек саммаризации — FE

- Что: `SummarizePrefs`, `load_summarize_prefs`, `save_summarize_prefs` в
  `prefs.py`.
- Зависимости: нет (можно начинать сразу).
- DoD: функции тотальные; значения по умолчанию `(None, True, "")`;
  многострочный кириллический текст переживает roundtrip.
- Тесты (в `tests/test_gui_prefs.py`, существующая фикстура `ini_store`):
  roundtrip; bool из строк `"true"`/`"false"`/мусор → `True` по умолчанию;
  `custom_prompt` не-строка → `""`; текст с `\n`, `=`, `;`, `#`, кириллицей
  сохраняется дословно; сохранение с `use_default_prompt=True` не стирает
  `custom_prompt`; в INI есть секция `[summarize]`.

### F2. Панель, диалог и проводка worker'а — FE

- Что: `prompt_dialog.py`; виджеты в `_build_session_panel`; предвыбор из
  prefs; disabled-пункты без `EMBED_*` (через `config.embed.is_configured`,
  config уже загружается в `MainWindow.__init__`); сохранение при действиях
  пользователя; `request_summarize(dir, options)`;
  `SessionWorker.summarize(session_dir, options)`.
- Зависимости: F1, B3 (API `SummarizationOptions`/`STRATEGIES`/
  `DEFAULT_STRATEGY_ID`/`CUSTOM_FORMAT_TEMPLATE`). Можно начинать параллельно
  с B3 по сигнатурам из этого ADR.
- DoD: чек-лист ручной проверки ниже, пункты 3–8. Тексты UI на русском. Без
  изменения логики `_update_action_buttons`.
- Тесты: чистую функцию `resolve_options(prefs_or_widgets) -> SummarizationOptions`
  вынести из `main_window` (например, в `prefs.py` или отдельно) и покрыть
  unit-тестом: use_default → `custom_prompt=None`; иначе текст; неизвестный id →
  `DEFAULT_STRATEGY_ID`. Виджеты — ручная проверка: `pytest-qt` нет в
  зависимостях, и добавлять его ради этого не стоит.

### B7. Документация — BE

- Что: README («Саммаризация встреч»: выбор метода, `EMBED_*`, свой промпт,
  ограничения Markdown); MEMORY.md (решения ADR-002, ловушки: никакого
  `.format()` на тексте пользователя, bool в QSettings — строка).
- Зависимости: B5, B6, F2.
- DoD: `.env.example`, README и `config.py` согласованы.

---

## Чек-лист верификации (ручной, на реальном эндпоинте)

1. `chat_structured` с `reasoning_effort` + strict `json_schema` принимается
   прокси (LiteLLM/FrankAI). Эта комбинация **ещё не проверялась**; от неё
   зависит путь по умолчанию.
2. Путь по умолчанию на реальной сессии (например, `20260924_161938`) до и
   после рефакторинга: то же число вызовов в логе, `summary.docx` с теми же
   разделами.
3. Без `EMBED_*` в `.env`: пункты EACSS и Hierarchical серые, tooltip виден;
   выбрать их нельзя.
4. С `EMBED_*`: выбрать EACSS → «Саммаризация» → docx содержит «Метод: EACSS…».
   То же для Hierarchical.
5. Снять чекбокс при пустом тексте → «Саммаризация» → предупреждение, запуска нет.
6. «Промпт отчёта…» → шаблон подставлен → поправить (например, добавить раздел
   «Риски») → OK → снять чекбокс → «Саммаризация» → docx с разделами из промпта.
7. Включить чекбокс обратно → перезапустить приложение → открыть «Промпт
   отчёта…»: текст на месте. Саммаризация даёт структурированный протокол.
8. Выбор метода, чекбокс и текст сохраняются после перезапуска (Windows:
   `HKCU\Software\LiveRecorder\LiveRecorder\summarize`).
9. Длинная встреча (≥ 1 ч) в Hierarchical укладывается в
   `LLM_REQUEST_TIMEOUT`. Иначе поднять таймаут в `.env` (Q5).

---

## Риски и rollback

### Риски

| # | Риск | Митигирование |
|---|---|---|
| R1 | Прокси отклоняет `reasoning_effort` вместе с `response_format=json_schema` | Чек-лист п.1. Та же пара параметров уже работает в `chat()`, риск низкий. При отказе — отдельное решение, молча не откатываемся |
| R2 | LLM выдаёт Markdown, который рендерер не понимает | `MARKDOWN_OUTPUT_RULES` + рендерер, который деградирует до текста и никогда не падает |
| R3 | Текст промпта по умолчанию дословно поменяется, качество дрейфует | Тест на состав вызовов + ручное сравнение (п.2) |
| R4 | 32k-промпты eacss/hierarchical не укладываются в `LLM_REQUEST_TIMEOUT` (сейчас 300 с, см. выше, п.5) или `max_tokens` | Встроенные ретраи SDK. Рекомендация в README поднять таймаут. Q5 |
| R5 | Долгая работа eacss/hierarchical при последовательных вызовах | Прогресс по шагам. Q6 (ThreadPoolExecutor) как follow-up |
| R6 | Сохранённый метод недоступен (убрали `EMBED_*`) | Предвыбор по умолчанию + backend-guard |

### Rollback

Изменения локализованы в `app/summarize/`, `app/config.py` и трёх GUI-файлах.
Аудио и STT не затрагиваются, миграций данных нет, `summary.docx` —
производный файл. Rollback = `git revert` коммитов фичи. B1 (исправление
`chat_structured`) стоит оставить и после отката: это баг-фикс.

---

## Открытые вопросы (для человека)

- **Q1. Что значит «Map-Reduce» в списке?** Предлагается **существующий алгоритм
  приложения** (чанки 8000, один reduce, спикеры в тексте), а не
  `summary_tests/map_reduce` (рекурсивный reduce/collapse, 32k, параллельный
  map). Подтвердить. Отдельный вопрос: переносить ли рекурсивный reduce в
  приложение, чтобы очень длинные встречи не переполняли контекст финального
  шага? Это существующий риск, в этот ADR не входит.
- **Q2. Гибрид (c) или «всегда Markdown» (a)?** (c) сохраняет текущий надёжный
  structured-протокол, но «промпт по умолчанию» — это не текст, который можно
  взять за основу, а встроенная схема (стартовой точкой служит шаблон). (a)
  проще и прозрачнее, но убирает strict-схему. Рекомендуется (c).
- **Q3. Ограничения своего формата.** Таблиц в docx не будет (выводятся
  текстом). Приемлемо? Нужно ли дополнительно сохранять сырой `summary.md` рядом
  с docx (одна строка кода, без потерь)?
- **Q4. Один `summary.docx` (перезапись) или отдельный файл на метод** для
  сравнения методов на одной встрече?
- **Q5. Параметры eacss/hierarchical.** Оставить константы экспериментов (чанк
  32000, overlap 500) или брать `LLM_CHUNK_CHARS`? Поднять ли дефолтный
  `LLM_REQUEST_TIMEOUT` (120 с) ради 32k-промптов?
- **Q6. Параллелизм.** v1 последовательный. Нужна ли сразу параллельная
  обработка (4 потока), как в экспериментах? Тогда меняется и путь по умолчанию
  (сейчас map последовательный).
- **Q7. Методы без эмбеддингов:** показывать серыми (рекомендуется) или скрывать?
- **Q8. Промпт один на все методы** (рекомендуется) или свой у каждого?
- **Q9. Диалог:** должен ли OK в «Промпт отчёта…» сам снимать «Использовать
  промпт по умолчанию»? Предлагается «нет», без скрытых побочных эффектов.
- **Q10. CLI:** нужна ли команда `python -m app.cli summarize --strategy …
  --prompt-file …`? Сейчас в CLI саммаризации нет, в этот ADR не входит.

## Допущения

- **A1.** Три пункта списка: map-reduce из приложения (по умолчанию), EACSS,
  Hierarchical. `summary_tests/map_reduce` не портируется (Q1).
- **A2.** Свой промпт задаёт **только формат и содержание итогового
  отчёта**. Промежуточные промпты методов пользователь не редактирует.
- **A3.** Стратегии получают текст со спикерами и таймкодами
  (`[mm:ss] Спикер: …`), а не плоский текст, как в экспериментах.
- **A4.** Эмбеддинги берутся из того же `.env` (`EMBED_URL/EMBED_API_KEY/EMBED_MODEL`),
  что и в экспериментах. Доступность методов определяется при старте GUI.
- **A5.** Отчёт всегда на русском. Строки экспериментов «Отвечай на языке
  исходного текста» сохраняются в промежуточных промптах дословно; язык
  финального шага задаёт `SYSTEM`.
- **A6.** Отмену саммаризации не добавляем (её нет и сейчас). (Заменено, см.
  выше, п.4: отмена добавлена.)
- **A7.** `summary_tests/` не меняем и не удаляем. Дублирование кода между
  лабораторией и приложением допустимо: у экспериментов свой контракт
  самодостаточности.
