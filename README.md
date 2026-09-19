# Live Recorder

Лайв-транскрипция встреч: одновременный захват **микрофона** и **системного
звука** (loopback) на раздельных каналах, детекция речи через **Silero VAD** и
распознавание переключаемым **STT**-движком (whisper-large-v3-turbo или
GigaAM-v3). Есть десктоп-GUI на PySide6 и headless CLI для сценариев без
интерфейса.

Каналы не смешиваются: микрофон → «Я», системный звук → «Собеседники». Сессия
целиком пишется на диск (`session.wav`), поэтому её всегда можно
перерасшифровать заново. Целевая среда — **CPU** (рабочий компьютер без GPU):
режим live — best-effort (движки могут не успевать за реальным временем),
надёжный результат даёт batch/file-расшифровка из уже записанного файла.

## Требования

- Windows 10/11 (WASAPI loopback — Windows-специфичная возможность, `soundcard`
  на других ОС системный звук так не отдаёт)
- Python >= 3.12
- [uv](https://docs.astral.sh/uv/) для управления окружением и зависимостями

## Установка

```powershell
uv sync
```

Это ставит базовый набор (whisper, CPU-сборка torch). Чтобы добавить движок
GigaAM-v3 (только русский язык):

```powershell
uv sync --extra gigaam
```

По умолчанию `uv sync` ставит CPU-сборку `torch`/`torchaudio` с обычного PyPI
индекса — это целевая среда проекта (рабочий комп без GPU). Если появится GPU,
в `pyproject.toml` есть закомментированный блок с индексом `pytorch-cu128` —
раскомментируйте его и повторите `uv sync`, чтобы переключиться на CUDA-сборку.

## Запуск GUI

```powershell
uv run python -m app
```

Или через готовый скрипт — `scripts\run-gui.ps1` / `scripts\run-gui.bat` (см.
[ниже](#скрипты-запуска)).

В окне доступны:

- выпадающие списки **микрофона** и **системного звука (loopback)**;
- выбор **модели**: whisper / gigaam;
- выбор **режима**: live / после записи (batch) / файл;
- выбор **языка**: russian / english / auto (для whisper; GigaAM всегда
  русский);
- кнопки **Старт** / **Стоп**;
- кнопка **«Открыть аудиофайл…»** — импорт готового WAV/аудио в режиме
  «файл»;
- область живого транскрипта со строками вида `[мм:сс] Я: …` /
  `[мм:сс] Собеседники: …`;
- индикатор **бэклога очереди STT** — сколько сегментов ждут расшифровки (на
  CPU при активной речи очередь может отставать и догоняет в паузах);
- кнопка **«Открыть папку сеанса»** — быстрый переход к файлам записанной
  сессии.

## Запуск CLI (headless)

CLI дублирует тот же пайплайн, что и GUI, но без Qt — удобно для скриптов и
регресс-тестов.

### Список устройств

```powershell
python -m app.cli list-devices
```

Печатает микрофоны и loopback-устройства с их id — эти id нужны для `record`.

### Расшифровка готового файла

```
python -m app.cli transcribe FILE [--engine {whisper,gigaam}] [--language {russian,english,auto}] [--output-dir DIR]
```

- `--engine` — по умолчанию `whisper`.
- `--language` — по умолчанию `auto` (для whisper; GigaAM игнорирует язык —
  только русский).
- `--output-dir` — куда писать папку сеанса (по умолчанию — `session.output_dir`
  из конфига, т.е. `recordings`).

Пример:

```powershell
uv run python -m app.cli transcribe recording.wav --engine whisper --language russian
```

Файл с двумя каналами трактуется как своя сессия (`session.wav`) с полной
атрибуцией «Я»/«Собеседники»; моно-файл получает один нейтральный спикер.

### Запись с устройств

```
python -m app.cli record --mic-id ID --loopback-id ID [--mode {live,batch}] [--engine {whisper,gigaam}] [--language {russian,english,auto}] [--output-dir DIR]
```

- `--mic-id` / `--loopback-id` — обязательны, берутся из `list-devices`.
- `--mode` — `live` (расшифровка по ходу записи) или `batch` (расшифровка
  запускается по остановке); по умолчанию `batch`.
- `--engine`, `--language`, `--output-dir` — как у `transcribe`.

Пример:

```powershell
uv run python -m app.cli record --mic-id "<id микрофона>" --loopback-id "<id loopback>" --mode batch --engine gigaam
```

Команда работает до нажатия **Ctrl-C** (или EOF на stdin — удобно в
неинтерактивных пайплайнах). В режиме `batch` остановка запускает финальный
проход расшифровки перед завершением.

### Поведение вывода и кодов возврата

- `transcribe` печатает сегменты в stdout строками `[мм:сс] speaker: text`,
  статусы/бэклог/ошибки — в stderr.
- Коды возврата: `0` — успех, `1` — ошибка, `130` — прервано по Ctrl-C.

## Результат сессии

Каждая сессия сохраняется в `recordings/<timestamp>/` (или в каталог из
`--output-dir`):

- `session.wav` — стерео, 16 кГц, PCM16; левый канал = «Я» (микрофон), правый
  = «Собеседники» (loopback);
- `transcript.txt`, `transcript.json`, `transcript.srt` — один и тот же
  транскрипт в трёх форматах. `transcript.json` — список сегментов
  (start/end/speaker/text) и задуман как вход для будущего модуля
  саммаризации.

## Конфигурация

Настройки читаются из `.env` (см. `.env.example`) через `pydantic-settings`.
Не заданные ключи используют значения по умолчанию из `app/config.py`.

| Группа | Ключевые переменные | Назначение |
| --- | --- | --- |
| STT | `STT_ENGINE` (whisper/gigaam), `STT_WHISPER_MODEL_ID`, `STT_GIGAAM_MODEL_ID`, `STT_GIGAAM_REVISION` (e2e_rnnt/e2e_ctc), `STT_LANGUAGE`, `STT_DEVICE` (auto/cpu/cuda:0) | Выбор и настройка движка распознавания |
| VAD | `VAD_THRESHOLD`, `VAD_SILENCE_TIMEOUT`, `VAD_MAX_DURATION` | Чувствительность детекции речи и нарезка длинных сегментов |
| Capture | `CAPTURE_FRAME_SIZE`, `CAPTURE_TARGET_SAMPLE_RATE`, `CAPTURE_CHUNK_FRAMES`, `CAPTURE_MIC_LABEL`, `CAPTURE_LOOPBACK_LABEL`, `CAPTURE_NEUTRAL_LABEL` | Параметры захвата аудио и метки спикеров |
| Session | `SESSION_MODE` (live/batch/file), `SESSION_OUTPUT_DIR` | Режим по умолчанию и каталог сессий |

Полезные детали:

- `STT_DEVICE=auto` выбирает `cuda:0`, если доступна видеокарта, иначе `cpu` —
  на рабочей машине без GPU это всегда CPU.
- Если live на CPU не успевает за речью (turbo/large медленнее реального
  времени), задайте более лёгкую модель через `STT_WHISPER_MODEL_ID`
  (например, `openai/whisper-small`) как быстрый путь для живой расшифровки.

## Особенности и ограничения

- **Live на CPU — best-effort.** whisper-large-v3-turbo и GigaAM-v3 на CPU
  часто медленнее реального времени; очередь STT **никогда не дропает
  сегменты**, а отстаёт и догоняет в паузах (индикатор бэклога в GUI). Надёжный
  вариант — batch/file-расшифровка уже записанного `session.wav`.
- **GigaAM — только русский язык** и транскрибирует через временный WAV-файл
  (модель принимает путь к файлу, а не массив в памяти).
- **Атрибуция «Я»/«Собеседники»** гарантированно сохраняется для сессий,
  записанных этим приложением (оба канала выровнены по общим часам). При
  импорте произвольного стороннего моно-файла диаризация не выполняется —
  весь файл получает один нейтральный спикер.

## Скрипты запуска

В каталоге `scripts/` — обёртки, которые не требуют помнить команды и не
зависят от текущей директории запуска:

| Скрипт | Назначение |
| --- | --- |
| `scripts/run-gui.ps1` / `scripts/run-gui.bat` | Запуск GUI (`uv run python -m app`) |
| `scripts/run-cli.ps1` / `scripts/run-cli.bat` | Проброс аргументов в CLI (`uv run python -m app.cli ...`) |

Пример:

```powershell
.\scripts\run-cli.ps1 transcribe recording.wav --engine whisper
```
