# ADR-001: Миграция аудио-захвата с `soundcard` на `PyAudioWPatch`

## Status

Accepted (предложено к реализации)

- Дата: 2026-09-21
- Статус исходной задачи: «заменить `soundcard` на `sounddevice`» — **отклонено** (см. «Критическое открытие»)
- Решение: заменить `soundcard` на **`PyAudioWPatch`** (форк PyAudio/PortAudio с поддержкой WASAPI loopback)

---

## Context

### Задача

Windows desktop-приложение (PySide6 GUI + headless CLI, Python ≥ 3.12, `uv`, цель — CPU)
ведёт live-запись двух каналов:

- **mic** = «Я» (левый канал) — USB-микрофон **Jabra Evolve2 30 SE**;
- **loopback** = «Собеседники» (правый канал) — системный звук через **WASAPI loopback**.

Результат: `session.wav` (stereo PCM16, L=mic / R=loopback) + поканальные `mic.wav` /
`loopback.wav`; target-rate 16 kHz mono для VAD/STT.

### Почему мы уходим с `soundcard`

`soundcard==0.4.6` падает на USB Jabra Evolve2 30 SE: в `_AudioClient.__init__`
(`soundcard/_soundcard.py` / `mediafoundation.py:516`) стоит жёсткое утверждение, что
формат должен быть `WAVEFORMATEXTENSIBLE`/float32, а этот микрофон возвращает
`WAVEFORMATEX` (обычный PCM). Результат — `AssertionError` при открытии стрима.
Upstream (`bastibe/SoundCard#93`) не исправлен. То есть **текущий бэкенд не может
записать целевой микрофон**.

### Жёсткие требования (инварианты), которые обязан сохранить новый бэкенд

1. **Точная identity устройства, без fuzzy-matching.** Mic-канал НИКОГДА не должен
   открыть loopback-эндпоинт (иначе системный звук продублируется на канале «Я» —
   корневая причина старого бага дублирования транскрипции). Невозможно разрешить
   устройство → fail-fast с понятной русской ошибкой, а не «тихо записать не то».
2. **WASAPI loopback обязателен** (правый канал «Собеседники»).
3. **Partial failure**: один мёртвый канал — non-terminal (`on_status`, silence-pad);
   все каналы мертвы — terminal (`on_error`). Start-time failures — RAISE через
   `CaptureThread.resolved` + `_abort_if_all_captures_failed_at_start` (poll до 3 c).
4. **Контракт `CaptureThread`**: точные `frame_size`-фреймы в очередь, resample
   native→target, `None`-сентинел в `finally`, атрибут `.error`, событие `.resolved`.
5. **`record_mic=False`** полностью отключает mic-канал.
6. **COM/Qt**: бэкенд не должен «забирать» COM-апартмент главного потока до создания
   `QApplication` (причина, по которой `soundcard` импортируется лениво).
7. **Native rate**: захват на нативной частоте устройства, resample в `soxr` до 16 kHz.
8. **CLI** не импортирует PySide6; **GUI** не персистит выбранные устройства.

### Исходный план

Исходная формулировка задачи — «заменить `soundcard` на `sounddevice` (PortAudio)».
Это план был выбран, потому что `sounddevice` тоже стоит на PortAudio и, как
ожидалось, не имеет float32-asserion бага `soundcard`. **В ходе проверки выяснилось,
что `sounddevice` не удовлетворяет требованию №2 (loopback).** Ниже — доказательная
база и пересмотр вариантов.

---

## Критическое открытие: `sounddevice` не умеет WASAPI loopback

Это центральное открытие ADR. Оно подтверждено тремя независимыми линиями доказательств:

1. **Бандл PortAudio в `sounddevice` — это v19.7.0.**
   Wheels `sounddevice` берут PortAudio из submodule
   `spatialaudio/portaudio-binaries`. Workflow сборки этого репозитория
   (`.github/workflows/build-libs.yml`, ветка `master`) явно чекает
   `repository: PortAudio/portaudio, ref: v19.7.0`. Коммиты «Update binaries»
   (последний — 2025-11-14) генерируются именно этим workflow. `NEWS.rst`
   `sounddevice` фиксирует обновление PortAudio до 19.7.0 (0.4.2) и не упоминает
   более новых версий.

2. **PortAudio v19.7.0 не содержит кода WASAPI loopback.**
   Прямой анализ `src/hostapi/wasapi/pa_win_wasapi.c` из тега `v19.7.0`:
   в файле **нет** ни одного упоминания `loopback`/`LOOPBACK`, нет
   `PA_WASAPI_LOOPBACK_NAME_IDENTIFICATOR`, нет `FillLooopbackDeviceInfo`, нет
   `PaWasapi_IsLoopback`, нет `GetDeviceListDeviceCount`. То есть v19.7.0 не
   перечисляет loopback-эндпоинты как отдельные input-устройства и не умеет
   открывать loopback-стрим.
   Для контраста: в `master` PortAudio этот код **есть** (loopback-устройства
   дублируются в конец списка как input-устройства с суффиксом ` [Loopback]`,
   есть `PaWasapi_IsLoopback`). Функциональность добавлена позже v19.7.0
   (PortAudio PR #672, 2022).

3. **Мейнтейнеры `sounddevice` отказали в loopback.**
   - Issue `spatialaudio/python-sounddevice#281` «Feature Request: WASAPI loopback
     recording» — **открыт с 2020 года**.
   - Issue `spatialaudio/python-sounddevice#510` «Feature request for loopback of
     output device» — **закрыт (отказ)**, 2023. Автор прямо указывает, что loopback
     появился в PortAudio (PR #672) в 2022, но в `sounddevice` его не добавили.

**Вывод:** ни один актуальный релиз `sounddevice` не может записать WASAPI loopback.
Так как loopback — жёсткое требование (правый канал), **чистая миграция
`soundcard → sounddevice` невозможна** без потери функциональности.

---

## Варианты

| # | Вариант | Jabra mic (баг) | WASAPI loopback | Один COM/библ. | Точная identity | Обслуживание | Вердикт |
|---|---------|:---:|:---:|:---:|:---:|:---:|---------|
| A | **PyAudioWPatch** (оба канала) | ✅ (PortAudio) | ✅ (есть) | ✅ | ✅ (`isLoopbackDevice` + name) | ✅ (активен, wheels) | **Выбран** |
| B | `sounddevice` (mic) + `soundcard` (loopback) | ✅ | ✅ | ❌ (2 бэкенда, MTA+STA) | ⚠️ | ❌ (сложно) | Отклонён |
| C | `sounddevice` (mic) + `PyAudioWPatch` (loopback) | ✅ | ✅ | ❌ (2 PortAudio) | ⚠️ | ❌ (сложно) | Отклонён |
| D | `sounddevice` + свой PortAudio (master) | ✅ | ✅ | ⚠️ | ⚠️ | ❌ (сборка от источника) | Отклонён |
| E | `sounddevice` + виртуальный кабель (VB-Cable) | ✅ | ⚠️ (обходной) | ✅ | ⚠️ | ❌ (UX: доп. софт) | Отклонён |
| F | Оставить `soundcard`, зафоркать/залатать | ⚠️ (форк) | ✅ | ✅ | ✅ | ❌ (форк чужого кода) | Отклонён |
| G | `sounddevice`, отказаться от loopback | ✅ | ❌ | ✅ | ⚠️ | ✅ | **Неприемлемо** (ломает требование) |

### Почему отклонены гибриды (B, C)

Два аудио-бэкенда в одном процессе означают две независимые инициализации COM и два
набора PortAudio/WASAPI-объектов. `soundcard` использует MTA (`CoInitializeEx`),
`PortAudio`/`PyAudioWPatch` — STA; их смешивание на потоках повышает риск
COM-конфликтов и «зависаний» при открытии/закрытии стримов. Это противоречит
инварианту №6 и значительно усложняет разбор сбоев. Один бэкенд — проще и надёжнее.

### Почему отклонён свой PortAudio (D)

`sounddevice` использует бандл-бинарник PortAudio; подмена на master (с loopback)
требует сборки `sounddevice` из источника с кастомным PortAudio. Это ломает простоту
`uv`/wheels, тяжело поддерживается и не даёт выигрыша над вариантом A.

---

## Decision

**Заменить `soundcard` на `PyAudioWPatch`** (`pip`/`uv` пакет `PyAudioWPatch`,
модуль `import pyaudiowpatch as pyaudio`) для **обоих** каналов (mic + loopback).

### Почему `PyAudioWPatch`

- **Исправляет баг Jabra.** Под капотом — стандартный WASAPI host API PortAudio
  (не собственный код `soundcard` с float32-asserion). PortAudio корректно
  обрабатывает `WAVEFORMATEX`/PCM (`WaveToPaFormat`, переговоры формата,
  `GetMixFormat`), поэтому обычный USB-микрофон открывается без `AssertionError`.
  *(Обязательно проверить на целевой Jabra — см. чек-лист, п. 1.)*
- **Умеет WASAPI loopback.** Это главная фича форка: loopback-эндпоинты
  дублируются в конец списка устройств как **input**-устройства с суффиксом
  ` [Loopback]` (идентичная реализация, что в PortAudio master:
  `PA_WASAPI_LOOPBACK_NAME_IDENTIFICATOR "[Loopback]"`, `FillLooopbackDeviceInfo`,
  `PaWasapi_IsLoopback`).
- **Явный флаг `isLoopbackDevice`.** В dict устройства есть поле
  `isLoopbackDevice: bool` — чистый, надёжный дискриминатор mic/loopback
  (сильнее, чем сопоставление по имени). Есть готовые генераторы:
  `get_loopback_device_info_generator()`, `get_wasapi_loopback_analogue_by_index()`.
- **Один бэкенд, один COM-модель.** PortAudio инициализирует COM как STA
  (`CoInitialize(0)`), что совместимо с Qt (в отличие от MTA у `soundcard`).
- **Актуален и дистрибутируется.** Репозиторий `s0d3s/PyAudioWPatch` активно
  поддерживается (последний push 2026-01-14, ~240 звёзд), на PyPI есть готовые
  Windows-wheels для Python 3.7–3.13 (цель — 3.12). Версия 0.2.12.8; основа —
  PortAudio v19 (commit `8b6d16f`) + PyAudio v0.2.12.

### Ключевые проектные решения бэкенда

1. **Identity устройства = `name` + флаг `isLoopbackDevice`.**
   PortAudio (как и `sounddevice`) не отдаёт на Python-уровне стабильный WASAPI
   endpoint id (в C-структуре он есть — `deviceId`, — но не экспортируется).
   Поэтому ключ устройства — его **имя** плюс флаг loopback. Разрешение:
   - найти устройство с **точно** совпадающим `name` **и** ожидаемым
     `isLoopbackDevice`;
   - ровно 1 совпадение → OK; 0 → ошибка «устройство не найдено»;
     >1 → ошибка «неоднозначно» (fail-fast, **без** fuzzy/substring-фолбэков).
   - Mic-канал разрешает только `isLoopbackDevice == False`, loopback-канал —
     только `True`. Это сохраняет инвариант №1.
   - *Trade-off:* мы теряем стабильный endpoint id из `soundcard` (устройство можно
     переименовать, возможны коллизии имён). Коллизии обрабатываются fail-fast.
     Это заранее учтённый компромисс (см. MEMORY: «PortAudio не имеет WASAPI
     endpoint id, нужна name-based защита + явные ошибки»).

2. **Native rate — per-device, из `defaultSampleRate`.**
   При разрешении устройства читаем его `defaultSampleRate` (mix-format rate) и
   используем его как `native_sample_rate`. Это лучше, чем жёсткие `48000`:
   просим нативную частоту устройства → WASAPI не делает лишний SRC, а `soxr`
   доводит до 16 kHz. `DEFAULT_NATIVE_RATE = 48000` становится fallback'ом на
   случай, если `defaultSampleRate` некорректен (≤ 0).

3. **Формат захвата — `paFloat32`.**
   Поток открывается с `format=paFloat32`, `channels=<maxInputChannels устройства>`,
   `rate=<defaultSampleRate>`, `input=True`, `input_device_index=<index>`,
   `frames_per_buffer=<chunk_frames>`. `stream.read(num_frames)` возвращает `bytes`
   → `np.frombuffer(data, dtype=np.float32)` → reshape `(frames, channels)` →
   существующий `_downmix_mono` (float32). Так downstream-пайплайн
   (`StreamingResampler`, `AlignedRecorder`) не меняется — он уже работает на
   float32 mono.
   *(Точный тип возврата `stream.read()` — `bytes` или `(bytes, frames)` —
   уточнить при реализации; обработать оба варианта.)*

4. **Один общий `PyAudio()`-инстанс.**
   `Pa_Initialize()` рефсчитается и должен вызываться один раз. Вводим ленивый
   модульный синглтон `PyAudio()` (новый модуль `app/audio/backend.py`), создаваемый
   **после** `QApplication` (лениво, при первом обращении). Им пользуются и
   перечисление устройств, и оба capture-потока (каждый открывает свой стрим).
   `Pa_Terminate()` — при завершении. Это сохраняет инвариант №6.

5. **COM/Qt: ленивая инициализация сохраняется.**
   Хотя PortAudio использует STA (совместимо с Qt, в отличие от MTA у `soundcard`),
   конвенцию «не трогать аудио до готовности GUI» сохраняем: `PyAudio()`
   инстанцируется лениво, не в момент `import`. Комментарий в `main_window.py`
   (строки ~51–58) переписывается под `PyAudioWPatch`.

6. **Контракт `CaptureThread` сохраняется.**
   Конструктор, очередь, `None`-сентинел в `finally`, `.error`, `.resolved`,
   partial/total policy в `Session` — без изменений. Меняется только «внутренность»:
   вместо `mic.recorder(...)` / `rec.record(...)` — `p.open(...)` / `stream.read(...)`.
   `StreamingResampler` создаётся в `run()` **после** разрешения устройства
   (т.к. native rate теперь известен только после разрешения).

---

## Consequences

### Что становится проще / лучше

- Целевой микрофон Jabra записывается (баг `soundcard` уходит).
- Loopback остаётся, теперь на том же бэкенде, что и mic (один COM-модель).
- Явный флаг `isLoopbackDevice` делает mic/loopback-дискриминацию надёжнее.
- Per-device native rate убирает лишний драйверный SRC.
- `PyAudioWPatch` даёт контекст-менеджеры и генераторы устройств — чище перечисление.

### Что становится сложнее / хуже

- **Identity по имени** (а не по стабильному endpoint id): коллизии имён →
  fail-fast; переименование устройства в Windows «сдвигает» выбор. Митигируется
  явными ошибками и тем, что GUI не персистит выбор.
- **PyAudio-стиль API**: `stream.read()` возвращает `bytes` (нужен `np.frombuffer`),
  а не готовый numpy-массив, как в `sounddevice`. Небольшой, но реальный overhead
  в `capture.py`.
- **Один мейнтейнер** (`s0d3s`) — bus-factor. Форк небольшой и стабильный; риск
  умеренный. Митигируется тем, что бэкенд-слой изолирован (легко подменить).
- **Нужна ручная верификация на Windows** (CI без железа не покрывает захват).

### Что НЕ меняется

- `Session`, `AlignedRecorder`, `StreamingResampler`, VAD/STT, запись WAV,
  partial/total failure policy, `record_mic=False`, CLI-интерфейс (кроме текста
  help про «id»), GUI-комбо (интерфейс `devices` сохраняется).

---

## Ответы на открытые вопросы

1. **Identity / коллизии / исчезновение устройства.**
   Ключ = `name` + `isLoopbackDevice`. Точное совпадение; 0 → «не найдено»,
   >1 → «неоднозначно» (fail-fast, без fuzzy). Исчезновение устройства во время
   записи → `stream.read()` бросает `OSError` (PortAudio-код) → поток ставит `.error`
   и сентинел → `Session` применяет partial/total policy. См. п. 4.

2. **Loopback API.**
   `PyAudioWPatch`: loopback-устройства — это input-устройства с
   `isLoopbackDevice == True` и именем `<имя_выхода> [Loopback]`. Перечисление —
   `get_loopback_device_info_generator()` (или фильтр по флагу в
   `get_device_info_generator()`). Захват — обычный input-стрим на этом
   `input_device_index`.

3. **Per-device native rate.**
   `native_sample_rate = device['defaultSampleRate']` (fallback `48000`, если ≤ 0).
   `StreamingResampler(native, target=16000)` создаётся в `run()` после разрешения.
   `DEFAULT_NATIVE_RATE` остаётся только как fallback-константа.

4. **Маппинг ошибок и hot-unplug.**
   PortAudio/PyAudio бросают `OSError` с кодом (`paUnanticipatedHostError`,
   `paDeviceUnavailable`, `paInvalidSampleRate`, `paInputOverflowed`, …).
   `CaptureThread.run()` ловит `Exception` → `.error` + сентинел (как сейчас).
   Добавляем маппинг PortAudio-кодов → короткие русские фразы (например,
   `paDeviceUnavailable` → «устройство отключено», `paInputOverflowed` →
   «переполнение буфера ввода»). Hot-unplug = `OSError` на `read()` → та же цепочка.

5. **CI без железа vs ручная проверка Windows.**
   Аудио-захват **не покрывается CI** (нет Windows-аудио в CI). Верификация —
   ручная на Windows с Jabra (чек-лист ниже). Опционально: `python -m app.audio.devices`
   и `python -m app.cli list-devices` как ручные smoke-инструменты.

6. **Rollback.**
   Миграция — замена бэкенда, локализованная в `app/audio/capture.py`,
   `app/audio/devices.py`, `app/audio/backend.py` (новый), `pyproject.toml`,
   `uv.lock` + косметика в GUI/CLI. Контракт `CaptureThread` и `devices`
   сохраняется, поэтому **rollback = `git revert`** одного коммита (возврат
   `soundcard`). Никаких мигрант-схем/необратимых изменений.

7. **File-level план.** — раздел ниже.

8. **Blocksize / chunking.**
   Без изменений: `frames_per_buffer = chunk_frames` (1024), `stream.read(1024)`,
   `frame_size = 512` (VAD), pending-паттерн точных фреймов — как сейчас.

---

## File-level план

> Реализация не входит в данный ADR; ниже — карта правок по файлам.

| Файл | Что меняется |
|------|--------------|
| `app/audio/backend.py` **(новый)** | Ленивый синглтон `PyAudio()` (создаётся после `QApplication`), `get_backend()`, `shutdown()`; обёртка над `pa.initialize()/terminate()`. Точка единственной инициализации COM/PortAudio. |
| `app/audio/devices.py` | `import pyaudiowpatch as pyaudio` вместо `soundcard`. `list_microphones()` — WASAPI input-устройства с `maxInputChannels>0` и `isLoopbackDevice==False`; `list_loopbacks()` — с `isLoopbackDevice==True`. Возврат `(name, id)`, где `id` = `name` (identity по имени). `get_microphone(name, *, expect_loopback)` — точное разрешение по `name`+флагу (fail-fast при 0/>1). `_main()` — печать списков. |
| `app/audio/capture.py` | `import pyaudiowpatch` вместо `soundcard`. `_resolve_device(name, expect_loopback)` — разрешение через `backend` по `name`+`isLoopbackDevice` (fail-fast). `CaptureThread.run()` — `p.open(format=paFloat32, channels=maxInputChannels, rate=defaultSampleRate, input=True, input_device_index=idx, frames_per_buffer=chunk)` → цикл `stream.read(chunk)` → `np.frombuffer` → `_downmix_mono` → resample → `_emit_frames`; flush + сентинел в `finally`. `StreamingResampler` создаётся в `run()` (native rate из `defaultSampleRate`). Маппинг PortAudio-ошибок → русские сообщения. `DEFAULT_NATIVE_RATE` — только fallback. |
| `app/pipeline/session.py` | `_start_captures`: убрать жёсткий `native_sample_rate=DEFAULT_NATIVE_RATE` (теперь per-device внутри `CaptureThread`); импорт `DEFAULT_NATIVE_RATE` убрать/оставить как fallback. Остальное (partial/total, `record_mic`, `_abort_if_all_captures_failed_at_start`) — без изменений. |
| `app/gui/main_window.py` | Комментарий про COM (строки ~51–58) переписать под `PyAudioWPatch` (STA, ленивая инстанциация). `_populate_devices` — ленивый импорт `app.audio.devices` (интерфейс тот же). Комбо: `addItem(name, name)` (identity = имя). |
| `app/cli.py` | `_cmd_list_devices` — ленивый импорт `app.audio.devices` (интерфейс тот же). Help для `--mic-id` / `--loopback-id`: «id» теперь = **имя** устройства (из `list-devices`). |
| `pyproject.toml` | Зависимость: убрать `soundcard`, добавить `PyAudioWPatch`. |
| `uv.lock` | Обновить (`uv lock`). |
| `README.md` | Заменить упоминания `soundcard` на `PyAudioWPatch`; пометить, что loopback — через input-устройства `[Loopback]`. |
| `tests/` | Обновить/добавить тесты, имитирующие `pyaudiowpatch` (mock `PyAudio`/`Stream`): разрешение по имени+флагу, fail-fast при коллизии, downmix float32, сентинел/`.error`. |

**Не меняется:** `app/audio/resample.py`, `app/audio/recorder.py`, `app/pipeline/*`
(кроме мелкой правки `session.py`), VAD/STT, `app/config.py` (значения те же).

---

## Чек-лист верификации (ручной, на Windows)

> CI аудио-захват не покрывает. Выполнить на целевой машине с Jabra Evolve2 30 SE.

1. **[Критично] Jabra mic.** `python -m app.cli list-devices` → Jabra виден как
   микрофон. Запись с `--mic-id "<имя Jabra>"` → **без `AssertionError`**, в
   `mic.wav` есть голос. *(Главный риск: убедиться, что PortAudio открывает
   `WAVEFORMATEX`/PCM без float32-asserion.)*
2. **Loopback.** В `list-devices` видны input-устройства с суффиксом ` [Loopback]`
   и `isLoopbackDevice=True`. Запись loopback → в `loopback.wav` слышен системный звук.
3. **Стерео-сшивка.** `session.wav` — 2 канала: L=mic, R=loopback, синхронно.
4. **Mic не берёт loopback.** При отключённом физическом микрофоне mic-канал НЕ
   «падает» на loopback (fail-fast «не найдено», а не тихий захват системного звука).
5. **Partial failure.** Отключить один канал во время записи → non-terminal
   (`on_status`), сессия дописывается по живому каналу.
6. **Total failure / start-time.** Оба устройства недоступны → `start()` RAISE
   (быстро, ≤ 3 c), GUI/CLI сбрасываются.
7. **Hot-unplug.** Вытащить Jabra во время записи → канал умирает non-terminal,
   сентинел выпущен, `stop()` не висит (bounded join).
8. **`record_mic=False` / `--no-mic`.** Mic-канал не открывается, левый канал
   silence-pad, транскрипция только loopback.
9. **COM/Qt.** GUI стартует без COM-конфликтов; `PyAudio()` инициализируется
   лениво (после `QApplication`); закрытие приложения чистое (нет зависаний).
10. **CLI без PySide6.** `python -m app.cli ...` не импортирует PySide6.
11. **Native rate.** Для каждого устройства `native = defaultSampleRate`;
    resample до 16 kHz корректен (нет «чипмунка»/замедления).

---

## Риски и rollback

### Риски

- **R1 (высокий):** `PyAudioWPatch` может вести себя на Jabra иначе, чем ожидается
  (формат/частота). *Митигция:* п. 1 чек-листа — первый шаг реализации; при сбое —
  рассмотреть `explicit` формат/частоту или `is_format_supported()` перед открытием.
- **R2 (средний):** Identity по имени — коллизии/переименование. *Митигция:*
  fail-fast, явные ошибки; GUI не персистит выбор.
- **R3 (средний):** COM-поточность при общем `PyAudio()`-синглтоне, используемом из
  capture-потоков. *Митигция:* п. 9 чек-листа; при проблемах — по-потоковый
  `PyAudio()` (Pa_Initialize рефсчитается).
- **R4 (низкий):** Bus-factor (один мейнтейнер форка). *Митигция:* бэкенд-слой
  изолирован (`backend.py`/`capture.py`/`devices.py`) — подменяем на другой
  PortAudio-бэкенд без изменения пайплайна.
- **R5 (низкий):** Тип возврата `stream.read()` (`bytes` vs tuple). *Митигция:*
  уточнить при реализации, обработать оба варианта.

### Rollback

Миграция локализована и контракт `CaptureThread`/`devices` сохранён →
**rollback = `git revert <commit>`** (возврат `soundcard==0.4.6` в
`pyproject.toml`/`uv.lock` и исходных `capture.py`/`devices.py`). Необратимых
изменений данных/схем нет. До успешного прохода чек-листа (п. 1–3) рекомендуется
не удалять `soundcard` из истории, чтобы откат был мгновенным.

---

## Доказательная база / источники

- `portaudio-binaries/.github/workflows/build-libs.yml` (master): `ref: v19.7.0`.
- `PortAudio/PortAudio` `v19.7.0` `src/hostapi/wasapi/pa_win_wasapi.c` — нет loopback.
- `PortAudio/PortAudio` `master` `src/hostapi/wasapi/pa_win_wasapi.c` — есть loopback
  (`PA_WASAPI_LOOPBACK_NAME_IDENTIFICATOR "[Loopback]"`, `FillLooopbackDeviceInfo`,
  `PaWasapi_IsLoopback`); loopback добавлен в PR #672 (2022).
- `spatialaudio/python-sounddevice` issues: #281 (open, loopback request),
  #510 (closed, отказ); `NEWS.rst` (PortAudio 19.7.0 в 0.4.2).
- `s0d3s/PyAudioWPatch`: `README.md`, `src/pyaudiowpatch/__init__.py`
  (`isLoopbackDevice`, `get_loopback_device_info_generator`, `get_wasapi_loopback_*`),
  `portaudio_v19/src/hostapi/wasapi/pa_win_wasapi.c` (loopback-реализация),
  метаданные репозитория (push 2026-01-14, wheels, Python 3.7–3.13).
- Локальный код: `app/audio/capture.py`, `app/audio/devices.py`,
  `app/audio/resample.py`, `app/pipeline/session.py`, `app/config.py`,
  `app/gui/main_window.py`, `app/cli.py`, `MEMORY.md`.
