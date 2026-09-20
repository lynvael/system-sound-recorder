# Project Memory — Live Recorder

Facts that help future work and are NOT obvious from the code alone.

## What this is
Windows desktop app (single Python process) for live meeting transcription: captures
mic ("Я", left channel) + system-audio loopback ("Собеседники", right channel) on
separate channels, Silero VAD per channel, GigaAM-v3 STT (Russian-only; the sole
engine — whisper was removed),
PySide6 GUI + headless CLI (`app/cli.py`). Design spec: `kind-hatching-allen.md` (RU).
Originally reused near-verbatim from sibling project `C:\develop\Chisa` (STT + Silero
segmenter, `app/common/{config,log}.py`).

## Architecture decisions & why
- **Disk is the single source of truth.** One stereo WAV `session.wav` (16 kHz, PCM16,
  L=mic/R=loopback) is streamed to disk via soundfile, never buffered whole in RAM
  (meetings are hours long). batch/file modes re-read it from disk in blocks.
- **`AlignedRecorder` aligns both channels to a common monotonic clock** (`expected =
  round(elapsed*16000)`, silence-pad short channel / silent-trim ahead channel). This
  fixes loopback silence gaps AND two-device clock drift. VAD sample indices therefore
  equal positions in `session.wav`, so segment timestamps line up across channels.
- **Single FIFO STT queue under one `threading.Lock`**, one worker; queue **never drops**
  segments (correctness > latency for a recorder). Backlog depth is surfaced to the GUI.
- **`target_sample_rate` (16000) in config is the single source of truth** — VAD and the
  soxr resampler read it from config; do NOT hardcode 16000 elsewhere or channels desync.
- **CPU is the target env**; live is best-effort (GigaAM-v3 on CPU is slower than
  real-time). file-mode is the reliable + primary regression path.
- **Mic noise-suppression (NS) ADR exists but is DEPRIORITIZED / not started** (Luno,
  2026-09-20). Recommendation: RNNoise (`rnnnoise` pkg) in `CaptureThread` at 48 kHz,
  before soxr, mic channel only; NS-processed mic written into `session.wav` (keeps
  live==batch==re-read consistent); config `CAPTURE_NOISE_SUPPRESSION` default on; GUI
  checkbox; CLI `--no-noise-suppression`. Alternatives weighed: WebRTC NS (permissive
  licence, lower quality), `noisereduce` (offline-only → live≠batch), OS-level (not
  controllable). IMPORTANT: NS solves BACKGROUND noise (fan/keyboard) — it does NOT fix
  the «Я»/«Собеседники» duplication, which is an ELECTRICAL bleed (see Gotchas). Do not
  implement NS expecting it to fix the duplication. Open ADR questions were: default on?,
  GPL-3.0 (RNNoise) acceptable?, losing the raw mic acceptable? — all deferred along with
  the bleed decision.

## Conventions / contracts (don't break)
- **`on_error` callback = TERMINAL only** (session is ending). Non-terminal problems
  (e.g. a single segment failing to transcribe) go to `on_status`, NOT `on_error`.
  History: conflating the two orphaned live sessions — cost two review rounds to fix.
- **GUI worker resets `self._session = None` on exactly three paths:** `start_session`'s
  `except` (start-time raise), `_emit_finished` (on_finished), `stop_session` finally.
  `_emit_error` must NOT reset it.
- **Session start-time failures RAISE** (not silent return) so the GUI worker's `except`
  resets — applies to `_start_live`, `_start_recording_only`, and `_start_file`.
- Mid-run recorder-writer death drives a real teardown via a daemon thread calling
  `stop()` (can't call stop() inline — it joins the writer thread → self-join deadlock).
- CLI path must NOT import PySide6 (keep Qt imports lazy / GUI-only).
- **Per-channel WAVs are DERIVED from the aligned stereo, not a replacement for it.**
  `session.wav` (stereo) stays the single source of truth; `AlignedRecorder` additionally
  streams `mic.wav` (L/"Я") and `loopback.wav` (R/"Собеседники") from the SAME popped
  L/R blocks in `_write_block`, so all three stay sample-for-sample aligned. Transcript
  stays a single MERGED file. Cost: ~1 extra audio copy on disk (~230 MB/hr). Do NOT
  switch to writing only the two mono files — it breaks the source-of-truth invariant,
  the batch `_run_disk_pass` re-read (reads ch0/ch1 of session.wav), and stereo import.
  Filenames are ASCII on purpose (Cyrillic-filename risk); mapping documented in code.
- **File mode = ALWAYS a single speaker (`neutral_label`, "Speaker") — no diarization.**
  `_start_file` unconditionally spawns `_run_disk_pass(import_path, stereo=False)`, so any
  external import (mono or N-channel) downmixes to one mono track via `iter_blocks(channel=
  None)` and one VAD → one "Speaker". Deliberate product decision: do NOT attempt L/R
  speaker attribution on foreign files (an ordinary stereo recording has ~identical L/R and
  dual VAD transcribed every phrase TWICE, once per label). Dual "Я"/"Собеседники"
  attribution is reserved for BATCH/live re-reading our own `session.wav` (genuine mic vs
  loopback), which still calls `_run_disk_pass(stereo=True)`. A rejected earlier approach
  (content-based channel-correlation detection) was intentionally removed for simplicity.
- **Systemic all-fail surfaces as terminal `on_error` at session end.** A SINGLE segment
  failing stays non-terminal (`on_status`) per the contract above; but when
  `_attempted > 0 and _failed == _attempted` (every attempted segment failed) the session
  emits a terminal `on_error` just before `_finish()` — defense-in-depth against the
  "empty transcript, no error" trap (e.g. GigaAM's over-30s swallow, below). Guarded by `_attempted > 0` so a
  zero-speech recording (VAD found nothing) never false-positives.

## Gotchas / traps
- **Don't import `soundcard` (directly or transitively) at module scope in GUI code
  that loads before `QApplication` exists** — it breaks launch with
  `QWindowsContext: OleInitialize() failed: COM error 0x80010106 (RPC_E_CHANGED_MODE)`.
  `soundcard`'s import runs `CoInitializeEx(MTA)` on the main thread; if that happens
  before Qt creates `QApplication` (which does `OleInitialize` → STA), Qt loses the
  apartment race. Only `app/audio/devices.py` and `app/audio/capture.py` import
  soundcard, but BOTH are reachable from `app/gui/main_window.py` at import time:
  directly (`devices`) and transitively via `app.gui.worker → app.pipeline.session →
  app.audio.capture`. Fix: those imports in main_window are LAZY (inside
  `_populate_devices` / `__init__` / `_on_start_clicked`), so they run only after
  `main()` has constructed `QApplication`. soundcard itself tolerates STA (it catches
  `RPC_E_CHANGED_MODE` and falls back), so no backend/threading change is needed — just
  keep those imports off module scope. See the NOTE block near the top of main_window.py.
- **Native Windows `QFileDialog` freezes the GUI on file-open.** `getOpenFileName`'s
  native Win32 dialog synchronously enumerates shell namespace extensions
  (OneDrive/cloud overlays, network/mapped drives in Quick access/Recent) on the UI
  thread while showing — can hang for many seconds/indefinitely. `_on_open_file_clicked`
  passes `options=QFileDialog.Option.DontUseNativeDialog` to force Qt's own dialog.
- **Never name an instance attribute `_stop` (or other `threading.Thread` internals)
  on a `threading.Thread` SUBCLASS.** `CaptureThread(threading.Thread)` originally set
  `self._stop = threading.Event()`, which shadowed `Thread._stop()` — an internal method
  `Thread.join()` calls via `_wait_for_tstate_lock`. Result: every session stop crashed
  with `TypeError: 'Event' object is not callable` on `ct.join()`. Fixed by renaming to
  `self._stop_event`. Avoid these names on Thread subclasses: `_stop`, `_started`,
  `_tstate_lock`, `_target`, `_args`, `_kwargs`, `_name`, `_ident`, `_daemonic`,
  `_is_stopped`. (AlignedRecorder's `self._stop` Event is fine — it is NOT a Thread
  subclass, it runs a separate `threading.Thread(target=self._run)`.) This class of bug
  is invisible to compile/import checks — only shows at runtime when join() is called.
- **`soundcard.get_microphone(id, include_loopback=True)` does FUZZY id matching** and
  will silently fall back to a LOOPBACK endpoint when a mic id no longer resolves exactly
  (e.g. mic physically unplugged). `_match_device` tries exact-id → name-substring → regex
  fuzzy; `all_microphones(include_loopback=True)` lists loopbacks first, so a disconnected
  mic resolved to a system-audio endpoint. Result: the mic ("Я") channel captured the SAME
  system audio as the loopback ("Собеседники") channel → EVERY utterance transcribed twice
  (duplicated live replicas). Fix (in `app/audio/capture.py`): `CaptureThread` takes a
  required `expect_loopback` bool; `_resolve_device` passes it as `include_loopback` (mic=
  False so a loopback can't even be a candidate) AND validates the resolved device — exact
  `device.id == requested id` (rejects any fuzzy fallback) and `device.isloopback ==
  expect_loopback` — raising a Russian error. `_start_captures` passes
  `expect_loopback=[False, True]` for `[mic, loopback]`. Do NOT "solve" this class of
  duplication with content/correlation dedup (deliberately removed — see file-mode note
  above); fix device resolution at the source.
- **Capture-device failure is PARTIAL-tolerant** (see `Session._handle_capture_errors`,
  `_abort_if_all_captures_failed_at_start`). A capture thread that can't resolve its device
  sets `.error` and exits WITHOUT setting `.resolved`, but still emits
  its `None` sentinel in `finally`, so feeders/segmenters drain and AlignedRecorder
  silence-pads the dead channel and the SURVIVING channel records cleanly. Policy: a SINGLE
  dead channel is NON-TERMINAL — a Russian on_status warning names the channel («Я» mic /
  «Собеседники» loopback) and the session finishes on the other channel's transcript. Only
  when EVERY capture channel failed is it terminal on_error (per the on_error=TERMINAL
  contract). All-dead is normally caught FAST at start: after `ct.start()`,
  `_abort_if_all_captures_failed_at_start` polls up to `_START_RESOLVE_TIMEOUT` (3s) — any
  channel setting `CaptureThread.resolved` short-circuits the wait (normal start is instant),
  but if every thread already errored it RAISES (start-time-failures-RAISE contract → GUI
  worker resets) instead of recording hours of silence. `_handle_capture_errors` (called in
  `_stop_live`/`_stop_batch`) is the stop-time safety net for channels that resolved OK at
  start but died mid-run.
- **The user's live "system-audio duplicated" symptom is an ELECTRICAL bleed
  (codec crosstalk / ground loop) into the mic channel — NOT acoustic echo, NOT the
  device bug.** The «Я» (mic) channel carries the system/stream audio via a path inside
  the PC's audio hardware (Realtek codec output → mic preamp/ADC) that the physical mic
  mute does NOT cut. Evidence (2026-09-20, cross-correlation of mic.wav vs loopback.wav
  + a controlled speaker test):
  - mic channel ≈ loopback delayed ~30 ms (stable +28…+37 ms, drifting to +48 ms late in
    long sessions), ~17–20 dB lower level (≈3–10% of the output signal), per-window corr
    0.29–0.98; mic spectral tilt ≈ loopback (an acoustic path via the monitor speakers
    would be markedly darker in the highs).
  - DECISIVE TEST: with the monitor speakers physically at volume 0, the stream audio in
    «Я» stayed at the SAME level → the path is electrical (inside the codec / via shared
    ground), not through the air. An acoustic path would have vanished.
  - The physical mic mute mutes the user's own voice (the capsule) but NOT the bleed (it
    enters after the mute point, at the preamp/ADC). Session 225815 (mic physically
    muted): user's voice absent except a "Раз-раз" mic-test at 00:15 (before the mute),
    stream audio present all session. Session 230147 (mic off via the app checkbox): «Я»
    fully silent (−180 dB).
  - The fuzzy-fallback device bug is NOT the cause (the «Я» channel is the real Realtek
    mic endpoint, validated by exact-id + isloopback check).
  Consequences: (a) the physical mic toggle is USELESS for this problem; (b) headphones
  will NOT help (the bleed is in the PC, not the room); (c) NOISE SUPPRESSION will NOT
  help (the bleed is speech, not noise) — the RNNoise NS ADR (see below) solves a
  different problem (background fan/keyboard noise) and is NOT the fix for the
  duplication.
  Working workaround (implemented 2026-09-20): GUI checkbox «Записывать микрофон»
  (default on) → `SessionParams.record_mic` → `Session(record_mic=...)`; CLI
  `record --no-mic` (then `--mic-id` optional). When off the mic capture+feeder are not
  started; AlignedRecorder silence-pads the left channel → loopback-only transcript,
  mic.wav silent. Verified (session 230147).
  **DECISION PENDING (user deferred, 2026-09-20): is the user's own voice needed in «Я»?**
  Options on the table:
  1. App checkbox (mic off) — no code, but loses own voice.
  2. Try a different OUTPUT device (HDMI↔3.5 mm) — free test; if the bleed disappears on
     another output, the coupling is on a specific line.
  3. USB microphone — own ADC/ground, usually eliminates the bleed (purchase).
  4. AEC (real-time, e.g. WebRTC AEC3) — software, keeps mic on + own voice, cancels the
     loopback-correlated component at capture.
  5. Echo-gating (post-hoc) — software, cheap, retroactive on recorded sessions: drop «Я»
     VAD segments highly correlated with loopback in the same window; user's own
     (uncorrelated) speech survives.
  Diagnostic method that worked (reusable): per-30s-window FFT cross-correlation of
  mic.wav vs loopback.wav (peak lag + normalized corr), per-window spectral tilt
  (hi 4–8k / lo 0.3–1k), and Silero VAD speech-fraction on the residual (mic − best-lag
  scaled loopback copy) to detect the user's own voice.
- **`silero-vad==6.2.2` imports `onnxruntime` at package import but doesn't declare it** —
  `onnxruntime` is in base deps for this reason. Don't remove it.
- **Native capture rate is assumed 48000 Hz** (`soundcard` needs an explicit samplerate
  and has no reliable native-rate API); soxr resamples to 16 kHz. Revisit if a device's
  native rate differs — would need plumbing through config.
- **Dep pins are chosen for GigaAM-v3 compat:** `torch==2.8.0`, `torchaudio==2.8.0`,
  `transformers==4.57.1`. GigaAM extras (`[gigaam]`): pyannote-audio, hydra-core,
  omegaconf, sentencepiece, torchcodec. GigaAM is Russian-only and transcribes via a
  temp WAV (model takes a file path, not numpy).
- **GigaAM `transcribe()` rejects clips over ~30s** ("Too long wav file, use
  'transcribe_longform' method."). VAD `max_duration` is 60s, so max-length segments
  overflowed and were silently swallowed by the worker's non-terminal per-segment except
  (recorded fine, phrases missing, no error). We do NOT use `transcribe_longform` (product
  decision). Instead `GigaAMEngine.transcribe()` routes any clip ≥ `_MAX_TRANSCRIBE_SECONDS`
  (24s) to `_transcribe_sliced`: it re-feeds the audio through a FRESH `ContinuousSegmenter`
  (reused from `app/vad/segmenter.py`) capped at `_SPLIT_MAX_DURATION` (20s, NOT the config's
  60s), so each sub-chunk is ≤ ~20.3s (max_frames + PREROLL, well under 30s at any Silero
  rate), short-form `transcribe()`s each, and joins texts chronologically. VAD settings come
  via `config.vad` threaded through the factory (engine `__init__(stt, sample_rate, vad)`),
  Silero loads LAZILY on first long segment. If sub-VAD yields ZERO chunks for known-speech
  audio, `_fixed_windows` hard-splits into ≤20s windows so a long segment is never dropped.
  Sub-chunk transcribe errors propagate (non-terminal).
- `transformers==4.57.1` pipeline uses the kwarg `dtype` (NOT `torch_dtype`).
- Drift-trim bound in recorder is `_DRIFT_TRIM_SECONDS = 1.0` — tune after a 30–60 min
  sync test if drift becomes audible.
- **`Session._cancel` is set per-mode in `stop()`, NOT unconditionally.** Its ONLY purpose
  is aborting an in-progress FILE-mode import (the `_disk_thread` running `_run_disk_pass`,
  whose block loop breaks on `if self._cancel.is_set()`). It is NOT used to stop captures
  (those stop via `ct.stop()`), so live never checks it. Trap: for BATCH, `stop()` is the
  trigger that RUNS transcription (`stop → _stop_batch → _run_disk_pass` synchronously) —
  so if `stop()` sets `_cancel` before that, the disk-pass loop breaks on block one and the
  transcript comes out EMPTY (recorded fine, nothing transcribed, no error). That was a
  real bug. Keep `_cancel.set()` in the `file` branch of `stop()` only.

## Not yet verified (needs real hardware — none in the build env)
No mic / GPU / model downloads were available during implementation. Static correctness,
clean imports, `uv sync`, and `list-devices` were verified. Still needs the user's manual
run: actual capture, VAD segmentation on real audio, GigaAM transcription,
GUI live/batch/file runs, CPU RTF measurement, hour-long RAM-stability + channel-sync
tests. See "Верификация" in `kind-hatching-allen.md`.

## Session naming (`app/pipeline/session.py`, GUI)
- Sessions can be **optionally named** by the user. GUI: a `QLineEdit` →
  `SessionParams.session_name` → `Session(name=...)`. Dir is
  `<YYYYmmdd_HHMMSS>_<sanitized name>` (timestamp PREFIX keeps the picker's
  name-descending sort chronological AND collision-safe across seconds; empty/
  None/unusable name → timestamp only, i.e. the original behavior).
- `_sanitize_session_name` (module fn in session.py): **preserves Cyrillic/
  Unicode letters** (do NOT ASCII-fold — users name sessions in Russian), strips
  Windows-unsafe punctuation `< > : " / \ | ? *`, drops all `Cc`/`Cf` chars via
  `unicodedata.category` (C0/C1/DEL controls + zero-width/bidi like U+200B/U+202E),
  collapses whitespace→`_`, trims leading/trailing `._ `, caps to 80, and suffixes
  the STEM (before first dot) when it's a reserved device name (CON/PRN/AUX/NUL/
  COM1-9/LPT1-9). Audio-file basenames inside the dir stay ASCII (session.wav/
  mic.wav/loopback.wav) — only the DIRECTORY component is user-controlled.
- GUI display is **mode-aware**: File mode shows "обработка файла"/"Обработка файла"
  (it transcribes an existing file, nothing is recorded), live/batch show "запись"/
  "Идёт запись". Derived from `_active_session_params.mode` via
  `_active_session_is_file_mode()`; `self._recording` keeps its "session active"
  semantics unchanged (gates buttons/timer/delete-guard/teardown).

## Summarization (`app/summarize/`)
- **On-demand, decoupled from Session lifecycle.** `run_summarization()` (the sole
  public entrypoint, `app/summarize/pipeline.py`) is NOT wired into recording
  start/stop — it just reads a finished session's `transcript.json` from disk and
  writes `summary.docx` next to it. Works for any past session under `recordings/`,
  not just the one just recorded. Called from the GUI's right-hand "Сеанс" panel
  («Саммаризация» button), off the UI thread.
- **Final protocol uses the LLM's NATIVE structured output**, not Markdown
  scraping. `schema.py` defines pydantic `MeetingSummary`{tldr, key_decisions[],
  tasks[{text, assignee?}], topics[], open_questions[]} + `Task`.
  `client.chat_structured()` sends `response_format={"type":"json_schema",
  "json_schema":{...,"strict":True}}` and validates the reply into the model; the
  free-text MAP pass still uses plain `chat()`. `docx_export.write_docx` renders
  from the OBJECT (no more `_parse_sections`). Russian section HEADINGS now live in
  docx_export as the single source of truth.
- **`strict_json_schema` (schema.py) MUST strip `default` from every node** —
  genuine OpenAI strict mode 400s on `'default'` (its own SDK strips None defaults).
  pydantic emits `"default": null` for `assignee: str|None = None`, so `_strictify`
  pops `default` and sets `additionalProperties:false` + `required=all props` on
  every object node INCLUDING nested `Task` under `$defs`. Nullable `assignee` stays
  `anyOf:[string,null]` and stays REQUIRED (OpenAI's nullable pattern). Lenient
  local servers (llama.cpp/some vLLM) ignore a stray `default`, so this bug only
  shows against real OpenAI — test there, not just locally.
- **Map-reduce over an OpenAI-compatible endpoint**, configured entirely via
  `LLM_*` env vars (`config.llm`: `url`, `api_key`, `model`, `chunk_chars`,
  `chunk_overlap`, `temperature`, `max_tokens`, `request_timeout`). Consecutive
  same-speaker segments are merged first (`merge_consecutive` from
  `app/pipeline/transcript.py`, reused — not duplicated), then the transcript is
  character-chunked with overlap; each chunk is mapped, and reduce is skipped
  entirely when there's exactly one chunk.
- **New base deps**: `openai` (LLM client) and `python-docx` (`.docx` export) —
  both in `pyproject.toml` base `dependencies`, so plain `uv sync` installs them
  (no extra needed).
- **All failures wrapped in `SummarizationError`** (Russian-language message,
  defined in `app/summarize/pipeline.py`) — missing/empty/malformed transcript,
  LLM/transport errors, and docx save failures (e.g. file open in Word on
  Windows) all funnel through it. It is always non-terminal in the GUI: a failed
  summarization never touches or deletes the existing `transcript.*` files, so
  retrying is always safe.

## Fixed footguns
- **`pyproject.toml`'s `readme =` pointed at a nonexistent `kind-hatching-allen.md`**,
  which broke `uv sync` (hatchling build fails without a valid readme file). Fixed
  to `readme = "README.md"`. Don't reintroduce a dangling `readme` path — verify
  the target file actually exists before pointing at it.

## Env
Windows, Python >=3.12, package manager **uv**. `uv sync` (base) / `uv sync --extra gigaam`.
CPU torch via default PyPI index; commented `pytorch-cu128` index in pyproject.toml is the
GPU opt-in. Run: `python -m app` (GUI), `python -m app.cli {list-devices,transcribe,record}`
(CLI), or `scripts/run-{gui,cli}.{ps1,bat}`.
