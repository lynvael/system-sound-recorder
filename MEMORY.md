# Project Memory — Live Recorder

Facts that help future work and are NOT obvious from the code alone.

## What this is
Windows desktop app (single Python process) for live meeting transcription: captures
mic ("Я", left channel) + system-audio loopback ("Собеседники", right channel) on
separate channels, Silero VAD per channel, switchable STT (whisper / GigaAM-v3),
PySide6 GUI + headless CLI (`app/cli.py`). Design spec: `kind-hatching-allen.md` (RU).
Reused near-verbatim from sibling project `C:\develop\Chisa` (whisper STT + Silero
segmenter in `app/stt/engine.py`, `app/common/{config,log}.py`).

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
- **CPU is the target env**; live is best-effort (turbo/large slower than real-time).
  file-mode is the reliable + primary regression path.

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
- Language `"auto"` is mapped to `None` before constructing Session (both GUI and CLI).
- CLI path must NOT import PySide6 (keep Qt imports lazy / GUI-only).
- **Per-channel WAVs are DERIVED from the aligned stereo, not a replacement for it.**
  `session.wav` (stereo) stays the single source of truth; `AlignedRecorder` additionally
  streams `mic.wav` (L/"Я") and `loopback.wav` (R/"Собеседники") from the SAME popped
  L/R blocks in `_write_block`, so all three stay sample-for-sample aligned. Transcript
  stays a single MERGED file. Cost: ~1 extra audio copy on disk (~230 MB/hr). Do NOT
  switch to writing only the two mono files — it breaks the source-of-truth invariant,
  the batch `_run_disk_pass` re-read (reads ch0/ch1 of session.wav), and stereo import.
  Filenames are ASCII on purpose (Cyrillic-filename risk); mapping documented in code.
- **Systemic all-fail surfaces as terminal `on_error` at session end.** A SINGLE segment
  failing stays non-terminal (`on_status`) per the contract above; but when
  `_attempted > 0 and _failed == _attempted` (every attempted segment failed) the session
  emits a terminal `on_error` just before `_finish()` — defense-in-depth against the
  torchcodec-style "empty transcript, no error" trap. Guarded by `_attempted > 0` so a
  zero-speech recording (VAD found nothing) never false-positives.

## Gotchas / traps
- **Broken `torchcodec` silently empties whisper transcripts.** transformers 4.57.1's
  ASR pipeline runs `import torchcodec` in `preprocess` (every `pipe()` call) whenever
  torchcodec is merely *installed* (`find_spec` gate — it never verifies the native lib
  loads). torchcodec comes via the `[gigaam]` extra; if its native lib can't load
  (FFmpeg/torch mismatch) EVERY whisper segment raises `RuntimeError`. The session
  worker's per-segment `except` swallows that as a non-terminal skip → completed
  recording with an EMPTY transcript and NO error. Fixed in `WhisperEngine.__init__`
  via `_neutralize_broken_torchcodec()`: only when torchcodec is present but genuinely
  fails to import, it sets `transformers.utils.import_utils._torchcodec_available =
  False` so the pipeline skips the dead import (we always feed `{"raw", "sampling_rate"}`,
  never a decoder object). A working torchcodec is left untouched. GigaAM is unaffected
  (bypasses the transformers pipeline entirely).
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
- **`silero-vad==6.2.2` imports `onnxruntime` at package import but doesn't declare it** —
  `onnxruntime` is in base deps for this reason. Don't remove it.
- **Native capture rate is assumed 48000 Hz** (`soundcard` needs an explicit samplerate
  and has no reliable native-rate API); soxr resamples to 16 kHz. Revisit if a device's
  native rate differs — would need plumbing through config.
- **Dep pins are chosen for GigaAM-v3 compat:** `torch==2.8.0`, `torchaudio==2.8.0`,
  `transformers==4.57.1`. GigaAM extras (`[gigaam]`): pyannote-audio, hydra-core,
  omegaconf, sentencepiece, torchcodec. GigaAM is Russian-only and transcribes via a
  temp WAV (model takes a file path, not numpy).
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
run: actual capture, VAD segmentation on real audio, whisper + gigaam transcription,
GUI live/batch/file runs, CPU RTF measurement, hour-long RAM-stability + channel-sync
tests. See "Верификация" in `kind-hatching-allen.md`.

## Env
Windows, Python >=3.12, package manager **uv**. `uv sync` (base) / `uv sync --extra gigaam`.
CPU torch via default PyPI index; commented `pytorch-cu128` index in pyproject.toml is the
GPU opt-in. Run: `python -m app` (GUI), `python -m app.cli {list-devices,transcribe,record}`
(CLI), or `scripts/run-{gui,cli}.{ps1,bat}`.
