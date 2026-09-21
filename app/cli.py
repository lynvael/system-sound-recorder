"""Headless command-line entrypoint for Live Recorder.

Runs the exact same :class:`app.pipeline.session.Session` pipeline as the GUI,
but without Qt — for deterministic, mic/GPU-free regression runs (file/batch
transcription) and for scripting.

Run with ``python -m app.cli`` (the GUI entrypoint stays ``python -m app`` /
``app/__main__.py``; this module never imports PySide6).

Subcommands:
  - ``list-devices``          enumerate microphones and loopbacks (name — the
                              device "id" is the name itself; PortAudio
                              exports no stable endpoint id).
  - ``transcribe FILE``       headless file transcription of an arbitrary
                              audio/WAV file (a 2-channel file is treated as our
                              stereo ``session.wav`` for full attribution; mono
                              gets a neutral speaker label).
  - ``record``                headless live/batch capture from a mic + loopback,
                              running until Ctrl-C / EOF.

Threading: ``Session`` fires every callback (on_segment/on_status/on_backlog/
on_error/on_finished) from BACKGROUND threads. The CLI's main thread blocks on a
:class:`threading.Event` set by on_finished/on_error rather than busy-waiting;
segments stream to stdout as they arrive, status/backlog go to stderr.
"""

from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path

from app.config import load_config
from app.log import get_logger
from app.log import setup as setup_logging
from app.pipeline.session import Session
from app.pipeline.transcript import Segment

logger = get_logger("cli")


def _fmt_clock(seconds: float) -> str:
    """mm:ss (or h:mm:ss past an hour) — mirrors transcript._fmt_clock."""
    seconds = max(0.0, seconds)
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _print_segment(seg: Segment) -> None:
    """Stream one transcribed segment to stdout as ``[мм:сс] speaker: text``."""
    print(f"[{_fmt_clock(seg.start)}] {seg.speaker}: {seg.text}", flush=True)


def _eprint(text: str) -> None:
    """Status / backlog / errors go to stderr so stdout stays transcript-only."""
    print(text, file=sys.stderr, flush=True)


class _SessionRunner:
    """Wires Session callbacks to a blocking main-thread wait.

    ``done`` is set on either a natural finish (on_finished) or a terminal error
    (on_error); ``failed`` records whether the terminating condition was an
    error, driving the process exit code.
    """

    def __init__(self) -> None:
        self.done = threading.Event()
        self.failed = False
        self.session_dir: Path | None = None

    def on_segment(self, seg: Segment) -> None:
        _print_segment(seg)

    def on_status(self, text: str) -> None:
        _eprint(f"[статус] {text}")

    def on_backlog(self, depth: int) -> None:
        _eprint(f"[очередь] {depth}")

    def on_error(self, text: str) -> None:
        # on_error is the backend's terminal channel in headless file/batch
        # runs (per-segment transcription hiccups are rerouted to on_status).
        # Treat it as a failure and unblock the main thread.
        _eprint(f"[ошибка] {text}")
        self.failed = True
        self.done.set()

    def on_finished(self, session_dir: Path) -> None:
        self.session_dir = session_dir
        self.done.set()


# --- list-devices ----------------------------------------------------------
def _cmd_list_devices(args: argparse.Namespace) -> int:
    # Imported lazily so `list-devices` failures don't block `--help` and so the
    # PyAudioWPatch backend is only touched when actually enumerating.
    from app.audio import devices

    try:
        mics = devices.list_microphones()
        loopbacks = devices.list_loopbacks()
    except Exception as exc:  # noqa: BLE001 - e.g. non-Windows: no backend
        _eprint(f"[ошибка] Не удалось получить список устройств: {exc}")
        return 1
    print("=== Микрофоны ===")
    for name, dev_id in mics:
        print(f"  {name}\n    id: {dev_id}")
    print("\n=== Системный звук (loopback) ===")
    for name, dev_id in loopbacks:
        print(f"  {name}\n    id: {dev_id}")
    return 0


# --- transcribe ------------------------------------------------------------
def _cmd_transcribe(args: argparse.Namespace) -> int:
    file_path = Path(args.file)
    if not file_path.exists():
        _eprint(f"[ошибка] Файл не найден: {file_path}")
        return 1

    config = load_config()
    config.session.mode = "file"
    if args.output_dir:
        config.session.output_dir = args.output_dir

    runner = _SessionRunner()
    session = Session(
        config,
        "file",
        import_path=file_path,
        on_segment=runner.on_segment,
        on_status=runner.on_status,
        on_backlog=runner.on_backlog,
        on_error=runner.on_error,
        on_finished=runner.on_finished,
    )

    try:
        session.start()
    except Exception as exc:  # noqa: BLE001 - surface any startup failure
        # start() already emitted via on_error (which set runner.failed); this
        # catches the re-raised propagation so we exit non-zero cleanly.
        logger.debug("transcribe start() raised: %s", exc)
        return 1

    # Block until the file pass finishes (or errors) — no busy-wait.
    try:
        runner.done.wait()
    except KeyboardInterrupt:
        _eprint("[статус] Прерывание — остановка…")
        session.stop()
        return 130

    if runner.failed:
        return 1
    if runner.session_dir is not None:
        _eprint(
            f"[готово] Транскрипт записан в: {runner.session_dir} "
            "(transcript.txt / transcript.json / transcript.srt)"
        )
    return 0


# --- record ----------------------------------------------------------------
def _cmd_record(args: argparse.Namespace) -> int:
    if not args.no_mic and not args.mic_id:
        print("[ошибка] Нужен --mic-id (или --no-mic, чтобы не записывать микрофон).",
              file=sys.stderr)
        return 2
    config = load_config()
    config.session.mode = args.mode
    if args.output_dir:
        config.session.output_dir = args.output_dir

    runner = _SessionRunner()
    session = Session(
        config,
        args.mode,
        mic_id=args.mic_id,
        loopback_id=args.loopback_id,
        record_mic=not args.no_mic,
        on_segment=runner.on_segment,
        on_status=runner.on_status,
        on_backlog=runner.on_backlog,
        on_error=runner.on_error,
        on_finished=runner.on_finished,
    )

    try:
        session.start()
    except Exception as exc:  # noqa: BLE001
        logger.debug("record start() raised: %s", exc)
        return 1

    _eprint("[статус] Запись… нажмите Ctrl-C (или EOF на stdin) для остановки.")
    # Wait for the operator to end the capture: Ctrl-C, or EOF on stdin (so the
    # command composes in pipelines / non-interactive shells). on_finished/
    # on_error can also end it early (e.g. recorder-writer death).
    try:
        while not runner.done.is_set():
            line = sys.stdin.readline()
            if line == "":  # EOF
                break
    except KeyboardInterrupt:
        pass

    if not runner.done.is_set():
        _eprint("[статус] Остановка…")
        # stop() runs the batch transcription pass (blocking) for mode=batch,
        # then fires on_finished; for live it joins and finishes too.
        session.stop()
        runner.done.wait()

    if runner.failed:
        return 1
    if runner.session_dir is not None:
        _eprint(f"[готово] Сессия записана в: {runner.session_dir}")
    return 0


# --- argument parser -------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.cli",
        description="Live Recorder — headless CLI (no GUI).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser(
        "list-devices",
        help="List microphones and loopback (system-audio) devices.",
    )

    p_tr = sub.add_parser(
        "transcribe",
        help="Transcribe an audio/WAV file headlessly (file mode).",
    )
    p_tr.add_argument("file", help="Path to an audio/WAV file to transcribe.")
    p_tr.add_argument(
        "--output-dir",
        default=None,
        help="Where session dirs are written (default: config session.output_dir).",
    )

    p_rec = sub.add_parser(
        "record",
        help="Headless live/batch capture from a mic + loopback until Ctrl-C/EOF.",
    )
    p_rec.add_argument(
        "--mic-id",
        default=None,
        help="Microphone device NAME (from `list-devices`; the device id is "
        "the name). Required unless --no-mic is given.",
    )
    p_rec.add_argument(
        "--no-mic",
        action="store_true",
        help="Do not capture the mic channel (loopback only).",
    )
    p_rec.add_argument(
        "--loopback-id",
        required=True,
        help="Loopback device NAME (from `list-devices`; the device id is "
        "the name).",
    )
    p_rec.add_argument(
        "--mode",
        choices=("live", "batch"),
        default="batch",
        help="live = stream+transcribe as you go; batch = transcribe on stop. "
        "Default: batch.",
    )
    p_rec.add_argument(
        "--output-dir",
        default=None,
        help="Where session dirs are written (default: config session.output_dir).",
    )

    return parser


_HANDLERS = {
    "list-devices": _cmd_list_devices,
    "transcribe": _cmd_transcribe,
    "record": _cmd_record,
}


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Match the app entrypoint: configure logging (to stderr) from config.
    config = load_config()
    setup_logging(config.app.log_level)

    handler = _HANDLERS[args.command]
    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
