"""Tests for GigaAM temp-WAV directory handling (app/stt/gigaam_engine.py).

The production GigaAM model is unavailable in the build environment (no
model downloads), so `transformers.AutoModel.from_pretrained` is monkeypatched
with a fake model that records the path handed to `transcribe()`. This
verifies, without the real model:

- the temp WAV is created in the dedicated `work_dir` (created on demand)
  and cleaned up afterwards;
- graceful fallback to the system tempdir when work_dir is None,
  non-ASCII, or uncreatable (no raises);
- transcription errors (from `sf.write` or `model.transcribe`) propagate
  with the temp file's FULL PATH embedded in the message — the session
  worker logs only the exception's str, so that line must be
  self-diagnosable.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

import app.stt.gigaam_engine as gigaam_mod
from app.config import Config, STTSettings, VADSettings
from app.stt.factory import build_engine

SAMPLE_RATE = 16000


class _FakeModel:
    """Stand-in for the GigaAM model: records paths, returns fixed text."""

    def __init__(self, text: str = "привет мир") -> None:
        self.text = text
        self.seen_paths: list[str] = []
        self.error: Exception | None = None

    def transcribe(self, path: str):
        self.seen_paths.append(path)
        if self.error is not None:
            raise self.error
        return self.text


@pytest.fixture()
def fake_model(monkeypatch: pytest.MonkeyPatch) -> _FakeModel:
    import transformers

    model = _FakeModel()
    monkeypatch.setattr(
        transformers.AutoModel,
        "from_pretrained",
        staticmethod(lambda *args, **kwargs: model),
    )
    return model


def _make_engine(work_dir: str | Path | None) -> gigaam_mod.GigaAMEngine:
    return gigaam_mod.GigaAMEngine(
        STTSettings(), SAMPLE_RATE, VADSettings(), work_dir=work_dir
    )


def _one_second() -> np.ndarray:
    # 1 s at 16 kHz — well under _MAX_TRANSCRIBE_SECONDS, so the short form
    # (the temp-WAV path under test) is taken.
    return np.zeros(SAMPLE_RATE, dtype=np.float32)


def _ascii_work_dir(tmp_path: Path) -> Path:
    """A work dir whose resolved path is ASCII (a test tmp_path may not be)."""
    d = tmp_path / "stt_tmp"
    if not str(d.resolve()).isascii():
        pytest.skip(f"test tmp_path resolves to a non-ASCII path: {d.resolve()}")
    return d


def test_temp_wav_written_to_work_dir_and_cleaned(tmp_path, fake_model):
    work_dir = _ascii_work_dir(tmp_path)
    engine = _make_engine(work_dir)  # dir does not exist yet

    # The engine creates the dir on demand (mkdir parents/exist_ok).
    assert work_dir.is_dir()
    assert engine._tmp_dir == str(work_dir.resolve())

    text = engine.transcribe(_one_second())
    assert text == "привет мир"
    assert len(fake_model.seen_paths) == 1
    seen = fake_model.seen_paths[0]
    assert Path(seen).parent == work_dir.resolve()
    assert Path(seen).name.startswith("gigaam_")
    assert Path(seen).name.endswith(".wav")

    # The temp WAV is removed after transcribe (finally-cleanup intact).
    assert list(work_dir.glob("gigaam_*.wav")) == []


def test_fallback_to_system_tempdir_when_work_dir_none(fake_model):
    engine = _make_engine(None)
    assert engine._tmp_dir is None

    engine.transcribe(_one_second())
    seen = fake_model.seen_paths[0]
    assert Path(seen).is_relative_to(Path(tempfile.gettempdir()))


def test_fallback_to_system_tempdir_when_work_dir_empty(fake_model):
    # An empty string means "not given" (like None), not Path(".") == CWD.
    engine = _make_engine("")
    assert engine._tmp_dir is None

    engine.transcribe(_one_second())
    seen = fake_model.seen_paths[0]
    assert Path(seen).is_relative_to(Path(tempfile.gettempdir()))


def test_fallback_when_work_dir_not_ascii(tmp_path, fake_model):
    engine = _make_engine(tmp_path / "кириллица")
    assert engine._tmp_dir is None  # rejected with a warning, not raised

    engine.transcribe(_one_second())
    seen = fake_model.seen_paths[0]
    assert Path(seen).is_relative_to(Path(tempfile.gettempdir()))


def test_fallback_when_work_dir_uncreatable(tmp_path, fake_model):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a directory")
    engine = _make_engine(blocker / ".stt_tmp")
    assert engine._tmp_dir is None  # mkdir fails -> fallback, no raise

    engine.transcribe(_one_second())
    seen = fake_model.seen_paths[0]
    assert Path(seen).is_relative_to(Path(tempfile.gettempdir()))


def test_model_error_message_contains_full_temp_path(tmp_path, fake_model):
    work_dir = _ascii_work_dir(tmp_path)
    fake_model.error = FileNotFoundError(
        "[WinError 2] Не удается найти указанный файл"
    )
    engine = _make_engine(work_dir)

    with pytest.raises(RuntimeError) as excinfo:
        engine.transcribe(_one_second())

    msg = str(excinfo.value)
    seen = fake_model.seen_paths[0]
    assert seen in msg  # full temp path -> diagnosable from the log line
    assert "[WinError 2]" in msg
    assert isinstance(excinfo.value.__cause__, FileNotFoundError)


def test_write_error_message_contains_full_temp_path(
    tmp_path, fake_model, monkeypatch
):
    work_dir = _ascii_work_dir(tmp_path)

    def _boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(gigaam_mod.sf, "write", _boom)
    engine = _make_engine(work_dir)

    with pytest.raises(RuntimeError) as excinfo:
        engine.transcribe(_one_second())

    msg = str(excinfo.value)
    assert "disk full" in msg
    assert "(temp WAV:" in msg
    assert "gigaam_" in msg
    # even a failed write leaves no temp file behind
    assert list(work_dir.glob("gigaam_*.wav")) == []


def test_factory_passes_work_dir_through(monkeypatch):
    captured: dict = {}

    class _SpyEngine:
        def __init__(self, settings, sample_rate, vad, work_dir=None):
            captured["work_dir"] = work_dir

    monkeypatch.setattr(gigaam_mod, "GigaAMEngine", _SpyEngine)
    build_engine(Config(), work_dir="/some/ascii/dir")
    assert captured["work_dir"] == "/some/ascii/dir"


def test_session_stt_tmp_dir_is_outside_session_dir(tmp_path):
    try:
        from app.pipeline.session import Session
    except Exception as exc:  # soundcard/PulseAudio may be unavailable headless
        pytest.skip(f"app.pipeline.session not importable here: {exc}")

    config = Config()
    config.session.output_dir = str(tmp_path / "recordings")

    plain = Session(config, "file", import_path="/nonexistent.wav")
    assert plain._stt_tmp_dir == Path(config.session.output_dir) / ".stt_tmp"
    # Siblings under output_dir — never nested in each other.
    assert plain._stt_tmp_dir.is_relative_to(Path(config.session.output_dir))
    assert not plain.session_dir.is_relative_to(plain._stt_tmp_dir)
    assert not plain._stt_tmp_dir.is_relative_to(plain.session_dir)

    # A Cyrillic session name stays in session_dir only; the temp dir is
    # shared and name-independent (session names are deliberately
    # allowed to keep Cyrillic — see _sanitize_session_name).
    named = Session(
        config, "file", import_path="/nonexistent.wav", name="Встреча с Иваном"
    )
    assert "Встреча" in named.session_dir.name
    assert named._stt_tmp_dir == plain._stt_tmp_dir
