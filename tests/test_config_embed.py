"""Tests for EmbedSettings and the changed LLMSettings defaults."""

from __future__ import annotations

from app.config import Config, EmbedSettings, LLMSettings


class TestEmbedSettings:
    def test_defaults_not_configured(self):
        s = EmbedSettings(_env_file=None)
        assert s.url == ""
        assert s.model == ""
        assert s.api_key == "not-needed"
        assert s.request_timeout == 60.0
        assert not s.is_configured

    def test_partial_not_configured(self, monkeypatch):
        monkeypatch.setenv("EMBED_URL", "http://e/v1")
        assert not EmbedSettings(_env_file=None).is_configured
        monkeypatch.delenv("EMBED_URL")
        monkeypatch.setenv("EMBED_MODEL", "model-x")
        assert not EmbedSettings(_env_file=None).is_configured

    def test_full_configured(self, monkeypatch):
        monkeypatch.setenv("EMBED_URL", "http://e/v1")
        monkeypatch.setenv("EMBED_MODEL", "model-x")
        s = EmbedSettings(_env_file=None)
        assert s.is_configured
        assert s.url == "http://e/v1"
        assert s.model == "model-x"

    def test_whitespace_only_not_configured(self, monkeypatch):
        monkeypatch.setenv("EMBED_URL", "   ")
        monkeypatch.setenv("EMBED_MODEL", "model-x")
        assert not EmbedSettings(_env_file=None).is_configured


class TestLLMSettingsDefaults:
    def test_request_timeout_default_is_300(self):
        assert LLMSettings(_env_file=None).request_timeout == 300.0

    def test_concurrency_default_is_3(self):
        assert LLMSettings(_env_file=None).concurrency == 3

    def test_concurrency_from_env(self, monkeypatch):
        monkeypatch.setenv("LLM_CONCURRENCY", "5")
        assert LLMSettings(_env_file=None).concurrency == 5


def test_config_exposes_embed():
    cfg = Config()
    assert isinstance(cfg.embed, EmbedSettings)
    assert isinstance(cfg.llm, LLMSettings)
