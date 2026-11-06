"""Tests for the LLM client helper: every call must carry the reasoning
kwargs (the proxy is unusably slow without them) and no debug output."""

from __future__ import annotations

from types import SimpleNamespace

from app.config import LLMSettings
from app.summarize.client import chat


class FakeClient:
    def __init__(self, content: str = "ответ"):
        self.calls: list[dict] = []
        self._content = content
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self._content))]
        )

    def close(self):
        pass


def test_chat_passes_reasoning_kwargs_on_every_call():
    llm = LLMSettings(_env_file=None)
    client = FakeClient()
    for _ in range(3):
        assert chat(client, llm, "система", "пользователь") == "ответ"

    assert len(client.calls) == 3
    for call in client.calls:
        assert call["reasoning_effort"] == "medium"
        assert call["extra_body"] == {"allowed_openai_params": ["reasoning_effort"]}
        assert "response_format" not in call
        assert call["messages"] == [
            {"role": "system", "content": "система"},
            {"role": "user", "content": "пользователь"},
        ]


def test_chat_raises_on_no_choices():
    client = FakeClient()
    client.chat.completions.create = lambda **kw: SimpleNamespace(choices=[])
    llm = LLMSettings(_env_file=None)
    try:
        chat(client, llm, "s", "u")
        raise AssertionError("expected RuntimeError")
    except RuntimeError as exc:
        assert "пустой ответ" in str(exc)


def test_chat_raises_on_blank_content():
    client = FakeClient(content="   ")
    llm = LLMSettings(_env_file=None)
    try:
        chat(client, llm, "s", "u")
        raise AssertionError("expected RuntimeError")
    except RuntimeError as exc:
        assert "пустой текст" in str(exc)
