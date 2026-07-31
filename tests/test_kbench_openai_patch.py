"""Tests for Model Proxy null-message workaround."""

from __future__ import annotations

from types import SimpleNamespace

from benchmarks.kbench_openai_patch import _normalize_null_message_response


class TestNormalizeNullMessageResponse:
    def test_leaves_normal_message_alone(self):
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="A", tool_calls=None))]
        )
        out = _normalize_null_message_response(response)
        assert out.choices[0].message.content == "A"

    def test_clears_choices_when_message_is_none(self):
        response = SimpleNamespace(choices=[SimpleNamespace(message=None)])
        out = _normalize_null_message_response(response)
        assert out.choices == []

    def test_empty_choices_unchanged(self):
        response = SimpleNamespace(choices=[])
        assert _normalize_null_message_response(response).choices == []

    def test_none_choices_unchanged(self):
        response = SimpleNamespace(choices=None)
        assert _normalize_null_message_response(response).choices is None
