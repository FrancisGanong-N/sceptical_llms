"""Tests for simple-rate Kaggle task helpers."""

from __future__ import annotations

from benchmarks.simple_rate_tasks import (
    SIMPLE_MAX_OUTPUT_TOKENS,
    _llm_extra_api_params,
    _prompt_llm,
)


class TestPromptLlm:
    def test_default_max_output_tokens_constant(self):
        assert SIMPLE_MAX_OUTPUT_TOKENS == 512

    def test_llm_extra_api_params(self):
        params = _llm_extra_api_params(512)
        assert params["max_tokens"] == 512
        assert params["modalities"] == ["text"]
        assert params["extra_body"]["max_output_tokens"] == 512

    def test_prompt_llm_passes_max_output_tokens_from_caller(self):
        class MockLLM:
            def __init__(self) -> None:
                self.last_extra: dict | None = None

            def prompt(self, prompt, *, reasoning=None, extra_api_params=None):
                self.last_extra = extra_api_params
                return "A"

        llm = MockLLM()
        _prompt_llm(llm, "test prompt", max_output_tokens=256)
        assert llm.last_extra is not None
        assert llm.last_extra["max_tokens"] == 256
        assert llm.last_extra["modalities"] == ["text"]
        assert llm.last_extra["extra_body"]["max_output_tokens"] == 256
