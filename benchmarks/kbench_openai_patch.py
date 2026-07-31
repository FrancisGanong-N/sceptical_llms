"""Work around Model Proxy responses with a null chat ``message``.

kaggle-benchmarks ``OpenAI._call_api`` does ``message.tool_calls`` after reading
``response.choices[0].message``. Anthropic models (notably Claude Opus 5) can
return a choice whose ``message`` is ``None``, which aborts the whole evaluate
loop. Normalize those responses to empty ``choices`` so the existing empty-choice
path returns blank content instead of raising.
"""

from __future__ import annotations

_PATCHED = False


def _normalize_null_message_response(response):
    choices = getattr(response, "choices", None)
    if not choices:
        return response
    first = choices[0]
    if first is None or getattr(first, "message", None) is not None:
        return response
    try:
        response.choices = []
    except Exception:
        try:
            object.__setattr__(response, "choices", [])
        except Exception:
            pass
    return response


def apply_null_message_patch() -> bool:
    """Patch ``OpenAI._call_api`` once. Returns True if newly applied."""
    global _PATCHED
    if _PATCHED:
        return False

    from kaggle_benchmarks.actors.llms import OpenAI

    original = OpenAI._call_api

    def _call_api(self, messages, **kwargs):
        create = self.client.chat.completions.create

        def create_wrapper(*args, **create_kwargs):
            response = create(*args, **create_kwargs)
            return _normalize_null_message_response(response)

        self.client.chat.completions.create = create_wrapper
        try:
            return original(self, messages, **kwargs)
        finally:
            self.client.chat.completions.create = create

    OpenAI._call_api = _call_api
    _PATCHED = True
    return True
