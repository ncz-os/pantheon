from __future__ import annotations

from mnemos.api.routes.pantheon import _stream_usage_response


def test_stream_usage_fallback_uses_character_count_without_completion_text():
    response = _stream_usage_response(
        model="model",
        usage=None,
        fallback_prompt_tokens=3,
        completion_chars=9,
    )

    assert response == {
        "model": "model",
        "usage": {"prompt_tokens": 3, "completion_tokens": 3, "total_tokens": 6},
    }


def test_stream_usage_prefers_provider_usage():
    response = _stream_usage_response(
        model="model",
        usage={"prompt_tokens": 4, "completion_tokens": 2},
        fallback_prompt_tokens=100,
        completion_chars=100,
    )

    assert response == {
        "model": "model",
        "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
    }
