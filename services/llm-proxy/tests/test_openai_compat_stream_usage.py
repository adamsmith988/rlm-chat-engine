"""stream_options.include_usage must be set so metering can capture tokens."""

from src.adapters.openai_compat import OpenAICompatAdapter


def _adapter() -> OpenAICompatAdapter:
    return OpenAICompatAdapter(
        base_url="https://example.invalid",
        api_key="k",
        model="m",
    )


def test_stream_payload_includes_usage_by_default():
    payload = _adapter()._build_chat_payload(
        [{"role": "user", "content": "hi"}],
        temperature=0.2,
        max_tokens=8,
        stream=True,
    )
    assert payload["stream"] is True
    assert payload["stream_options"] == {"include_usage": True}


def test_stream_payload_preserves_caller_stream_options():
    payload = _adapter()._build_chat_payload(
        [{"role": "user", "content": "hi"}],
        temperature=0.2,
        max_tokens=8,
        stream=True,
        stream_options={"include_usage": False, "foo": 1},
    )
    # Caller may disable; we only setdefault when missing.
    assert payload["stream_options"] == {"include_usage": False, "foo": 1}


def test_non_stream_payload_omits_stream_options():
    payload = _adapter()._build_chat_payload(
        [{"role": "user", "content": "hi"}],
        temperature=0.2,
        max_tokens=8,
        stream=False,
    )
    assert "stream_options" not in payload
