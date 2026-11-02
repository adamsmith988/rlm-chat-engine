from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adapters.base import ChatResponse, extract_message_reasoning


def test_extract_message_reasoning_prefers_reasoning_content():
    text = extract_message_reasoning(
        {
            "content": "visible",
            "reasoning_content": "plan first",
            "reasoning": "ignored",
        }
    )
    assert text == "plan first"


def test_openai_format_copies_reasoning_content():
    resp = ChatResponse(content="hi", reasoning="think step")
    payload = resp.to_openai_format()
    msg = payload["choices"][0]["message"]
    assert msg["content"] == "hi"
    assert msg["reasoning"] == "think step"
    assert msg["reasoning_content"] == "think step"


def test_chat_message_to_dict_preserves_reasoning_content():
    from adapters.base import ChatMessage

    msg = ChatMessage(
        role="assistant",
        content=None,
        tool_calls=[{"id": "1", "type": "function", "function": {"name": "ipython", "arguments": "{}"}}],
        reasoning_content="Continue from step 3 of the plan",
    )
    payload = msg.to_dict()
    assert payload["reasoning_content"] == "Continue from step 3 of the plan"
    assert payload["reasoning"] == "Continue from step 3 of the plan"
    assert payload["tool_calls"]


def test_chat_message_from_dict_reads_reasoning_aliases():
    from adapters.base import ChatMessage

    msg = ChatMessage.from_dict(
        {"role": "assistant", "content": "ok", "reasoning": "prior plan state"}
    )
    assert msg.reasoning_content == "prior plan state"
    assert msg.to_dict()["reasoning_content"] == "prior plan state"
