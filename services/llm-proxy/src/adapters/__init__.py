"""
LLM Adapters
آداپتورهای LLM برای بک‌اندهای مختلف
"""

from .base import BaseLLMAdapter, ChatMessage, ChatResponse
from .triton import TritonAdapter
from .openai_compat import OpenAICompatAdapter, DeepSeekAdapter, OpenAIAdapter

__all__ = [
    "BaseLLMAdapter",
    "ChatMessage",
    "ChatResponse",
    "TritonAdapter",
    "OpenAICompatAdapter",
    "DeepSeekAdapter",
    "OpenAIAdapter",
]
