"""
Base LLM Adapter
کلاس پایه آداپتور LLM
"""

import asyncio
import json
import logging
import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional, Union

import httpx

logger = logging.getLogger(__name__)


class StreamTimeoutError(asyncio.TimeoutError):
    """Timeout carrying a machine-readable stream phase and budget."""

    def __init__(self, *, phase: str, url: str, timeout_sec: float):
        self.phase = phase
        self.url = url
        self.timeout_sec = timeout_sec
        self.diagnostic = {
            "event": "llm_stream_timeout",
            "phase": phase,
            "url": url,
            "timeout_sec": timeout_sec,
        }
        super().__init__(
            f"llm_stream_timeout phase={phase} timeout_sec={timeout_sec:g} url={url}"
        )


def extract_message_reasoning(message: Dict[str, Any]) -> str:
    """Pull CoT from OpenAI-compat assistant messages (vLLM/Qwen/llama.cpp)."""
    if not isinstance(message, dict):
        return ""
    for key in ("reasoning_content", "reasoning", "reasoning_text"):
        value = message.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, dict):
            text = str(value.get("content") or value.get("text") or "").strip()
            if text:
                return text
    return ""


@dataclass
class ChatMessage:
    """A single chat message (supports multimodal content parts)"""
    role: str  # system, user, assistant, tool
    content: Any  # str or List[dict] for multimodal
    name: Optional[str] = None
    tool_calls: Optional[List[Dict]] = None
    tool_call_id: Optional[str] = None
    # Prior-turn CoT for Qwen/Gemma preserve_thinking chat templates.
    reasoning_content: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {"role": self.role, "content": self.content}
        if self.name:
            d["name"] = self.name
        if self.tool_calls:
            d["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        if self.reasoning_content:
            d["reasoning_content"] = self.reasoning_content
            # Some OpenAI-compat stacks read `reasoning` instead.
            d["reasoning"] = self.reasoning_content
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChatMessage":
        reasoning = ""
        if isinstance(data, dict):
            reasoning = extract_message_reasoning(data)
        return cls(
            role=data.get("role", "user"),
            content=data.get("content", ""),
            name=data.get("name"),
            tool_calls=data.get("tool_calls"),
            tool_call_id=data.get("tool_call_id"),
            reasoning_content=reasoning or None,
        )


@dataclass
class ChatResponse:
    """Response from LLM"""
    content: str
    role: str = "assistant"
    finish_reason: Optional[str] = None
    tool_calls: Optional[List[Dict]] = None
    # Qwen3 / vLLM thinking: set when caller asked enable_thinking=True (passthrough)
    reasoning: Optional[str] = None
    usage: Optional[Dict[str, int]] = None
    model: Optional[str] = None
    latency_ms: Optional[float] = None
    raw_response: Optional[Dict] = None
    
    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "content": self.content,
            "role": self.role,
            "finish_reason": self.finish_reason,
            "tool_calls": self.tool_calls,
            "usage": self.usage,
            "model": self.model,
            "latency_ms": self.latency_ms,
        }
        if self.reasoning is not None:
            d["reasoning"] = self.reasoning
        return d
    
    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to OpenAI API format"""
        message: Dict[str, Any] = {"role": self.role}
        # null content is valid when thinking output lives in `reasoning`
        message["content"] = self.content if self.content else None
        if self.tool_calls:
            message["tool_calls"] = self.tool_calls
        if self.reasoning is not None:
            message["reasoning"] = self.reasoning
            message["reasoning_content"] = self.reasoning
        
        return {
            "id": f"chatcmpl-{int(time.time()*1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model or "unknown",
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": self.finish_reason or "stop"
            }],
            "usage": self.usage or {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        }


class BaseLLMAdapter(ABC):
    """
    Base class for LLM adapters
    کلاس پایه برای آداپتورهای LLM

    ویژگی‌ها:
    - Connection pooling (httpx.AsyncClient اشتراکی)
    - Retry با exponential backoff
    - مدیریت lifecycle (close)
    """

    name: str = "base"

    # ── Retry defaults (قابل override از config) ─────────────────────
    MAX_RETRIES: int = 3
    RETRY_DELAY: float = 1.0
    RETRYABLE_EXCEPTIONS = (httpx.ConnectError, httpx.ConnectTimeout)

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: Optional[str] = None,
        timeout: float = 120.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        **kwargs
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.extra_config = kwargs
        # A read timeout in httpx is an inactivity timeout, not a total response
        # deadline. This keeps long generations valid while bounding a stalled
        # stream. Existing constructors remain source-compatible.
        self.stream_connect_timeout = float(
            kwargs.get("stream_connect_timeout")
            or os.getenv("LLM_STREAM_CONNECT_TIMEOUT_SEC", "10")
        )
        self.stream_pool_timeout = float(
            kwargs.get("stream_pool_timeout")
            or os.getenv("LLM_STREAM_POOL_TIMEOUT_SEC", "10")
        )
        # Agent Flow mid-run prompts (tool traces + media parts) routinely
        # exceed 30s to upload; a short write budget falsely trips health.
        self.stream_write_timeout = float(
            kwargs.get("stream_write_timeout")
            or os.getenv("LLM_STREAM_WRITE_TIMEOUT_SEC", "180")
        )
        self.stream_first_byte_timeout = float(
            kwargs.get("stream_first_byte_timeout")
            or os.getenv("LLM_STREAM_FIRST_BYTE_TIMEOUT_SEC", "120")
        )
        self.stream_idle_timeout = float(
            kwargs.get("stream_idle_timeout")
            or os.getenv("LLM_STREAM_IDLE_TIMEOUT_SEC", str(timeout))
        )

        # ── Shared connection pool ────────────────────────
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout, connect=10.0),
            limits=httpx.Limits(
                max_connections=20,
                max_keepalive_connections=10,
                keepalive_expiry=30.0,
            ),
        )
    
    @abstractmethod
    async def chat(
        self,
        messages: List[ChatMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict]] = None,
        tool_choice: Optional[Union[str, Dict]] = None,
        response_format: Optional[Dict] = None,
        **kwargs
    ) -> ChatResponse:
        """
        Send chat request and get response
        ارسال درخواست چت و دریافت پاسخ
        """
        pass
    
    @abstractmethod
    async def chat_stream(
        self,
        messages: List[ChatMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncIterator[Union[str, Dict[str, Any]]]:
        """
        Send chat request and stream response
        ارسال درخواست چت و استریم پاسخ
        
        ✨ می‌تونه str یا Dict برگردونه:
        - str: فقط content (قدیمی)
        - Dict: {"content": ..., "reasoning": ...} (جدید)
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """
        Check if backend is healthy
        بررسی سلامت بک‌اند
        """
        pass
    
    @abstractmethod
    async def list_models(self) -> List[Dict[str, Any]]:
        """
        List available models
        لیست مدل‌های موجود
        """
        pass
    
    def _extract_json(self, text: str) -> Optional[Dict]:
        """Extract JSON from text response"""
        # Try to find JSON block
        patterns = [
            r"```json\s*([\s\S]*?)\s*```",
            r"```\s*([\s\S]*?)\s*```",
            r"\{[\s\S]*\}"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                json_str = match.group(1) if "```" in pattern else match.group(0)
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    # Try to fix common issues
                    fixed = json_str.replace("\n", " ")
                    fixed = re.sub(r",\s*}", "}", fixed)
                    fixed = re.sub(r",\s*]", "]", fixed)
                    try:
                        return json.loads(fixed)
                    except:
                        continue
        return None
    
    def _build_headers(self) -> Dict[str, str]:
        """Build request headers"""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    # ── Connection lifecycle ──────────────────────────────────────────

    async def close(self) -> None:
        """Close the shared HTTP client. Call on shutdown."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    def _stream_timeout(self) -> httpx.Timeout:
        """Explicit phase budgets; read is reset by every received network chunk."""
        return httpx.Timeout(
            connect=self.stream_connect_timeout,
            pool=self.stream_pool_timeout,
            write=self.stream_write_timeout,
            read=self.stream_idle_timeout,
        )

    def _stream_timeout_from_httpx(
        self, exc: httpx.TimeoutException, *, url: str
    ) -> StreamTimeoutError:
        phase = "idle_read"
        budget = self.stream_idle_timeout
        if isinstance(exc, httpx.PoolTimeout):
            phase, budget = "pool", self.stream_pool_timeout
        elif isinstance(exc, httpx.ConnectTimeout):
            phase, budget = "connect", self.stream_connect_timeout
        elif isinstance(exc, httpx.WriteTimeout):
            phase, budget = "write", self.stream_write_timeout
        err = StreamTimeoutError(phase=phase, url=url, timeout_sec=budget)
        logger.warning("%s", json.dumps(err.diagnostic, sort_keys=True))
        return err

    async def _iter_stream_lines(
        self, response: httpx.Response, *, url: str
    ) -> AsyncIterator[str]:
        """Apply a first-byte budget, then rely on httpx's idle read timeout."""
        lines = response.aiter_lines()
        try:
            first = await asyncio.wait_for(
                lines.__anext__(), timeout=self.stream_first_byte_timeout
            )
        except StopAsyncIteration:
            return
        except asyncio.TimeoutError as exc:
            err = StreamTimeoutError(
                phase="first_byte",
                url=url,
                timeout_sec=self.stream_first_byte_timeout,
            )
            logger.warning("%s", json.dumps(err.diagnostic, sort_keys=True))
            raise err from exc
        yield first
        try:
            async for line in lines:
                yield line
        except httpx.TimeoutException as exc:
            raise self._stream_timeout_from_httpx(exc, url=url) from exc

    # ── Retry helper ─────────────────────────────────────────────────

    async def _request_with_retry(
        self,
        method: str,
        url: str,
        *,
        max_retries: Optional[int] = None,
        **kwargs,
    ) -> httpx.Response:
        """
        Send an HTTP request with automatic retry + exponential backoff.

        Retries on connection errors and timeouts.
        Does NOT retry on HTTP 4xx/5xx (those propagate immediately).
        """
        retries = max_retries if max_retries is not None else self.max_retries
        last_exc: Optional[Exception] = None

        for attempt in range(retries + 1):
            try:
                resp = await self._client.request(method, url, **kwargs)
                return resp
            except self.RETRYABLE_EXCEPTIONS as exc:
                last_exc = exc
                if attempt < retries:
                    delay = self.retry_delay * (2 ** attempt)
                    logger.warning(
                        f"Retry {attempt + 1}/{retries} for {method} {url} "
                        f"after {type(exc).__name__}: {exc} — waiting {delay:.1f}s"
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        f"All {retries + 1} attempts failed for {method} {url}: {exc}"
                    )

        raise last_exc  # type: ignore[misc]
