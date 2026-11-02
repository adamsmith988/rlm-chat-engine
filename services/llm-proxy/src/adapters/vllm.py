"""
vLLM Adapter
آداپتور vLLM برای اجرای سریع مدل‌ها با GPU
"""

import json
import time
import logging
from typing import Any, AsyncIterator, Dict, List, Optional, Union

import httpx

from .base import BaseLLMAdapter, ChatMessage, ChatResponse, extract_message_reasoning

logger = logging.getLogger(__name__)


class VLLMAdapter(BaseLLMAdapter):
    """
    Adapter for vLLM inference server
    آداپتور برای سرور استنتاج vLLM
    
    vLLM provides OpenAI-compatible API with:
    - High throughput (continuous batching)
    - Low latency
    - Tensor parallelism
    - Tool calling support
    """
    
    name = "vllm"
    
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        model: str = "",
        api_key: Optional[str] = None,
        timeout: float = 120.0,
        **kwargs
    ):
        super().__init__(base_url, model, api_key=api_key, timeout=timeout, **kwargs)
    
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
        """Send chat request to vLLM"""
        start_time = time.time()
        
        # Prepare messages
        msg_list = [m.to_dict() if isinstance(m, ChatMessage) else m for m in messages]
        
        # Build OpenAI-compatible payload
        payload = {
            "model": kwargs.get("model", self.model),
            "messages": msg_list,
            "temperature": temperature,
            "stream": False
        }
        
        if max_tokens:
            payload["max_tokens"] = max_tokens
        
        if tools:
            payload["tools"] = tools
            if tool_choice:
                payload["tool_choice"] = tool_choice
        
        if response_format:
            payload["response_format"] = response_format
        
        if "top_p" in kwargs:
            payload["top_p"] = kwargs["top_p"]
        if "top_k" in kwargs:
            payload["top_k"] = kwargs["top_k"]
        if "repetition_penalty" in kwargs:
            payload["repetition_penalty"] = kwargs["repetition_penalty"]
        if "chat_template_kwargs" in kwargs:
            payload["chat_template_kwargs"] = kwargs["chat_template_kwargs"]
        
        try:
            response = await self._request_with_retry(
                "POST",
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                headers=self._build_headers(),
            )
            
            # اگه خطای tool calling بود، بدون tools امتحان کن
            if response.status_code == 400 and tools:
                try:
                    error_data = response.json()
                    error_msg = error_data.get("error", {}).get("message", "")
                    if "auto" in error_msg or "tool" in error_msg.lower():
                        logger.warning(f"Tool calling not supported, retrying without tools")
                        payload.pop("tools", None)
                        payload.pop("tool_choice", None)
                        response = await self._request_with_retry(
                            "POST",
                            f"{self.base_url}/v1/chat/completions",
                            json=payload,
                            headers=self._build_headers(),
                        )
                except Exception:
                    pass
            
            # Include response body in 4xx errors so callers can see vLLM's
            # detailed error message (e.g. context length exceeded)
            if not response.is_success:
                try:
                    err_body = response.text[:500]
                except Exception:
                    err_body = "<unreadable>"
                logger.error("vLLM returned %d: %s", response.status_code, err_body)
                # Raise with body info so it propagates up
                raise httpx.HTTPStatusError(
                    f"HTTP {response.status_code}: {err_body}",
                    request=response.request,
                    response=response,
                )

            data = response.json()
            
            choice = data.get("choices", [{}])[0]
            message = choice.get("message", {}) or {}

            ct_kw = kwargs.get("chat_template_kwargs")
            if not isinstance(ct_kw, dict):
                ct_kw = {}
            # Per-request: True = keep thinking in `reasoning`; False = JSON/tools see merged `content` only
            enable_thinking = ct_kw.get("enable_thinking")

            raw_content = message.get("content")
            if raw_content is None:
                text_content = ""
            elif isinstance(raw_content, str):
                text_content = raw_content
            else:
                text_content = str(raw_content)

            reasoning_text = extract_message_reasoning(message)

            out_reasoning: Optional[str] = None
            if enable_thinking is True:
                # Orchestrator / chat UI: preserve chain-of-thought separately
                out_content = text_content
                out_reasoning = reasoning_text if reasoning_text else None
            else:
                # enable_thinking False or omitted: single visible answer in `content`
                out_content = text_content
                if not out_content.strip() and reasoning_text:
                    out_content = reasoning_text

            return ChatResponse(
                content=out_content or "",
                role=message.get("role", "assistant"),
                finish_reason=choice.get("finish_reason"),
                tool_calls=message.get("tool_calls"),
                reasoning=out_reasoning,
                usage=data.get("usage"),
                model=data.get("model", self.model),
                latency_ms=(time.time() - start_time) * 1000,
                raw_response=data
            )
                
        except httpx.HTTPError as e:
            logger.error(f"vLLM request failed: {e}")
            raise
    
    async def chat_stream(
        self,
        messages: List[ChatMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict]] = None,
        tool_choice: Optional[Union[str, Dict]] = None,
        response_format: Optional[Dict] = None,
        **kwargs
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Stream chat response from vLLM with reasoning + tool_calls support.

        Yields dicts with optional:
        - content
        - reasoning
        - tool_calls
        """
        msg_list = [m.to_dict() if isinstance(m, ChatMessage) else m for m in messages]

        payload: Dict[str, Any] = {
            "model": kwargs.get("model", self.model),
            "messages": msg_list,
            "temperature": temperature,
            "stream": True,
        }

        if max_tokens:
            payload["max_tokens"] = max_tokens
        if tools:
            payload["tools"] = tools
            if tool_choice:
                payload["tool_choice"] = tool_choice
        if response_format:
            payload["response_format"] = response_format
        if "top_p" in kwargs:
            payload["top_p"] = kwargs["top_p"]
        if "top_k" in kwargs:
            payload["top_k"] = kwargs["top_k"]
        if "repetition_penalty" in kwargs:
            payload["repetition_penalty"] = kwargs["repetition_penalty"]
        if "chat_template_kwargs" in kwargs:
            payload["chat_template_kwargs"] = kwargs["chat_template_kwargs"]

        try:
            stream_url = f"{self.base_url}/v1/chat/completions"
            async with self._client.stream(
                "POST",
                stream_url,
                json=payload,
                headers=self._build_headers(),
                timeout=self._stream_timeout(),
            ) as response:
                response.raise_for_status()
                async for line in self._iter_stream_lines(response, url=stream_url):
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            delta = data.get("choices", [{}])[0].get("delta", {}) or {}
                            content = delta.get("content", "") or ""
                            reasoning = (
                                delta.get("reasoning_content")
                                or delta.get("reasoning")
                                or delta.get("reasoning_text")
                                or ""
                            )
                            tool_calls = delta.get("tool_calls")
                            choice0 = (data.get("choices") or [{}])[0] or {}
                            finish_reason = choice0.get("finish_reason")
                            usage = data.get("usage")
                            if content or reasoning or tool_calls or finish_reason or usage:
                                out: Dict[str, Any] = {}
                                if content:
                                    out["content"] = content
                                if reasoning:
                                    out["reasoning"] = reasoning
                                if tool_calls:
                                    out["tool_calls"] = tool_calls
                                if finish_reason:
                                    out["finish_reason"] = finish_reason
                                if usage:
                                    out["usage"] = usage
                                yield out
                        except json.JSONDecodeError:
                            continue
        except httpx.TimeoutException as e:
            raise self._stream_timeout_from_httpx(e, url=stream_url) from e
        except httpx.HTTPError as e:
            logger.error(f"vLLM stream failed: {e}")
            raise
    
    async def health_check(self) -> Dict[str, Any]:
        """Check vLLM server health"""
        import time as _time

        t0 = _time.time()
        try:
            response = await self._client.get(
                f"{self.base_url}/v1/models",
                headers=self._build_headers(),
                timeout=5.0,
            )
            latency_ms = round((_time.time() - t0) * 1000, 1)
            if response.status_code == 200:
                data = response.json()
                models = [m["id"] for m in data.get("data", [])]
                return {
                    "healthy": True,
                    "backend": "vllm",
                    "base_url": self.base_url,
                    "latency_ms": latency_ms,
                    "models": models,
                    "current_model": self.model
                }
            body_head = (response.text or "").strip().replace("\n", " ")[:200]
            return {
                "healthy": False,
                "backend": "vllm",
                "base_url": self.base_url,
                "latency_ms": latency_ms,
                "error": f"HTTP {response.status_code}" + (f": {body_head}" if body_head else ""),
            }
        except Exception as e:
            logger.warning(f"vLLM health check failed: {e}")
            return {
                "healthy": False,
                "backend": "vllm",
                "base_url": self.base_url,
                "error": f"{type(e).__name__}: {e}",
            }
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """List available vLLM models"""
        try:
            response = await self._client.get(
                f"{self.base_url}/v1/models",
                headers=self._build_headers(),
                timeout=10.0,
            )
            response.raise_for_status()
            data = response.json()
            return data.get("data", [])
        except Exception as e:
            logger.error(f"Failed to list vLLM models: {e}")
            return []
    
    async def generate_completions(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Generate text completions (non-chat)"""
        payload = {
            "model": kwargs.get("model", self.model),
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        try:
            response = await self._request_with_retry(
                "POST",
                f"{self.base_url}/v1/completions",
                json=payload,
                headers=self._build_headers(),
            )
            response.raise_for_status()
            data = response.json()
            return data.get("choices", [{}])[0].get("text", "")
        except Exception as e:
            logger.error(f"vLLM completion failed: {e}")
            return ""
