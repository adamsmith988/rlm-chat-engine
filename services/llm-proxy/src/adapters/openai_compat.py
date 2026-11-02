"""
OpenAI Compatible Adapter
آداپتور سازگار با OpenAI برای DeepSeek و سایر سرویس‌ها
"""

import json
import time
import logging
from typing import Any, AsyncIterator, Dict, List, Optional, Union

import httpx

from .base import BaseLLMAdapter, ChatMessage, ChatResponse, extract_message_reasoning

logger = logging.getLogger(__name__)


class OpenAICompatAdapter(BaseLLMAdapter):
    """
    Adapter for OpenAI-compatible APIs
    آداپتور برای APIهای سازگار با OpenAI
    
    Supports:
    - OpenAI API
    - DeepSeek API
    - Azure OpenAI
    - Together AI
    - Groq
    - Any OpenAI-compatible endpoint
    """
    
    name = "openai_compat"
    
    def __init__(
        self,
        base_url: str = "https://api.openai.com",
        model: str = "",
        api_key: Optional[str] = None,
        timeout: float = 120.0,
        provider: str = "openai",
        chat_path: str = "/v1/chat/completions",
        models_path: Optional[str] = "/v1/models",
        **kwargs
    ):
        super().__init__(base_url, model, api_key=api_key, timeout=timeout, **kwargs)
        self.provider = provider
        self.chat_path = self._norm_path(chat_path) or "/v1/chat/completions"
        # None / "" → skip /v1/models health (chat-only providers)
        if models_path is None:
            self.models_path: Optional[str] = None
        else:
            mp = self._norm_path(models_path)
            self.models_path = mp or None

    @staticmethod
    def _norm_path(path: Optional[str]) -> str:
        raw = str(path or "").strip()
        if not raw:
            return ""
        if not raw.startswith("/"):
            raw = "/" + raw
        return raw

    def _effective_chat_path(self, kwargs: Dict[str, Any]) -> str:
        """Prefer per-request ResolvedTarget.endpoint_path when it is a chat path."""
        override = kwargs.pop("endpoint_path", None)
        path = self._norm_path(str(override or ""))
        if path and "chat" in path.lower():
            return path
        return self.chat_path

    def _url(self, path: str) -> str:
        return f"{self.base_url.rstrip('/')}{self._norm_path(path)}"
    
    def _build_headers(self) -> Dict[str, str]:
        """Build headers with proper authorization"""
        headers = {
            "Content-Type": "application/json"
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers
    
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
        """Send chat request to OpenAI-compatible API"""
        start_time = time.time()
        
        # Prepare messages
        msg_list = [m.to_dict() if isinstance(m, ChatMessage) else m for m in messages]
        
        # Build payload
        payload = {
            "model": kwargs.get("model", self.model),
            "messages": msg_list,
            "temperature": temperature
        }
        
        if max_tokens:
            payload["max_tokens"] = max_tokens
        
        if tools:
            payload["tools"] = tools
            if tool_choice:
                payload["tool_choice"] = tool_choice
        
        if response_format:
            payload["response_format"] = response_format
        
        # Provider-specific options
        if self.provider == "deepseek":
            # DeepSeek specific options
            if "presence_penalty" in kwargs:
                payload["presence_penalty"] = kwargs["presence_penalty"]
            if "frequency_penalty" in kwargs:
                payload["frequency_penalty"] = kwargs["frequency_penalty"]
        
        # Common additional options
        if "top_p" in kwargs:
            payload["top_p"] = kwargs["top_p"]
        if "stop" in kwargs:
            payload["stop"] = kwargs["stop"]
        if "seed" in kwargs:
            payload["seed"] = kwargs["seed"]

        # vLLM / Qwen3 (KServe OpenAI runtime): must forward so thinking is off
        # and JSON lands in `content` instead of chain-of-thought prose.
        ct_kw = kwargs.get("chat_template_kwargs")
        if isinstance(ct_kw, dict) and ct_kw:
            payload["chat_template_kwargs"] = ct_kw
        
        chat_url = self._url(self._effective_chat_path(kwargs))
        try:
            response = await self._request_with_retry(
                "POST",
                chat_url,
                json=payload,
                headers=self._build_headers(),
            )
            response.raise_for_status()
            data = response.json()
            
            choice = data.get("choices", [{}])[0]
            message = choice.get("message", {})
            raw_c = message.get("content")
            text_c = "" if raw_c is None else (raw_c if isinstance(raw_c, str) else str(raw_c))
            r = extract_message_reasoning(message)
            reasoning_text = r

            ct_kw = kwargs.get("chat_template_kwargs")
            if not isinstance(ct_kw, dict):
                ct_kw = {}
            enable_thinking = ct_kw.get("enable_thinking")

            out_reasoning: Optional[str] = None
            out_content = text_c
            if enable_thinking is True:
                out_reasoning = reasoning_text or None
            else:
                # Same merge rule as VLLMAdapter: some stacks put CoT in `reasoning`
                # and leave `content` empty when thinking is split.
                if not out_content.strip() and reasoning_text:
                    out_content = reasoning_text

            return ChatResponse(
                content=out_content,
                role=message.get("role", "assistant"),
                finish_reason=choice.get("finish_reason"),
                tool_calls=message.get("tool_calls"),
                reasoning=out_reasoning,
                usage=data.get("usage"),
                model=data.get("model", self.model),
                latency_ms=(time.time() - start_time) * 1000,
                raw_response=data
            )
                
        except httpx.HTTPStatusError as e:
            body = (e.response.text or "").strip().replace("\n", " ")[:900]
            logger.error(
                "%s API error: %s - %s",
                self.provider,
                e.response.status_code,
                body,
            )
            raise RuntimeError(
                f"HTTP {e.response.status_code} from {self._url(self.chat_path)}: "
                f"{body or str(e)}"
            ) from e
        except httpx.HTTPError as e:
            logger.error(f"{self.provider} request failed: {e}")
            raise
    
    def _build_chat_payload(
        self,
        msg_list: list,
        *,
        temperature: float,
        max_tokens: Optional[int],
        stream: bool,
        tools: Optional[List[Dict]] = None,
        tool_choice: Optional[Union[str, Dict]] = None,
        response_format: Optional[Dict] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Shared payload for chat + chat_stream (vLLM / KServe OpenAI runtime)."""
        payload: Dict[str, Any] = {
            "model": kwargs.get("model", self.model),
            "messages": msg_list,
            "temperature": temperature,
            "stream": stream,
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
        if "stop" in kwargs:
            payload["stop"] = kwargs["stop"]
        if "seed" in kwargs:
            payload["seed"] = kwargs["seed"]
        ct_kw = kwargs.get("chat_template_kwargs")
        if isinstance(ct_kw, dict) and ct_kw:
            payload["chat_template_kwargs"] = ct_kw
        return payload

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
        """Stream chat from OpenAI-compatible API (KServe / external vLLM)."""
        msg_list = [m.to_dict() if isinstance(m, ChatMessage) else m for m in messages]
        payload = self._build_chat_payload(
            msg_list,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            **kwargs,
        )

        try:
            stream_url = self._url(self._effective_chat_path(kwargs))
            async with self._client.stream(
                "POST",
                stream_url,
                json=payload,
                headers=self._build_headers(),
                timeout=self._stream_timeout(),
            ) as response:
                response.raise_for_status()
                async for line in self._iter_stream_lines(response, url=stream_url):
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        choice = (data.get("choices") or [{}])[0]
                        delta = choice.get("delta") or {}
                        content = delta.get("content") or ""
                        reasoning = (
                            delta.get("reasoning")
                            or delta.get("reasoning_content")
                            or ""
                        )
                        tool_calls = delta.get("tool_calls")
                        finish_reason = choice.get("finish_reason")
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
            logger.error(f"{self.provider} stream failed: {e}")
            raise
    
    async def health_check(self) -> Dict[str, Any]:
        """Check API health"""
        import time as _time

        t0 = _time.time()
        try:
            # Chat-only providers: no models listing — mark healthy if adapter configured.
            if not self.models_path:
                return {
                    "healthy": True,
                    "backend": self.provider,
                    "base_url": self.base_url,
                    "latency_ms": 0.0,
                    "models_count": 1 if self.model else 0,
                    "current_model": self.model,
                    "probe": "chat_only",
                    "chat_path": self.chat_path,
                }

            response = await self._client.get(
                self._url(self.models_path),
                headers=self._build_headers(),
                timeout=10.0,
            )
            latency_ms = round((_time.time() - t0) * 1000, 1)
            if response.status_code == 200:
                data = response.json()
                models = [m["id"] for m in data.get("data", [])]
                return {
                    "healthy": True,
                    "backend": self.provider,
                    "base_url": self.base_url,
                    "latency_ms": latency_ms,
                    "models_count": len(models),
                    "current_model": self.model
                }
            if response.status_code == 404:
                # TEI and some embed servers expose /health but not OpenAI /v1/models.
                health_resp = await self._client.get(
                    f"{self.base_url}/health",
                    headers=self._build_headers(),
                    timeout=10.0,
                )
                if health_resp.status_code == 200:
                    return {
                        "healthy": True,
                        "backend": self.provider,
                        "base_url": self.base_url,
                        "latency_ms": latency_ms,
                        "models_count": 1,
                        "current_model": self.model,
                        "probe": "health",
                    }
                # Chat-only APIs often 404 on models — still usable for chat.
                return {
                    "healthy": True,
                    "backend": self.provider,
                    "base_url": self.base_url,
                    "latency_ms": latency_ms,
                    "models_count": 1 if self.model else 0,
                    "current_model": self.model,
                    "probe": "models_404_assume_chat",
                    "chat_path": self.chat_path,
                }
            body_head = (response.text or "").strip().replace("\n", " ")[:200]
            return {
                "healthy": False,
                "backend": self.provider,
                "base_url": self.base_url,
                "latency_ms": latency_ms,
                "error": f"HTTP {response.status_code}" + (f": {body_head}" if body_head else ""),
            }
        except Exception as e:
            logger.warning(f"{self.provider} health check failed: {e}")
            return {
                "healthy": False,
                "backend": self.provider,
                "base_url": self.base_url,
                "error": f"{type(e).__name__}: {e}",
            }
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """List available models"""
        if not self.models_path:
            if self.model:
                return [{"id": self.model}]
            return []
        try:
            response = await self._client.get(
                self._url(self.models_path),
                headers=self._build_headers(),
                timeout=10.0,
            )
            response.raise_for_status()
            data = response.json()
            return data.get("data", [])
        except Exception as e:
            logger.error(f"Failed to list {self.provider} models: {e}")
            if self.model:
                return [{"id": self.model}]
            return []
    
    async def generate_embeddings(
        self,
        input_text: Union[str, List[str]],
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate embeddings using OpenAI embeddings API"""
        payload = {
            "model": model or "text-embedding-3-small",
            "input": input_text
        }
        
        try:
            response = await self._request_with_retry(
                "POST",
                f"{self.base_url}/v1/embeddings",
                json=payload,
                headers=self._build_headers(),
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Embeddings request failed: {e}")
            return {"error": str(e)}


class DeepSeekAdapter(OpenAICompatAdapter):
    """
    Specialized adapter for DeepSeek API
    آداپتور اختصاصی برای DeepSeek API
    """
    
    name = "deepseek"
    
    def __init__(
        self,
        base_url: str = "https://api.deepseek.com",
        model: str = "",
        api_key: Optional[str] = None,
        timeout: float = 120.0,
        **kwargs
    ):
        super().__init__(
            base_url=base_url,
            model=model,
            api_key=api_key,
            timeout=timeout,
            provider="deepseek",
            **kwargs
        )
    
    async def chat_with_reasoning(
        self,
        messages: List[ChatMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> ChatResponse:
        """Chat with DeepSeek reasoning model (R1)"""
        if not kwargs.get("model"):
            raise ValueError("model is required for chat_with_reasoning (configure via llm_routes)")
        return await self.chat(messages, temperature, max_tokens, **kwargs)


class OpenAIAdapter(OpenAICompatAdapter):
    """
    Specialized adapter for OpenAI API
    آداپتور اختصاصی برای OpenAI API
    """
    
    name = "openai"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "",
        timeout: float = 120.0,
        **kwargs
    ):
        super().__init__(
            base_url="https://api.openai.com",
            model=model,
            api_key=api_key,
            timeout=timeout,
            provider="openai",
            **kwargs
        )
