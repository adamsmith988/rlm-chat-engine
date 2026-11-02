"""
Model Manager
مدیریت و روتینگ مدل‌های LLM
"""

import json
import logging
from typing import Any, Dict, List, Optional, Type

from .adapters import (
    BaseLLMAdapter,
    TritonAdapter,
    OpenAICompatAdapter,
    DeepSeekAdapter,
    OpenAIAdapter,
    ChatMessage,
    ChatResponse,
)
from .adapters.vllm_omni import VllmOmniAdapter
from .config import get_settings

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Manager for multiple LLM backends
    مدیریت چندین بک‌اند LLM
    
    Features:
    - Dynamic adapter selection
    - Fallback routing
    - Health monitoring
    - Model discovery
    """
    
    ADAPTER_REGISTRY: Dict[str, Type[BaseLLMAdapter]] = {
        "openai": OpenAIAdapter,
        "deepseek": DeepSeekAdapter,
        "openai_compat": OpenAICompatAdapter,
    }
    
    def __init__(self):
        self.settings = get_settings()
        self.adapters: Dict[str, BaseLLMAdapter] = {}
        self.active_backend: str = self.settings.DEFAULT_BACKEND
        self.embedding_adapter: Optional[BaseLLMAdapter] = None
        self.triton_adapter: Optional[TritonAdapter] = None
        self.omni_adapter: Optional[VllmOmniAdapter] = None
        self._initialize_adapters()
    
    def _initialize_adapters(self):
        """No env-bootstrap adapters — registry_sync owns chat/embed/image/triton."""
        logger.info(
            "Env LLM adapters disabled; waiting for registry_sync from Storage"
        )

    def get_adapter(self, backend: Optional[str] = None) -> BaseLLMAdapter:
        """Get adapter by backend name"""
        backend = backend or self.active_backend
        if backend not in self.adapters:
            raise ValueError(f"Backend '{backend}' not configured. Available: {list(self.adapters.keys())}")
        return self.adapters[backend]
    
    def set_active_backend(self, backend: str) -> bool:
        """Set the active backend"""
        if backend not in self.adapters:
            return False
        self.active_backend = backend
        logger.info(f"Active backend set to: {backend}")
        return True
    
    def register_adapter(
        self,
        name: str,
        adapter_class: Type[BaseLLMAdapter],
        **kwargs
    ) -> BaseLLMAdapter:
        """Register a new adapter dynamically"""
        adapter = adapter_class(**kwargs)
        self.adapters[name] = adapter
        logger.info(f"Registered adapter: {name}")
        return adapter
    
    async def chat(
        self,
        messages: List[ChatMessage],
        backend: Optional[str] = None,
        **kwargs
    ) -> ChatResponse:
        """Route chat request to appropriate adapter"""
        adapter = self.get_adapter(backend)
        return await adapter.chat(messages, **kwargs)
    
    async def chat_stream(
        self,
        messages: List[ChatMessage],
        backend: Optional[str] = None,
        **kwargs
    ):
        """Route streaming chat request to appropriate adapter"""
        adapter = self.get_adapter(backend)
        async for chunk in adapter.chat_stream(messages, **kwargs):
            yield chunk
    
    async def health_check(self, backend: Optional[str] = None) -> Dict[str, Any]:
        """Check health of specific or all backends"""
        if backend:
            adapter = self.get_adapter(backend)
            return await adapter.health_check()
        
        results = {}
        for name, adapter in self.adapters.items():
            try:
                results[name] = await adapter.health_check()
            except Exception as e:
                results[name] = {"healthy": False, "error": str(e)}
        
        if self.embedding_adapter:
            try:
                results["vllm_embed"] = await self.embedding_adapter.health_check()
            except Exception as e:
                results["vllm_embed"] = {"healthy": False, "error": str(e)}

        if self.triton_adapter:
            try:
                results["triton"] = await self.triton_adapter.health_check()
            except Exception as e:
                results["triton"] = {"healthy": False, "error": str(e)}

        if self.omni_adapter:
            try:
                results["vllm_omni"] = await self.omni_adapter.health_check()
            except Exception as e:
                results["vllm_omni"] = {"healthy": False, "error": str(e)}

        return results
    
    async def list_models(self, backend: Optional[str] = None) -> Dict[str, List]:
        """List available models from specific or all backends"""
        if backend:
            adapter = self.get_adapter(backend)
            return {backend: await adapter.list_models()}
        
        results = {}
        for name, adapter in self.adapters.items():
            try:
                results[name] = await adapter.list_models()
            except Exception as e:
                results[name] = []
                logger.warning(f"Failed to list models from {name}: {e}")
        return results
    
    async def find_best_backend(self) -> Optional[str]:
        """Pick the first healthy backend."""
        try:
            from .health_monitor import get_health_monitor
            health = get_health_monitor()
        except Exception:
            health = None

        candidates = [self.active_backend] + [b for b in self.adapters.keys() if b != self.active_backend]
        for name in candidates:
            if not name:
                continue
            if health and not health.is_healthy(name):
                continue
            return name
        return None
    
    async def auto_select_backend(self) -> str:
        selected = await self.find_best_backend()
        if not selected:
            raise RuntimeError("no_healthy_backend")
        self.active_backend = selected
        return selected
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of model manager"""
        active_model = None
        if self.active_backend in self.adapters:
            active_model = getattr(self.adapters[self.active_backend], "model", None)
        return {
            "active_backend": self.active_backend,
            "available_backends": list(self.adapters.keys()),
            "active_model": active_model,
        }

    async def close(self) -> None:
        """Close all adapter HTTP clients. Call on shutdown."""
        for name, adapter in self.adapters.items():
            if hasattr(adapter, "close"):
                try:
                    await adapter.close()
                except Exception as e:
                    logger.warning(f"Error closing adapter {name}: {e}")
        if self.embedding_adapter and hasattr(self.embedding_adapter, "close"):
            try:
                await self.embedding_adapter.close()
            except Exception:
                pass
        if self.triton_adapter and hasattr(self.triton_adapter, "close"):
            try:
                await self.triton_adapter.close()
            except Exception:
                pass
        if self.omni_adapter:
            try:
                await self.omni_adapter.close()
            except Exception as e:
                logger.warning("Error closing vLLM-Omni adapter: %s", e)
        logger.info("All adapter HTTP clients closed.")


_model_manager: Optional[ModelManager] = None


def get_model_manager() -> ModelManager:
    """Get singleton ModelManager instance"""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager


async def init_model_manager() -> ModelManager:
    """Initialize model manager with auto-selection when legacy adapters exist."""
    manager = get_model_manager()
    if manager.adapters:
        try:
            await manager.auto_select_backend()
        except RuntimeError:
            logger.warning("No healthy backend found during initialization")
    return manager
