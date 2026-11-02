"""
LLM Proxy Configuration
تنظیمات سرویس پروکسی LLM
"""

import sys
from pathlib import Path
from typing import Optional, List

from pydantic_settings import BaseSettings

_REPO_ROOT = Path(__file__).resolve().parents[3]
_RUNTIME_CFG = _REPO_ROOT / "packages" / "runtime-config" / "src"
if _RUNTIME_CFG.is_dir() and str(_RUNTIME_CFG) not in sys.path:
    sys.path.insert(0, str(_RUNTIME_CFG))
try:
    from runtime_config import apply_settings_to_environ
    apply_settings_to_environ()
except Exception:
    pass


class Settings(BaseSettings):
    """
    Configuration for LLM Proxy Service
    تنظیمات سرویس پروکسی LLM
    """
    
    # Default backend (vllm, openai, deepseek)
    DEFAULT_BACKEND: str = "vllm"
    
    # Prefer admin-registered endpoints in Storage (registry_sync).
    STRICT_LLM_BINDINGS: bool = True

    # Optional timeouts for adapters created by registry_sync (not env bootstrap).
    VLLM_API_KEY: str = ""  # unused for env bootstrap; kept for rare local overrides
    VLLM_EMBED_API_KEY: str = ""

    # Triton Inference Server settings (ONNX models on GPU) — used when registry
    # creates triton adapters; URL comes from endpoint rows, not this env.
    # Must cover cold-start + queue under load; 30s caused dinov2 ReadTimeout.
    TRITON_URL: str = "http://localhost:8009"
    TRITON_TIMEOUT: float = 300.0

    # vLLM embedding server timeout floor for queue/cold start.
    VLLM_EMBED_TIMEOUT: float = 300.0

    # vLLM-Omni (diffusion / text-to-image) — optional env seed only when set.
    # Docs (repo): services/llm-proxy/vllm-omni/README.md
    VLLM_OMNI_BASE_URL: str = ""
    VLLM_OMNI_API_KEY: str = ""
    VLLM_OMNI_TIMEOUT: float = 600.0
    
    # OpenAI settings (model ids come from llm_routes / registry — not env defaults)
    OPENAI_BASE_URL: str = "https://api.openai.com/v1"
    OPENAI_API_KEY: str = ""
    
    # DeepSeek settings
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com/v1"
    DEEPSEEK_API_KEY: str = ""
    
    # Local model settings
    LOCAL_MODELS_DIR: str = "./models"
    
    # Request settings
    DEFAULT_TIMEOUT: float = 540.0
    DEFAULT_TEMPERATURE: float = 0.7
    DEFAULT_MAX_TOKENS: int = 4096
    
    # Retry settings
    MAX_RETRIES: int = 3
    RETRY_DELAY: float = 1.0
    
    # Rate limiting
    RATE_LIMIT_ENABLED: bool = False
    RATE_LIMIT_RPM: int = 60  # Requests per minute
    
    # Streaming
    STREAM_ENABLED: bool = True
    # Retry streaming chat on the next registry candidate when the primary
    # fails before any payload chunk is sent (external → KServe rollover).
    STREAM_FAILOVER_ENABLED: bool = True

    # ─── GPU Resource Management (centralized) ──────────────────
    GPU_TOTAL_MB: int = 0             # 0 = auto-detect
    GPU_RESERVED_VLLM_MB: int = 8000  # VRAM reserved for vLLM/Qwen
    # VRAM held by vLLM-Omni process (long-lived); subtract from tool-runner budget
    GPU_RESERVED_OMNI_MB: int = 0
    GPU_SAFETY_MARGIN_MB: int = 512
    GPU_MAX_CONCURRENT: int = 2       # Max concurrent GPU tool executions
    GPU_IDLE_TIMEOUT_SEC: int = 300   # Evict models after 5 min idle

    # Tool Runner URL (for remote model cleanup callbacks)
    TOOL_RUNNER_URL: str = "http://tool-runner:8002"
    # Internal auth between llm-gateway <-> tool-runner
    INTERNAL_API_KEY: str = "aiv2-internal-service-key-change-in-production"

    # Storage URL (to fetch super-admin routing config)
    STORAGE_URL: str = "http://storage:8004"

    # Shared secret for mother→agent /health probes (same as api-gateway).
    REMOTE_NODE_TOKEN: str = ""

    # Optional: bootstrap additional OpenAI-compatible endpoints (JSON list)
    # Example:
    #   [{"name":"kserve-qwen","base_url":"http://qwen-chat-predictor.aiv2.svc:80/v1","api_key":"","model":"qwen-chat"}]
    EXTRA_OPENAI_COMPAT_ENDPOINTS_JSON: str = ""

    # Logging
    LOG_REQUESTS: bool = True
    LOG_RESPONSES: bool = False  # Can be verbose
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Singleton pattern for settings
_settings = None


def get_settings() -> Settings:
    """Get settings singleton"""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


# For backward compatibility
settings = get_settings()
