from __future__ import annotations

from rag.config import LLMConfig, VisionConfig
from rag.llm.base import LLMClient, VisionClient


def build_llm(cfg: LLMConfig) -> LLMClient:
    if cfg.provider == "ollama":
        from rag.llm.ollama_client import OllamaLLM

        return OllamaLLM(cfg.ollama)
    if cfg.provider == "openai_compat":
        from rag.llm.openai_compat import OpenAICompatLLM

        return OpenAICompatLLM(cfg.openai_compat)
    if cfg.provider == "mcp":
        from rag.llm.mcp_client import MCPLLM

        return MCPLLM(cfg.mcp)
    raise ValueError(f"Unknown LLM provider: {cfg.provider}")


def build_vision(cfg: VisionConfig, llm_cfg: LLMConfig | None = None) -> VisionClient | None:
    if not cfg.enabled:
        return None
    if cfg.provider == "ollama":
        from rag.llm.ollama_client import OllamaVision

        return OllamaVision(cfg.ollama)
    if cfg.provider == "openai_compat":
        from rag.llm.openai_compat import OpenAICompatVision

        own = cfg.openai_compat
        if llm_cfg is not None:
            base = llm_cfg.openai_compat
            own = own.model_copy(update={
                "base_url": own.base_url or base.base_url,
                "model": own.model or base.model,
                "api_key_env": own.api_key_env or base.api_key_env,
            })
        return OpenAICompatVision(own)
    if cfg.provider == "mcp":
        from rag.llm.mcp_client import MCPVision

        return MCPVision(cfg.mcp)
    raise ValueError(f"Unknown vision provider: {cfg.provider}")
