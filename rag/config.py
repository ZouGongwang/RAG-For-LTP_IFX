"""Settings loaded from config.yaml, overridable by environment variables.

Precedence (highest first): explicit kwargs > environment (RAG_ prefix, "__" for nesting,
e.g. RAG_LLM__PROVIDER=mcp) > .env file > config.yaml > defaults below.

Secrets (tokens, API keys) are never stored in config.yaml; config only names the
environment variable that holds them (e.g. ``token_env: CONFLUENCE_TOKEN``).
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _config_path() -> Path:
    return Path(os.environ.get("RAG_CONFIG", PROJECT_ROOT / "config.yaml")).resolve()


class PathsConfig(BaseModel):
    data_dir: Path = Path("data")
    models_dir: Path | None = Path("data/models")


class LocalSourceConfig(BaseModel):
    enabled: bool = True
    path: Path = Path("documents")


class ConfluenceAPIConfig(BaseModel):
    enabled: bool = False
    deployment: Literal["cloud", "datacenter"] = "datacenter"
    url_env: str = "CONFLUENCE_URL"
    token_env: str = "CONFLUENCE_TOKEN"
    user_env: str = "CONFLUENCE_USER"
    spaces: list[str] = Field(default_factory=list)
    include_attachments: bool = True
    max_attachment_mb: int = 50
    verify_ssl: bool = True
    page_limit: int | None = None


class ConfluenceExportConfig(BaseModel):
    enabled: bool = False
    path: Path = Path("data/confluence_export")
    space: str = ""


class SourcesConfig(BaseModel):
    local: LocalSourceConfig = LocalSourceConfig()
    confluence_api: ConfluenceAPIConfig = ConfluenceAPIConfig()
    confluence_export: ConfluenceExportConfig = ConfluenceExportConfig()


class ParsingConfig(BaseModel):
    backend: Literal["docling", "simple"] = "docling"
    ocr: bool = True
    table_structure: bool = True
    describe_images: bool = True
    describe_low_text_pages: bool = True
    low_text_threshold: int = 100
    min_image_px: int = 120
    image_scale: float = 2.0
    docling_artifacts_path: Path | None = None


class ChunkingConfig(BaseModel):
    max_tokens: int = 400
    overlap_ratio: float = 0.15
    min_tokens: int = 25


class EmbeddingConfig(BaseModel):
    model: str = "BAAI/bge-small-en-v1.5"
    query_prefix: str = "Represent this sentence for searching relevant passages: "
    batch_size: int = 32


class VectorStoreConfig(BaseModel):
    collection: str = "bsp_docs"


class RerankConfig(BaseModel):
    enabled: bool = False
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    candidates: int = 20


class RetrievalConfig(BaseModel):
    dense_k: int = 20
    bm25_k: int = 20
    top_k: int = 5
    rrf_k: int = 60
    rerank: RerankConfig = RerankConfig()


class PromptConfig(BaseModel):
    template: str = "answer.md"
    max_context_chars: int = 12000
    display_sources: int = 3


class OllamaConfig(BaseModel):
    model: str = "deepseek-r1:8b"
    host: str = "http://localhost:11434"
    temperature: float = 0.1
    num_ctx: int = 8192
    timeout_s: float = 300


class OpenAICompatConfig(BaseModel):
    """Empty fields under ``vision.openai_compat`` inherit from ``llm.openai_compat``."""

    base_url: str = ""
    model: str = ""
    api_key_env: str = ""
    # None = don't send it (some models, e.g. GPT-5 reasoning models, reject non-default values).
    temperature: float | None = None
    timeout_s: float = 120


class MCPConfig(BaseModel):
    """How to reach the company LLM through an MCP server. Fill in once IT shares details."""

    transport: Literal["streamable_http", "sse", "stdio"] = "streamable_http"
    url: str = ""
    command: list[str] = Field(default_factory=list)
    headers: dict[str, str] = Field(default_factory=dict)
    auth_token_env: str = "MCP_AUTH_TOKEN"
    tool_name: str = "generate"
    argument_style: Literal["prompt", "messages"] = "prompt"
    prompt_arg: str = "prompt"
    system_arg: str | None = None
    messages_arg: str = "messages"
    image_arg: str = "image_base64"
    extra_args: dict[str, Any] = Field(default_factory=dict)
    timeout_s: float = 120


class LLMConfig(BaseModel):
    provider: Literal["ollama", "openai_compat", "mcp"] = "ollama"
    ollama: OllamaConfig = OllamaConfig()
    openai_compat: OpenAICompatConfig = OpenAICompatConfig()
    mcp: MCPConfig = MCPConfig()


class VisionConfig(BaseModel):
    enabled: bool = True
    provider: Literal["ollama", "openai_compat", "mcp"] = "ollama"
    ollama: OllamaConfig = OllamaConfig(model="llava:7b")
    openai_compat: OpenAICompatConfig = OpenAICompatConfig()
    mcp: MCPConfig = MCPConfig()


class APIConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000
    # Browsers on these sites may call /query. Confluence is HTTPS, so the API must be HTTPS too.
    cors_origins: list[str] = Field(default_factory=list)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_nested_delimiter="__",
        env_file=PROJECT_ROOT / ".env",
        extra="ignore",
    )

    device: Literal["auto", "cpu", "mps", "cuda"] = "auto"
    log_level: str = "INFO"
    paths: PathsConfig = PathsConfig()
    sources: SourcesConfig = SourcesConfig()
    parsing: ParsingConfig = ParsingConfig()
    chunking: ChunkingConfig = ChunkingConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    vectorstore: VectorStoreConfig = VectorStoreConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    prompt: PromptConfig = PromptConfig()
    llm: LLMConfig = LLMConfig()
    vision: VisionConfig = VisionConfig()
    api: APIConfig = APIConfig()

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        yaml_source = YamlConfigSettingsSource(settings_cls, yaml_file=_config_path())
        return init_settings, env_settings, dotenv_settings, yaml_source

    @property
    def root(self) -> Path:
        return _config_path().parent

    def resolve(self, path: Path | str) -> Path:
        """Resolve a config path relative to the folder containing config.yaml."""
        p = Path(path).expanduser()
        return p if p.is_absolute() else (self.root / p).resolve()

    @property
    def data_dir(self) -> Path:
        return self.resolve(self.paths.data_dir)

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def state_db(self) -> Path:
        return self.data_dir / "state.sqlite"

    @property
    def models_dir(self) -> Path | None:
        return self.resolve(self.paths.models_dir) if self.paths.models_dir else None

    def resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            if torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    load_dotenv(PROJECT_ROOT / ".env")
    settings = Settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    if settings.models_dir:
        settings.models_dir.mkdir(parents=True, exist_ok=True)
        # Keeps Hugging Face / Docling model downloads inside the project so the folder
        # can be copied to a machine behind a proxy.
        os.environ.setdefault("HF_HOME", str(settings.models_dir))
    return settings


def secret(env_name: str, required: bool = True) -> str:
    value = os.environ.get(env_name, "").strip()
    if required and not value:
        raise RuntimeError(f"Environment variable {env_name} is not set (add it to .env).")
    return value
