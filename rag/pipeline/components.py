"""Builds and caches the heavy shared components (models, DB clients) from Settings."""

from __future__ import annotations

from functools import cached_property

from rag.config import Settings
from rag.state import StateStore


class Components:
    def __init__(self, settings: Settings):
        self.settings = settings

    @cached_property
    def device(self) -> str:
        return self.settings.resolved_device()

    @cached_property
    def embedder(self):
        from rag.embedding import SentenceTransformerEmbedder

        return SentenceTransformerEmbedder(self.settings.embedding, self.device, self.settings.models_dir)

    @cached_property
    def store(self):
        from rag.vectorstore import ChromaStore

        return ChromaStore(self.settings.chroma_dir, self.settings.vectorstore.collection, self.settings.embedding.model)

    @cached_property
    def state(self) -> StateStore:
        return StateStore(self.settings.state_db)

    @cached_property
    def llm(self):
        from rag.llm import build_llm

        return build_llm(self.settings.llm)

    @cached_property
    def vision(self):
        from rag.llm import build_vision

        return build_vision(self.settings.vision)

    @cached_property
    def describer(self):
        from rag.parsing.figures import ImageDescriber

        enabled = self.settings.parsing.describe_images or self.settings.parsing.describe_low_text_pages
        return ImageDescriber(self.vision if enabled else None, self.state, self.settings.parsing.min_image_px)

    @cached_property
    def parsers(self):
        from rag.parsing import Parsers

        return Parsers(self.settings.parsing, self.describer)

    @cached_property
    def chunker(self):
        from rag.chunking import MarkdownChunker

        return MarkdownChunker(self.settings.chunking, self.embedder.count_tokens)

    @cached_property
    def retriever(self):
        from rag.retrieval import HybridRetriever

        return HybridRetriever(
            self.settings.retrieval, self.embedder, self.store, self.device, self.settings.models_dir
        )

    @cached_property
    def prompt_builder(self):
        from rag.prompt import PromptBuilder

        return PromptBuilder(self.settings.prompt)
