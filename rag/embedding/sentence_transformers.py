from __future__ import annotations

import logging
from pathlib import Path

from rag.config import EmbeddingConfig

log = logging.getLogger(__name__)


class SentenceTransformerEmbedder:
    def __init__(self, cfg: EmbeddingConfig, device: str = "cpu", cache_dir: Path | None = None):
        from sentence_transformers import SentenceTransformer

        log.info("Loading embedding model %s on %s", cfg.model, device)
        self.cfg = cfg
        self.name = cfg.model
        self._model = SentenceTransformer(
            cfg.model, device=device, cache_folder=str(cache_dir) if cache_dir else None
        )
        self._tokenizer = self._model.tokenizer

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vecs = self._model.encode(
            texts,
            batch_size=self.cfg.batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 64,
        )
        return vecs.tolist()

    def embed_query(self, text: str) -> list[float]:
        vec = self._model.encode([self.cfg.query_prefix + text], normalize_embeddings=True)
        return vec[0].tolist()

    def count_tokens(self, text: str) -> int:
        # Counting only (chunks are sized below the model limit), so no truncation warning.
        return len(self._tokenizer(text, add_special_tokens=False, verbose=False)["input_ids"])
