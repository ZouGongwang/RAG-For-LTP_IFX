"""Turn images (figures, diagrams, scanned pages, Confluence attachments) into searchable text."""

from __future__ import annotations

import hashlib
import io
import logging
from pathlib import Path

from rag.llm.base import VisionClient
from rag.state import StateStore

log = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompt" / "templates"


def _load_prompt(name: str) -> str:
    return (PROMPTS_DIR / name).read_text(encoding="utf-8").strip()


class ImageDescriber:
    """Caches captions by image hash so re-ingesting unchanged images costs nothing."""

    def __init__(self, vision: VisionClient | None, state: StateStore, min_px: int = 120):
        self.vision = vision
        self.state = state
        self.min_px = min_px
        self._disabled_reason: str | None = None if vision else "vision disabled"
        self._figure_prompt = _load_prompt("describe_image.md")
        self._page_prompt = _load_prompt("describe_page.md")

    @property
    def available(self) -> bool:
        return self._disabled_reason is None

    def describe(self, image, *, whole_page: bool = False, context: str = "") -> str | None:
        """``image`` may be PNG/JPEG bytes or a PIL image. Returns None if skipped."""
        if not self.available:
            return None
        png = self._to_png(image)
        if png is None:
            return None

        image_hash = hashlib.sha256(png).hexdigest()
        cached = self.state.get_caption(image_hash, self.vision.name)
        if cached is not None:
            return cached or None

        prompt = self._page_prompt if whole_page else self._figure_prompt
        if context:
            prompt = f"{prompt}\n\nContext: this image appears in \"{context}\"."
        try:
            caption = self.vision.describe(png, prompt).strip()
        except Exception as exc:  # model missing, server down, ...
            self._disabled_reason = str(exc)
            log.warning("Image description disabled for this run: %s", exc)
            return None
        self.state.set_caption(image_hash, self.vision.name, caption)
        return caption or None

    def _to_png(self, image) -> bytes | None:
        from PIL import Image

        try:
            img = image if isinstance(image, Image.Image) else Image.open(io.BytesIO(image))
            if min(img.size) < self.min_px:
                return None  # icons, bullets, logos
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()
        except Exception as exc:
            log.debug("Unreadable image skipped: %s", exc)
            return None
