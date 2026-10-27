# BSP/IFX RAG

Ask questions over department Confluence spaces, PDFs, slides and images, with cited sources.

```
rag/
  ingestion/   sources: local folder, Confluence REST API (Cloud + Data Center), Confluence export
  parsing/     Docling (PDF/PPTX/DOCX/XLSX/images), Confluence HTML -> Markdown, image captions
  chunking/    heading-aware Markdown chunker (tables kept whole, breadcrumbs, overlap)
  embedding/   sentence-transformers (BAAI/bge-small-en-v1.5)
  vectorstore/ Chroma (upsert/delete per document)
  retrieval/   dense + BM25 hybrid, optional cross-encoder rerank
  prompt/      templates/*.md
  llm/         ollama | openai_compat | mcp  (switch in config.yaml)
  pipeline/    ingest and query orchestration
  api/         CLI (typer) and HTTP API (FastAPI)
```

## Setup

macOS: `./scripts/setup.sh` | Windows (PowerShell): `.\scripts\setup.ps1`
Add `--lite` / `-Lite` to skip Docling, then set `parsing.backend: simple`.

```
# Put LLM_API_KEY (and SSL_CERT_FILE if needed) in .env — see .env.example
python -m rag doctor          # checks LiteLLM chat + vision, embeddings, Confluence
python -m rag ingest          # incremental; --force re-parses, --rebuild starts over
python -m rag ask             # interactive; or: python -m rag ask "question" --show-context
python -m rag serve           # http://127.0.0.1:8000  corner chat; /docs for the API
python -m rag eval            # golden questions in eval/questions.yaml
```

Config lives in `config.yaml`; secrets in `.env` (copy `.env.example`). Any setting can be
overridden by env var, e.g. `RAG_LLM__OPENAI_COMPAT__MODEL=claude-sonnet-4.6`.

## Confluence

API (live): set `sources.confluence_api.enabled: true`, `deployment` and `spaces`, and put
`CONFLUENCE_URL` / `CONFLUENCE_TOKEN` (+ `CONFLUENCE_USER` for Cloud) in `.env`.
Try `page_limit: 20` first. Export (offline): unzip a space HTML export into
`data/confluence_export` and enable `sources.confluence_export`.

## Chat button on Confluence pages

The same corner button can be loaded by the wiki. A Confluence admin pastes this into
**General configuration → Custom HTML → At end of the BODY**, replacing the host with the
HTTPS address of this API (`api.cors_origins` must include the wiki origin):

```html
<script src="https://YOUR-RAG-HOST/widget.js" data-space="BEBSP"></script>
```

`data-space` limits answers to that space. Omit it to search everything indexed.
The wiki is HTTPS, so the API host must be HTTPS too. `http://127.0.0.1:8000` is blocked
by the browser on a Confluence page. Until that host exists, use `python -m rag serve`
and open `http://127.0.0.1:8000` on the laptop.

## Company LiteLLM gateway (default)

Chat answers and image/diagram captions both use the **OpenAI-compatible** LiteLLM endpoint
(`llm.provider` / `vision.provider: openai_compat`).

1. Set `LLM_API_KEY` in `.env` (and `SSL_CERT_FILE` / `REQUESTS_CA_BUNDLE` to the corporate CA if TLS fails).
2. Confirm `base_url` ends with `/v1` and `model` matches a gateway id (e.g. `gpt-5`, `claude-sonnet-4.6`).
3. `python -m rag doctor` — both **LLM** and **Vision model** checks should pass.
4. Optional: swap models via env, e.g. `RAG_VISION__OPENAI_COMPAT__MODEL=claude-sonnet-4.6`.

Embeddings stay local (`BAAI/bge-small-en-v1.5`) until you add a cloud embedder for
`qwen3-embedding-8b`. Local Ollama remains available: set `llm.provider` / `vision.provider`
back to `ollama`. MCP: fill in `llm.mcp`, run `python -m rag mcp-tools`, then `llm.provider: mcp`.

## Windows test checklist

1. Copy `documents/` over (it is no longer in git), run `.\scripts\setup.ps1`.
2. Behind a proxy: set `HTTPS_PROXY`, or copy `data/models` from a machine that already
   downloaded the models (`paths.models_dir`).
3. `python -m rag doctor` all OK (set `device: cpu` if there is no NVIDIA GPU).
4. `python -m rag ingest`, then run it again: second run must report `unchanged=4`.
5. `python -m rag eval` hit rate matches the Mac run.
6. `python -m rag serve` and try a query at http://127.0.0.1:8000/docs.
7. `python -m pytest`.
