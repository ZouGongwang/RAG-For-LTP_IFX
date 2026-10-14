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
ollama pull deepseek-r1:8b
ollama pull llava:7b          # image / diagram descriptions
python -m rag doctor          # checks models and connections
python -m rag ingest          # incremental; --force re-parses, --rebuild starts over
python -m rag ask             # interactive; or: python -m rag ask "question" --show-context
python -m rag serve           # http://127.0.0.1:8000/docs  (POST /query, POST /ingest)
python -m rag eval            # golden questions in eval/questions.yaml
```

Config lives in `config.yaml`; secrets in `.env` (copy `.env.example`). Any setting can be
overridden by env var, e.g. `RAG_LLM__OLLAMA__MODEL=llama3.2`.

## Confluence

API (live): set `sources.confluence_api.enabled: true`, `deployment` and `spaces`, and put
`CONFLUENCE_URL` / `CONFLUENCE_TOKEN` (+ `CONFLUENCE_USER` for Cloud) in `.env`.
Try `page_limit: 20` first. Export (offline): unzip a space HTML export into
`data/confluence_export` and enable `sources.confluence_export`.

## Switching to the company cloud LLM

* MCP: fill in `llm.mcp` (url, auth token in `.env`), run `python -m rag mcp-tools` to see the
  tool's argument names, map them (`tool_name`, `prompt_arg`, ...), then `llm.provider: mcp`.
* OpenAI/Azure-style endpoint: fill in `llm.openai_compat`, then `llm.provider: openai_compat`.

## Windows test checklist

1. Copy `documents/` over (it is no longer in git), run `.\scripts\setup.ps1`.
2. Behind a proxy: set `HTTPS_PROXY`, or copy `data/models` from a machine that already
   downloaded the models (`paths.models_dir`).
3. `python -m rag doctor` all OK (set `device: cpu` if there is no NVIDIA GPU).
4. `python -m rag ingest`, then run it again: second run must report `unchanged=4`.
5. `python -m rag eval` hit rate matches the Mac run.
6. `python -m rag serve` and try a query at http://127.0.0.1:8000/docs.
7. `python -m pytest`.
