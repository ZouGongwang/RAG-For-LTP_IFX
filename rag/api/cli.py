"""Command-line interface: ``python -m rag --help`` (or ``rag --help`` after ``pip install -e .``)."""

from __future__ import annotations

import json
import logging
import os
import warnings
from typing import Optional

import typer

from rag.config import get_settings

app = typer.Typer(add_completion=False, help="BSP/IFX RAG: ingest documents and ask questions.")


def _setup_logging() -> None:
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("httpx", "chromadb", "urllib3", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    for very_noisy in ("docling", "docling_core", "huggingface_hub"):
        logging.getLogger(very_noisy).setLevel(logging.ERROR)
    # RapidOCR attaches its own handler and resets its level, so disable it outright.
    logging.getLogger("RapidOCR").disabled = True
    warnings.filterwarnings("ignore", module=r"docling_core\..*")
    warnings.filterwarnings("ignore", message=".*unauthenticated requests to the HF Hub.*")
    try:
        from huggingface_hub.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
    except Exception:
        pass
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _components():
    from rag.pipeline.components import Components

    _setup_logging()
    return Components(get_settings())


@app.command()
def ingest(
    source: list[str] = typer.Option(
        None, "--source", "-s", help="local, confluence_api or confluence_export (repeatable). Default: all enabled."
    ),
    force: bool = typer.Option(False, help="Re-process documents even if unchanged."),
    rebuild: bool = typer.Option(False, help="Drop the index and rebuild everything."),
):
    """Parse, chunk, embed and index documents (incremental by default)."""
    from rag.pipeline import IngestPipeline

    stats = IngestPipeline(_components()).run(source or None, force=force, rebuild=rebuild)
    typer.echo(f"\nIngest finished: {stats.summary()}")
    for err in stats.errors:
        typer.echo(f"  error: {err}", err=True)
    if stats.errors:
        raise typer.Exit(code=1)


@app.command()
def ask(
    question: Optional[str] = typer.Argument(None, help="Question. Omit for interactive mode."),
    space: Optional[str] = typer.Option(None, help="Only search this Confluence space key."),
    top_k: Optional[int] = typer.Option(None, help="Number of passages given to the LLM."),
    show_context: bool = typer.Option(False, help="Print the retrieved passages."),
):
    """Ask a question (one-off, or interactive when no question is given)."""
    from rag.pipeline import QueryPipeline

    c = _components()
    pipeline = QueryPipeline(c)
    typer.echo(f"Ready: {c.store.count()} chunks indexed, LLM = {c.llm.name}")
    filters = {"space": space} if space else None

    def run(q: str) -> None:
        stream, used = pipeline.ask_stream(q, filters=filters, top_k=top_k)
        if show_context:
            for i, ch in enumerate(used, 1):
                typer.echo(f"\n--- [{i}] {ch.label()} (score {ch.score:.3f})\n{ch.text[:600]}")
        typer.echo("\nAnswer:")
        parts = []
        for piece in stream:
            parts.append(piece)
            typer.echo(piece, nl=False)
        answer = "".join(parts)
        typer.echo("\n\nSources:")
        for src in pipeline.select_sources(answer, used):
            url = src.metadata.get("url", "")
            typer.echo(f"  - {src.label()}" + (f"\n    {url}" if url else ""))

    if question:
        run(question)
        return
    typer.echo("Type 'quit' or 'exit' to stop.")
    while True:
        try:
            q = input("\nYour question: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in {"quit", "exit"}:
            break
        if q:
            run(q)


@app.command()
def serve(host: Optional[str] = None, port: Optional[int] = None, reload: bool = False):
    """Start the HTTP API (POST /query, POST /ingest, GET /health)."""
    import uvicorn

    _setup_logging()
    s = get_settings()
    uvicorn.run("rag.api.server:app", host=host or s.api.host, port=port or s.api.port, reload=reload)


@app.command()
def status():
    """Show what is indexed."""
    c = _components()
    docs = c.state.list_documents()
    typer.echo(f"{c.store.count()} chunks in collection '{c.settings.vectorstore.collection}', {len(docs)} documents:")
    for doc_id, source, _version, title, n_chunks, _ts in docs:
        typer.echo(f"  [{source}] {title}  ({n_chunks} chunks)  {doc_id}")


@app.command("eval")
def evaluate(
    file: str = typer.Option("eval/questions.yaml", help="YAML list of questions with expectations."),
    retrieval_only: bool = typer.Option(False, help="Skip the LLM; only check retrieval hits."),
):
    """Run the golden question set and report retrieval hit rate (and answer checks)."""
    import yaml

    from rag.pipeline import QueryPipeline

    c = _components()
    cases = yaml.safe_load(c.settings.resolve(file).read_text(encoding="utf-8")) or []
    pipeline = QueryPipeline(c)
    hits = answer_ok = answer_total = 0
    for case in cases:
        q = case["question"]
        chunks = c.retriever.retrieve(q)
        want = (case.get("expect_source") or "").lower()
        hit = not want or any(
            want in f"{ch.metadata.get('title', '')} {ch.metadata.get('doc_id', '')}".lower() for ch in chunks
        )
        hits += hit
        line = f"[{'HIT ' if hit else 'MISS'}] {q}"
        if not retrieval_only:
            ans = pipeline.ask(q).answer
            expected = [s.lower() for s in case.get("expect_answer", [])]
            if expected:
                answer_total += 1
                ok = all(s in ans.lower() for s in expected)
                answer_ok += ok
                line += f"  answer:{'ok' if ok else 'missing ' + str(expected)}"
        typer.echo(line)
    if cases:
        typer.echo(f"\nRetrieval hit rate: {hits}/{len(cases)} ({100 * hits / len(cases):.0f}%)")
    if answer_total:
        typer.echo(f"Answer checks passed: {answer_ok}/{answer_total}")


@app.command("mcp-tools")
def mcp_tools():
    """List the tools offered by the configured MCP server (to map llm.mcp.* in config)."""
    from rag.llm.mcp_client import list_tools

    _setup_logging()
    for tool in list_tools(get_settings().llm.mcp):
        typer.echo(json.dumps(tool, indent=2))


@app.command()
def doctor():
    """Check configuration, models and connections."""
    _setup_logging()
    s = get_settings()
    ok = True

    def check(label: str, fn) -> None:
        nonlocal ok
        try:
            detail = fn()
            typer.echo(f"[ OK ] {label}" + (f": {detail}" if detail else ""))
        except Exception as exc:
            ok = False
            typer.echo(f"[FAIL] {label}: {type(exc).__name__}: {exc}")

    from rag.pipeline.components import Components

    c = Components(s)
    typer.echo(f"Config: {s.root / 'config.yaml'} | device: {c.device} | data: {s.data_dir}")
    check("Embedding model", lambda: f"{s.embedding.model}, {c.embedder.count_tokens('hello world')} tokens for 'hello world'")
    check("Vector store", lambda: f"{c.store.count()} chunks")
    check("LLM", lambda: f"{c.llm.name} -> {c.llm.generate([{'role': 'user', 'content': 'Reply with the single word OK.'}])[:60]!r}")
    if s.vision.enabled:
        def vision_check():
            from PIL import Image
            import io

            img = Image.new("RGB", (200, 200), "white")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return f"{c.vision.name} -> {c.vision.describe(buf.getvalue(), 'What colour is this image? One word.')[:60]!r}"

        check("Vision model", vision_check)
    if s.parsing.backend == "docling":
        check("Docling", lambda: __import__("docling") and "installed")
    if s.sources.confluence_api.enabled:
        from rag.ingestion.confluence_client import connect

        def confluence_check():
            api = connect(s.sources.confluence_api)
            spaces = s.sources.confluence_api.spaces
            if not spaces:
                raise RuntimeError("sources.confluence_api.spaces is empty")
            return ", ".join(f"{k}: {api.space_name(k)}" for k in spaces)

        check("Confluence API", confluence_check)
    raise typer.Exit(code=0 if ok else 1)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
