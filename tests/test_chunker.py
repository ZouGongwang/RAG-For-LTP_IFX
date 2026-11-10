from rag.chunking.markdown_chunker import MarkdownChunker, split_blocks
from rag.config import ChunkingConfig
from rag.models import ParsedDocument, Segment, SourceItem


def words(text: str) -> int:
    return len(text.split())


def item(**kw) -> SourceItem:
    return SourceItem(
        doc_id="doc:1", version="1", title="Capacity Planning", source="test",
        load=lambda: None, breadcrumb=["BSP", "Planning"], **kw,
    )


def test_split_blocks_recognises_structure():
    md = "# Title\n\nPara one\nstill para.\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\n```py\nx = 1\n\ny = 2\n```\n- item"
    kinds = [b.kind for b in split_blocks(md)]
    assert kinds == ["heading", "text", "table", "code", "text"]


def test_heading_breadcrumb_and_section_metadata():
    md = "# Overview\n\n" + "alpha " * 30 + "\n\n## Targets\n\n" + "beta " * 30
    chunks = MarkdownChunker(ChunkingConfig(max_tokens=100, min_tokens=10), words).chunk(
        item(), ParsedDocument([Segment(md)])
    )
    assert len(chunks) == 2
    assert chunks[0].text.startswith("BSP > Planning > Capacity Planning > Overview\n")
    assert chunks[1].metadata["section"] == "Overview > Targets"
    assert chunks[1].metadata["breadcrumb"] == "BSP > Planning"
    assert [c.chunk_id for c in chunks] == ["doc:1::0", "doc:1::1"]


def test_respects_max_tokens_with_overlap():
    paras = "\n\n".join(f"para{i} " + "word " * 19 for i in range(10))  # 10 x 20 words
    cfg = ChunkingConfig(max_tokens=60, overlap_ratio=0.4, min_tokens=5)
    chunks = MarkdownChunker(cfg, words).chunk(item(), ParsedDocument([Segment(paras)]))
    assert len(chunks) > 3
    header = "BSP > Planning > Capacity Planning\n\n"
    for c in chunks:
        assert words(c.text[len(header):]) <= 60
    # The last paragraph of a chunk is repeated at the start of the next one.
    first_last_para = chunks[0].text.split("\n\n")[-1]
    assert first_last_para in chunks[1].text


def test_tables_kept_whole_and_split_by_rows_with_header():
    rows = "\n".join(f"| r{i} | v{i} |" for i in range(40))
    table = "| name | value |\n|---|---|\n" + rows
    cfg = ChunkingConfig(max_tokens=50, min_tokens=5)
    chunks = MarkdownChunker(cfg, words).chunk(item(), ParsedDocument([Segment(table, kind="table")]))
    assert len(chunks) > 1
    for c in chunks:
        assert "| name | value |" in c.text
        assert c.metadata["content_type"] == "table"


def test_tables_are_compacted_and_icon_glyphs_removed():
    padded = "|  a   |   b      |\n|------|----------|\n| \uf082 1    | 2        |"
    chunks = MarkdownChunker(ChunkingConfig(), words).chunk(item(), ParsedDocument([Segment(padded, kind="table")]))
    assert chunks[0].text.endswith("| a | b |\n| --- | --- |\n| 1 | 2 |")


def test_title_not_repeated_in_breadcrumb():
    md = "# Capacity Planning\n\nIntro text here.\n\n## Scope\n\nScope text here."
    chunks = MarkdownChunker(ChunkingConfig(min_tokens=1), words).chunk(item(), ParsedDocument([Segment(md)]))
    assert chunks[0].text.startswith("BSP > Planning > Capacity Planning\n")
    assert chunks[1].text.startswith("BSP > Planning > Capacity Planning > Scope\n")


def test_pages_never_mixed():
    doc = ParsedDocument([Segment("page one text " * 3, page=1), Segment("page two text " * 3, page=2)])
    chunks = MarkdownChunker(ChunkingConfig(max_tokens=400), words).chunk(item(), doc)
    assert [c.metadata["page"] for c in chunks] == [1, 2]
