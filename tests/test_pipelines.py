from pathlib import Path

from rag.pipeline import IngestPipeline, QueryPipeline


def write_docs(d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / "capacity.md").write_text(
        "# Capacity Planning\n\n## Targets\n\nThe VRFC2506 capacity target is 42 wafers per week.\n",
        encoding="utf-8",
    )
    (d / "sub").mkdir(exist_ok=True)
    (d / "sub" / "faq.txt").write_text("Scenario enquiries go to the planning team mailbox.", encoding="utf-8")


def test_incremental_ingest_and_query(tmp_path, make_components):
    docs = tmp_path / "docs"
    write_docs(docs)
    c = make_components(docs)

    s1 = IngestPipeline(c).run()
    assert (s1.added, s1.failed) == (2, 0)
    assert c.store.count() == 2

    s2 = IngestPipeline(c).run()
    assert (s2.added, s2.updated, s2.unchanged) == (0, 0, 2)

    (docs / "sub" / "faq.txt").write_text("Updated: enquiries go to the BSP desk.", encoding="utf-8")
    (docs / "capacity.md").unlink()
    s3 = IngestPipeline(c).run()
    assert (s3.updated, s3.deleted, s3.unchanged) == (1, 1, 0)
    assert c.store.count() == 1

    (docs / "capacity.md").write_text("# Capacity\n\nVRFC2506 target is 42 wafers.", encoding="utf-8")
    IngestPipeline(c).run()

    answer = QueryPipeline(c).ask("What is the VRFC2506 capacity target?")
    assert answer.answer == "The capacity target is 42 wafers [1]."
    assert answer.sources[0].metadata["doc_id"] == "local:capacity.md"
    assert answer.sources[0].metadata["breadcrumb"] == ""
    assert "VRFC2506" in c.llm.last_messages[1]["content"]


def test_rebuild_clears_everything(tmp_path, make_components):
    docs = tmp_path / "docs"
    write_docs(docs)
    c = make_components(docs)
    IngestPipeline(c).run()
    stats = IngestPipeline(c).run(rebuild=True)
    assert stats.added == 2 and c.store.count() == 2


class RewritingLLM:
    """Answers the rewrite prompt with a standalone question and records every call."""

    name = "rewriting-llm"

    def __init__(self, rewrite=None, fail=False):
        self.rewrite, self.fail, self.calls = rewrite, fail, []

    def generate(self, messages):
        self.calls.append(messages)
        if messages[0]["content"].startswith("Rewrite the final question"):
            if self.fail:
                raise RuntimeError("gateway down")
            return self.rewrite
        return "It is 42 wafers [1]."

    def stream(self, messages):
        yield self.generate(messages)


HISTORY = [
    {"role": "user", "content": "Tell me about the VRFC2506 plan"},
    {"role": "assistant", "content": "## Plan\n- It is a capacity plan [1]."},
]


def test_follow_up_is_rewritten_for_retrieval_and_sees_history(tmp_path, make_components):
    docs = tmp_path / "docs"
    write_docs(docs)
    c = make_components(docs)
    IngestPipeline(c).run()
    c.llm = RewritingLLM(rewrite="VRFC2506 capacity target wafers per week")

    answer = QueryPipeline(c).ask("and how much is it?", history=HISTORY)

    rewrite_call, answer_call = c.llm.calls
    assert "User: Tell me about the VRFC2506 plan" in rewrite_call[0]["content"]
    assert "[1]" not in rewrite_call[0]["content"]
    assert [m["role"] for m in answer_call] == ["system", "user", "assistant", "user"]
    assert "42 wafers per week" in answer_call[-1]["content"]  # found through the rewritten query
    assert "Question: and how much is it?" in answer_call[-1]["content"]
    assert answer.sources


def test_first_question_makes_no_rewrite_call(tmp_path, make_components):
    docs = tmp_path / "docs"
    write_docs(docs)
    c = make_components(docs)
    IngestPipeline(c).run()
    c.llm = RewritingLLM()

    QueryPipeline(c).ask("What is the VRFC2506 capacity target?")

    assert len(c.llm.calls) == 1


def test_follow_up_survives_a_failed_rewrite(tmp_path, make_components):
    docs = tmp_path / "docs"
    write_docs(docs)
    c = make_components(docs)
    IngestPipeline(c).run()
    c.llm = RewritingLLM(fail=True)

    answer = QueryPipeline(c).ask("VRFC2506 capacity target?", history=HISTORY)

    assert answer.answer == "It is 42 wafers [1]."
