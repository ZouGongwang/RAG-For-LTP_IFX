from rag.config import PromptConfig
from rag.llm.base import ThinkFilter, flatten_messages, strip_think
from rag.models import RetrievedChunk
from rag.prompt import PromptBuilder, cited_indices
from rag.retrieval import reciprocal_rank_fusion
from rag.retrieval.retriever import tokenize


def test_strip_think():
    assert strip_think("<think>\nhmm\n</think>\n\nAnswer") == "Answer"
    assert strip_think("no think") == "no think"
    assert strip_think("partial reasoning</think>Answer") == "Answer"


def test_think_filter_streaming_split_tags():
    pieces = ["Hello <th", "ink>secret", " stuff</thi", "nk> world", "!"]
    f = ThinkFilter()
    out = "".join(f.feed(p) for p in pieces) + f.flush()
    assert out == "Hello  world!"


def test_flatten_messages():
    system, prompt = flatten_messages([{"role": "system", "content": "S"}, {"role": "user", "content": "Q"}])
    assert (system, prompt) == ("S", "Q")


def test_cited_indices():
    assert cited_indices("A [2] and B [1][2], C [3, 4].") == [2, 1, 3, 4]


def test_rrf_prefers_items_in_both_lists():
    a, b, c = (RetrievedChunk(x, x, {}) for x in "abc")
    fused = reciprocal_rank_fusion([[a, b], [c, b]])
    assert fused[0].chunk_id == "b"


def test_tokenize_keeps_identifiers():
    toks = tokenize("Schedule for VRFC-2506 rev1.2")
    assert "vrfc-2506" in toks and "vrfc" in toks and "2506" in toks


def test_prompt_builder_numbers_and_budget():
    chunks = [RetrievedChunk(str(i), "x" * 100, {"title": f"Doc {i}", "url": f"http://c/{i}"}) for i in range(10)]
    messages, used = PromptBuilder(PromptConfig(max_context_chars=400)).build("What?", chunks)
    assert 1 <= len(used) < 10
    assert messages[0]["role"] == "system"
    assert "[1] Doc 0\nURL: http://c/0" in messages[1]["content"]
    assert "Question: What?" in messages[1]["content"]
