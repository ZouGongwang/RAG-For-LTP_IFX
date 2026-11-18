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


def test_history_is_trimmed_and_citations_removed():
    builder = PromptBuilder(PromptConfig(history_turns=2, history_chars=40))
    history = [
        {"role": "user", "content": "old question"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "What is the target?"},
        {"role": "assistant", "content": "## Target\n- It is 42 wafers [1][2] .\n- " + "x" * 80},
        {"role": "user", "content": "And the limit?"},
        {"role": "assistant", "content": "It is 50 [3, 4]."},
    ]
    kept = builder.trim_history(history)
    assert [m["role"] for m in kept] == ["user", "assistant", "user", "assistant"]
    assert kept[0]["content"] == "What is the target?"
    assert "[1]" not in kept[1]["content"] and "42 wafers." in kept[1]["content"]
    assert len(kept[1]["content"]) <= 40 and kept[1]["content"].endswith("…")
    assert kept[3]["content"] == "It is 50."
    assert builder.trim_history(kept) == kept
    assert PromptBuilder(PromptConfig(history_turns=0)).trim_history(history) == []
    assert builder.trim_history([{"role": "assistant", "content": "orphan"}]) == []


def test_prompt_builder_places_history_between_system_and_question():
    builder = PromptBuilder(PromptConfig())
    chunks = [RetrievedChunk("1", "text", {"title": "Doc"})]
    history = [{"role": "user", "content": "Q1"}, {"role": "assistant", "content": "A1 [1]"}]

    messages, _ = builder.build("Q2", chunks, history)
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
    assert messages[2]["content"] == "A1"
    assert "Conversation history" in messages[0]["content"]
    assert "Question: Q2" in messages[3]["content"]

    messages, _ = builder.build("Q2", chunks)
    assert [m["role"] for m in messages] == ["system", "user"]
    assert "Conversation history" not in messages[0]["content"]
