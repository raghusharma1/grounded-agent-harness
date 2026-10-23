from grounded.tools import RunContext, default_tools
from grounded.verify import verify

EVIDENCE = {"caching#2": "cache-aside text", "caching#3": "write-through text"}


def test_grounded_answer_passes():
    answer = "Cache-aside reads the database on a miss [caching#2].\n- First reads are slow [caching#2][caching#3]."
    v = verify(answer, EVIDENCE)
    assert v.status == "grounded"
    assert v.citations == ["caching#2", "caching#3"]


def test_invented_citation_is_rejected():
    v = verify("LRU evicts old items [caching#9].", EVIDENCE)
    assert v.status == "rejected"
    assert any("caching#9" in p for p in v.problems)


def test_uncited_line_is_rejected_but_lead_ins_are_not():
    v = verify("Two patterns:\n- Cache-aside [caching#2].\n- Write-through is safer.", EVIDENCE)
    assert v.status == "rejected"
    assert len(v.problems) == 1 and "Write-through" in v.problems[0]


def test_abstention_is_accepted():
    assert verify("INSUFFICIENT_EVIDENCE The notes do not cover Raft.", set()).status == "abstained"


def test_empty_answer_is_rejected():
    assert verify("   ", EVIDENCE).status == "rejected"


def test_search_records_evidence(retriever):
    ctx = RunContext(retriever)
    content, err = default_tools().run("search_knowledge", {"query": "token bucket", "k": 2}, ctx)
    assert not err and len(ctx.evidence) == 2
    assert all(f"[{cid}]" in content for cid in ctx.evidence)


def test_invalid_input_comes_back_as_an_error_result(retriever):
    tools, ctx = default_tools(), RunContext(retriever)
    assert tools.run("search_knowledge", {"k": 3}, ctx) == ("invalid input for search_knowledge: missing required field(s): query", True)
    assert tools.run("search_knowledge", {"query": "x", "k": 50}, ctx)[1]
    assert tools.run("search_knowledge", {"query": "x", "debug": True}, ctx)[1]
    assert tools.run("drop_table", {}, ctx)[1]


def test_writes_are_denied_without_approval(retriever, tmp_path):
    ctx = RunContext(retriever, notes_path=tmp_path / "cards.jsonl")
    ctx.evidence["caching#2"] = "..."
    card = {"front": "What is cache-aside?", "back": "Lazy loading.", "sources": ["caching#2"]}
    content, err = default_tools().run("save_flashcard", card, ctx)
    assert err and content.startswith("denied")
    assert not ctx.notes_path.exists()


def test_approved_write_must_cite_retrieved_evidence(retriever, tmp_path):
    ctx = RunContext(retriever, notes_path=tmp_path / "cards.jsonl", approve=lambda name, data: True)
    card = {"front": "Q", "back": "A", "sources": ["caching#2"]}
    assert default_tools().run("save_flashcard", card, ctx)[1]  # nothing retrieved yet
    ctx.evidence["caching#2"] = "..."
    content, err = default_tools().run("save_flashcard", card, ctx)
    assert not err and ctx.notes_path.read_text().count("\n") == 1
