from conftest import CASES

from grounded.agent import Agent
from grounded.evals import agent_eval, load_cases
from grounded.model import ScriptedModel, response, text_block, tool_call


def search(query, call_id="t1"):
    return response(tool_call("search_knowledge", {"query": query, "k": 3}, call_id))


def first_hit(retriever, query):
    return retriever.search(query, k=3)[0].chunk.id


def test_grounded_answer_in_two_steps(retriever):
    cid = first_hit(retriever, "token bucket bursts")
    model = ScriptedModel([search("token bucket bursts"), response(text_block(f"A full bucket can be spent at once [{cid}]."))])
    result = Agent(model, retriever).run("How does a token bucket allow bursts?")
    assert result.status == "grounded"
    assert result.citations == [cid] and result.steps == 2 and result.tool_calls == 1
    assert result.cost_usd is not None and result.cost_usd > 0
    # the tool result went back in one user message, matched to the call id
    tool_msg = model.requests[1]["messages"][-1]
    assert tool_msg["role"] == "user" and tool_msg["content"][0]["tool_use_id"] == "t1"


def test_invented_citation_gets_one_repair_turn(retriever):
    cid = first_hit(retriever, "consistent hashing virtual nodes")
    model = ScriptedModel([
        search("consistent hashing virtual nodes"),
        response(text_block("Virtual nodes even out load [made-up#7].")),
        response(text_block(f"Virtual nodes spread load more evenly [{cid}].")),
    ])
    result = Agent(model, retriever).run("What do virtual nodes do?")
    assert result.status == "grounded" and result.repairs == 1
    repair_prompt = model.requests[2]["messages"][-1]["content"]
    assert "made-up#7" in repair_prompt


def test_answer_that_fails_twice_is_withheld_as_unverified(retriever):
    model = ScriptedModel([
        search("raft"),
        response(text_block("Raft elects a leader by majority vote.")),
        response(text_block("Raft uses randomized election timeouts.")),
    ])
    result = Agent(model, retriever).run("How does Raft elect a leader?")
    assert result.status == "unverified" and result.problems


def test_abstention_ends_the_run(retriever):
    model = ScriptedModel([search("kubernetes scheduler"), response(text_block("INSUFFICIENT_EVIDENCE The notes do not cover Kubernetes."))])
    assert Agent(model, retriever).run("How does the Kubernetes scheduler work?").status == "abstained"


def test_parallel_tool_calls_return_together(retriever):
    both = response(tool_call("search_knowledge", {"query": "LRU"}, "a"), tool_call("search_knowledge", {"query": "LFU"}, "b"))
    cid = first_hit(retriever, "LRU")
    model = ScriptedModel([both, response(text_block(f"LRU drops the least recently used item [{cid}]."))])
    result = Agent(model, retriever).run("Compare LRU and LFU")
    results = model.requests[1]["messages"][-1]["content"]
    assert [r["tool_use_id"] for r in results] == ["a", "b"] and result.tool_calls == 2


def test_step_budget_bounds_the_loop(retriever):
    model = ScriptedModel([search("cache", f"t{i}") for i in range(3)])
    result = Agent(model, retriever, max_steps=3).run("loop forever")
    assert result.status == "failed" and result.steps == 3


def test_truncated_turn_never_runs_its_tool(retriever, tmp_path):
    cut = response(tool_call("save_flashcard", {"front": "Q", "back": "A", "sources": []}, "t1"), stop_reason="max_tokens")
    agent = Agent(ScriptedModel([cut]), retriever, notes_path=tmp_path / "c.jsonl", approve=lambda n, d: True)
    result = agent.run("make a card")
    assert result.status == "failed" and result.tool_calls == 0


def test_refusal_is_reported_not_answered(retriever):
    result = Agent(ScriptedModel([response(stop_reason="refusal")]), retriever).run("anything")
    assert result.status == "failed" and "declined" in result.problems[0]


def test_agent_eval_scores_behaviour(retriever):
    cases = [c for c in load_cases(CASES) if c.id in {"token-bucket", "raft"}]
    cid = first_hit(retriever, "token bucket")
    scripts = {
        "How does a token bucket allow bursts while still enforcing an average rate?":
            [search("token bucket"), response(text_block(f"It allows bursts up to the bucket size [{cid}]."))],
        "How does Raft elect a leader in a distributed cluster?":
            [search("raft"), response(text_block("INSUFFICIENT_EVIDENCE Raft is not in the notes."))],
    }
    results, summary = agent_eval(lambda q: Agent(ScriptedModel(scripts[q]), retriever).run(q), cases)
    assert summary["correct_behaviour"] == 1.0
    assert summary["cited_expected_source"] == 1.0
