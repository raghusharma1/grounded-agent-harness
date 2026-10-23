"""The harness: a bounded tool loop whose final answer must pass verification.

Loop shape, per step:

1. call the model with the full, append-only conversation;
2. if it asked for tools, run them through the registry and send all results back
   in one message;
3. if it answered, verify the answer against the evidence ledger. A grounded
   answer or an explicit abstention ends the run. A rejected answer gets one
   repair turn with the verifier's findings; if it fails again, it is withheld.

The conversation is only ever appended to, never edited, so prompt caching keeps
working across turns and the trace is an exact record of what the model saw.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .model import Model, Usage
from .retrieval import HybridRetriever
from .tools import RunContext, ToolRegistry, default_tools
from .verify import ABSTAIN, verify

SYSTEM = f"""You answer questions using only the knowledge base you can reach through your tools.

- Call search_knowledge before answering. If the results look off-topic, search again with different words. Use read_chunk when a result looks relevant but is cut short.
- End every sentence or bullet of your answer with at least one citation in square brackets, using chunk ids exactly as the tools show them, for example [caching#2]. Cite only chunks your tools returned in this conversation.
- If the knowledge base does not cover the question, reply with {ABSTAIN} followed by one sentence saying what is missing. Do not fill gaps from general knowledge, because the reader cannot check it.
- Keep answers short and plain. Use save_flashcard only when the user asks for a flashcard."""

REPAIR = """Your answer did not pass verification:
{problems}

Revise it. Cite only chunk ids your tools returned in this conversation, search again if you need more evidence, or reply {abstain} if the knowledge base does not support an answer."""


@dataclass
class RunResult:
    question: str
    status: str  # grounded | abstained | unverified | failed
    answer: str
    citations: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    steps: int = 0
    tool_calls: int = 0
    repairs: int = 0
    usage: Usage = field(default_factory=Usage)
    cost_usd: float | None = None
    latency_s: float = 0.0
    trace: list[dict] = field(default_factory=list)

    @property
    def sources(self) -> set[str]:
        return {c.split("#", 1)[0] for c in self.citations}


def _text(content: list[Any]) -> str:
    return "\n".join(b.text for b in content if getattr(b, "type", None) == "text").strip()


class Agent:
    def __init__(self, model: Model, retriever: HybridRetriever, tools: ToolRegistry | None = None,
                 max_steps: int = 8, max_repairs: int = 1, notes_path: Path | None = None,
                 approve: Callable[[str, dict], bool] | None = None):
        self.model, self.retriever = model, retriever
        self.tools = tools or default_tools()
        self.max_steps, self.max_repairs = max_steps, max_repairs
        self.notes_path, self.approve = notes_path, approve

    def run(self, question: str) -> RunResult:
        ctx = RunContext(self.retriever, notes_path=self.notes_path, approve=self.approve)
        result = RunResult(question, "failed", "")
        messages: list[dict] = [{"role": "user", "content": question}]
        tools = self.tools.to_api()
        started = time.perf_counter()

        def finish(status: str, answer: str = "", **extra: Any) -> RunResult:
            result.status, result.answer = status, answer
            for k, v in extra.items():
                setattr(result, k, v)
            result.latency_s = round(time.perf_counter() - started, 3)
            result.cost_usd = result.usage.cost(self.model.name)
            return result

        while result.steps < self.max_steps:
            result.steps += 1
            t0 = time.perf_counter()
            resp = self.model.create(system=SYSTEM, messages=messages, tools=tools)
            result.usage.add(resp.usage)
            event = {"step": result.steps, "stop_reason": resp.stop_reason,
                     "latency_s": round(time.perf_counter() - t0, 3),
                     "input_tokens": getattr(resp.usage, "input_tokens", 0),
                     "output_tokens": getattr(resp.usage, "output_tokens", 0)}
            result.trace.append(event)

            if resp.stop_reason == "refusal":
                return finish("failed", problems=["the model declined this request"])
            if resp.stop_reason == "max_tokens":
                # A truncated turn may hold a half-written tool call; never run it.
                return finish("failed", problems=["the response hit max_tokens before finishing"])

            messages.append({"role": "assistant", "content": resp.content})
            if resp.stop_reason == "pause_turn":
                continue

            calls = [b for b in resp.content if getattr(b, "type", None) == "tool_use"]
            if calls:
                results = []
                for call in calls:
                    content, is_error = self.tools.run(call.name, call.input, ctx)
                    result.tool_calls += 1
                    event.setdefault("tools", []).append({"name": call.name, "input": call.input, "error": is_error})
                    results.append({"type": "tool_result", "tool_use_id": call.id, "content": content,
                                    **({"is_error": True} if is_error else {})})
                messages.append({"role": "user", "content": results})  # all results in one message
                continue

            answer = _text(resp.content)
            verdict = verify(answer, ctx.evidence)
            event["verdict"] = verdict.status
            if verdict.ok:
                return finish(verdict.status, answer, citations=verdict.citations)
            if result.repairs >= self.max_repairs:
                return finish("unverified", answer, citations=verdict.citations, problems=verdict.problems)
            result.repairs += 1
            problems = "\n".join(f"- {p}" for p in verdict.problems)
            messages.append({"role": "user", "content": REPAIR.format(problems=problems, abstain=ABSTAIN)})

        return finish("failed", problems=[f"no verified answer within {self.max_steps} steps"])
