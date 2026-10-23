"""Model adapters. The agent loop only needs ``create(system, messages, tools)``.

``ClaudeModel`` calls the official Anthropic SDK. ``ScriptedModel`` replays canned
responses with the same shape, so the whole harness (loop, tools, verifier, evals)
is testable offline and deterministically.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Protocol

# USD per million tokens: (input, output, cache read). Cache writes bill at 1.25x input.
PRICES = {
    "claude-opus-5-5": (4.00, 20.00, 0.20),
    "claude-sonnet-5-5": (2.00, 10.00, 0.20),
}


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def add(self, usage: Any) -> None:
        self.input_tokens += getattr(usage, "input_tokens", 0) or 0
        self.output_tokens += getattr(usage, "output_tokens", 0) or 0
        self.cache_read_tokens += getattr(usage, "cache_read_input_tokens", 0) or 0
        self.cache_write_tokens += getattr(usage, "cache_creation_input_tokens", 0) or 0

    def cost(self, model: str) -> float | None:
        if model not in PRICES:
            return None
        inp, out, read = PRICES[model]
        return (self.input_tokens * inp + self.output_tokens * out + self.cache_read_tokens * read
                + self.cache_write_tokens * inp * 1.25) / 1_000_000


class Model(Protocol):
    name: str

    def create(self, *, system: str, messages: list[dict], tools: list[dict]) -> Any: ...


class ClaudeModel:
    """Claude via the Anthropic SDK.

    * Effort is set explicitly: Claude Opus 5.5 defaults to "medium", and this is the main
      quality/cost dial. Thinking is always on for this model, so it is not configured.
    * Top-level ``cache_control`` caches the stable prefix (system prompt and tools) plus
      the growing conversation, so each loop turn re-reads earlier turns at cache price.
    * Server-side fallbacks (``fallbacks="default"``) let the API re-run a request that a
      safety classifier declines on a suitable model, instead of failing the whole run.
    """

    def __init__(self, model: str = "claude-opus-5-5", effort: str = "medium", max_tokens: int = 16000,
                 fallbacks: bool = True, client: Any = None):
        import anthropic

        self.name, self.effort, self.max_tokens, self.fallbacks = model, effort, max_tokens, fallbacks
        self.client = client or anthropic.Anthropic()  # credentials from the environment

    def create(self, *, system: str, messages: list[dict], tools: list[dict]) -> Any:
        params = dict(
            model=self.name,
            max_tokens=self.max_tokens,
            system=system,
            tools=tools,
            messages=messages,
            output_config={"effort": self.effort},
            cache_control={"type": "ephemeral"},
        )
        if self.fallbacks:
            return self.client.beta.messages.create(
                **params, betas=["server-side-fallback-2026-07-01"], fallbacks="default"
            )
        return self.client.messages.create(**params)


def text_block(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def tool_call(name: str, data: dict, call_id: str) -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", id=call_id, name=name, input=data)


def response(*blocks: SimpleNamespace, stop_reason: str | None = None, input_tokens: int = 100,
             output_tokens: int = 20) -> SimpleNamespace:
    if stop_reason is None:
        stop_reason = "tool_use" if any(b.type == "tool_use" for b in blocks) else "end_turn"
    usage = SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens,
                            cache_read_input_tokens=0, cache_creation_input_tokens=0)
    return SimpleNamespace(content=list(blocks), stop_reason=stop_reason, usage=usage)


class ScriptedModel:
    """Replays a fixed list of responses and records every request it receives."""

    def __init__(self, responses: list[SimpleNamespace], name: str = "claude-opus-5-5"):
        self.responses = list(responses)
        self.name = name
        self.requests: list[dict] = []

    def create(self, *, system: str, messages: list[dict], tools: list[dict]) -> Any:
        self.requests.append({"system": system, "messages": list(messages), "tools": tools})
        if not self.responses:
            raise RuntimeError("ScriptedModel ran out of responses")
        return self.responses.pop(0)
