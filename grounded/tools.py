"""Scoped tools: every tool declares a risk level, every input is validated, writes need approval.

The registry is the only way the agent touches the outside world, so this is where
the harness enforces scope. Read tools run freely. Write tools run only when the
caller supplied an approver that says yes; otherwise the model gets a clear
"denied" result it can explain to the user instead of a silent failure.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .retrieval import HybridRetriever

READ, WRITE = "read", "write"


@dataclass
class RunContext:
    """Per-run state the tools share with the harness."""

    retriever: HybridRetriever
    evidence: dict[str, str] = field(default_factory=dict)  # chunk id -> text the model was shown
    notes_path: Path | None = None
    approve: Callable[[str, dict], bool] | None = None


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict
    handler: Callable[[dict, RunContext], str]
    risk: str = READ

    def to_api(self) -> dict:
        # strict: true makes the API guarantee schema-valid arguments; we still validate,
        # because scripted models in tests and future providers give no such guarantee.
        return {"name": self.name, "description": self.description, "input_schema": self.input_schema, "strict": True}


class ToolError(Exception):
    pass


_TYPES = {"string": str, "integer": int, "array": list, "boolean": bool, "object": dict}


def validate(schema: dict, data: Any) -> None:
    if not isinstance(data, dict):
        raise ToolError("input must be a JSON object")
    props = schema.get("properties", {})
    missing = [k for k in schema.get("required", []) if k not in data]
    if missing:
        raise ToolError(f"missing required field(s): {', '.join(missing)}")
    extra = [k for k in data if k not in props]
    if extra and schema.get("additionalProperties") is False:
        raise ToolError(f"unexpected field(s): {', '.join(extra)}")
    for key, value in data.items():
        spec = props.get(key, {})
        expected = _TYPES.get(spec.get("type", ""))
        if expected and (not isinstance(value, expected) or (expected is int and isinstance(value, bool))):
            raise ToolError(f"{key} must be of type {spec['type']}")
        if expected is int and not (spec.get("minimum", value) <= value <= spec.get("maximum", value)):
            raise ToolError(f"{key} must be between {spec.get('minimum')} and {spec.get('maximum')}")
        if expected is str and not value.strip():
            raise ToolError(f"{key} must not be empty")
        if expected is list and spec.get("items", {}).get("type") == "string":
            if not all(isinstance(v, str) for v in value):
                raise ToolError(f"{key} must be a list of strings")


class ToolRegistry:
    def __init__(self, tools: list[Tool]):
        self.tools = {t.name: t for t in tools}

    def to_api(self) -> list[dict]:
        return [t.to_api() for t in self.tools.values()]

    def run(self, name: str, data: Any, ctx: RunContext) -> tuple[str, bool]:
        """Return (content, is_error). Never raises: errors go back to the model as tool results."""
        tool = self.tools.get(name)
        if tool is None:
            return f"unknown tool {name!r}; available: {', '.join(self.tools)}", True
        try:
            validate(tool.input_schema, data)
            if tool.risk == WRITE and not (ctx.approve and ctx.approve(name, data)):
                return f"denied: {name} changes saved data and was not approved for this run", True
            return tool.handler(data, ctx), False
        except ToolError as e:
            return f"invalid input for {name}: {e}", True


def _schema(properties: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": properties, "required": required, "additionalProperties": False}


def _search(data: dict, ctx: RunContext) -> str:
    hits = ctx.retriever.search(data["query"], k=data.get("k", 5))
    if not hits:
        return "no matching chunks; try different words"
    blocks = []
    for h in hits:
        ctx.evidence[h.chunk.id] = h.chunk.text
        blocks.append(h.chunk.render())
    return "\n\n---\n\n".join(blocks)


def _read(data: dict, ctx: RunContext) -> str:
    chunk = ctx.retriever.get(data["chunk_id"])
    if chunk is None:
        raise ToolError(f"no chunk with id {data['chunk_id']!r}")
    ctx.evidence[chunk.id] = chunk.text
    return chunk.render()


def _save_flashcard(data: dict, ctx: RunContext) -> str:
    unseen = [s for s in data["sources"] if s not in ctx.evidence]
    if not data["sources"] or unseen:
        raise ToolError("sources must be chunk ids retrieved in this run" + (f"; not retrieved: {unseen}" if unseen else ""))
    if ctx.notes_path is None:
        raise ToolError("no notes file configured")
    ctx.notes_path.parent.mkdir(parents=True, exist_ok=True)
    with ctx.notes_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"front": data["front"], "back": data["back"], "sources": data["sources"]}) + "\n")
    return f"saved flashcard to {ctx.notes_path.name}"


def default_tools() -> ToolRegistry:
    return ToolRegistry([
        Tool(
            "search_knowledge",
            "Search the knowledge base. Returns the most relevant chunks, each labelled with the id you must "
            "cite, e.g. [caching#2]. Search again with different wording if results look off-topic.",
            _schema({
                "query": {"type": "string", "description": "What to look for, in plain words."},
                "k": {"type": "integer", "minimum": 1, "maximum": 10, "description": "How many chunks to return (default 5)."},
            }, ["query"]),
            _search,
        ),
        Tool(
            "read_chunk",
            "Read one chunk in full by its id, e.g. when a search result looks relevant but is cut short.",
            _schema({"chunk_id": {"type": "string", "description": "A chunk id such as caching#2."}}, ["chunk_id"]),
            _read,
        ),
        Tool(
            "save_flashcard",
            "Save a study flashcard. Only use this when the user asks for a flashcard. Requires approval, and "
            "every source must be a chunk id you retrieved in this conversation.",
            _schema({
                "front": {"type": "string", "description": "The question side."},
                "back": {"type": "string", "description": "The answer side."},
                "sources": {"type": "array", "items": {"type": "string"}, "description": "Chunk ids that support the answer."},
            }, ["front", "back", "sources"]),
            _save_flashcard,
            risk=WRITE,
        ),
    ])
