"""Two evals: retrieval quality (free, offline) and end-to-end agent behaviour (calls the model).

Retrieval eval: for each answerable question, is a chunk from an expected source
file in the top k? Reported for BM25, vector and hybrid, so a retrieval change is
judged by numbers, not by eye.

Agent eval: for each question, did the agent do the right *kind* of thing?
Answer with verified citations when the knowledge base covers it, abstain when it
does not, and cite the expected sources. Plus latency, tokens and cost per run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Callable

from .agent import RunResult
from .retrieval import HybridRetriever


@dataclass
class Case:
    id: str
    question: str
    sources: list[str]
    answerable: bool


def load_cases(path: str | Path) -> list[Case]:
    cases = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            cases.append(Case(d["id"], d["question"], d.get("sources", []), d.get("answerable", True)))
    return cases


def retrieval_eval(retriever: HybridRetriever, cases: list[Case], k: int = 5) -> dict[str, dict[str, float]]:
    modes = {
        "bm25": lambda q: retriever.bm25.search(q, k),
        "vector": lambda q: retriever.vector.search(q, k),
        "hybrid": lambda q: retriever.search(q, k),
    }
    report = {}
    answerable = [c for c in cases if c.answerable]
    for mode, search in modes.items():
        hits, rr = [], []
        for case in answerable:
            sources = [h.chunk.source for h in search(case.question)]
            rank = next((i for i, s in enumerate(sources, 1) if s in case.sources), None)
            hits.append(rank is not None)
            rr.append(1 / rank if rank else 0.0)
        report[mode] = {f"recall@{k}": round(mean(hits), 3), "mrr": round(mean(rr), 3)}
    return report


@dataclass
class AgentCaseResult:
    case: Case
    run: RunResult

    @property
    def correct_behaviour(self) -> bool:
        return self.run.status == ("grounded" if self.case.answerable else "abstained")

    @property
    def cited_expected_source(self) -> bool:
        return bool(self.run.sources & set(self.case.sources))


def agent_eval(run: Callable[[str], RunResult], cases: list[Case]) -> tuple[list[AgentCaseResult], dict]:
    results = [AgentCaseResult(c, run(c.question)) for c in cases]
    answered = [r for r in results if r.case.answerable and r.run.status == "grounded"]
    costs = [r.run.cost_usd for r in results if r.run.cost_usd is not None]
    summary = {
        "cases": len(results),
        "correct_behaviour": round(mean(r.correct_behaviour for r in results), 3),
        "grounded_when_answerable": round(mean(r.run.status == "grounded" for r in results if r.case.answerable), 3),
        "abstained_when_unanswerable": round(mean(r.run.status == "abstained" for r in results if not r.case.answerable), 3)
        if any(not r.case.answerable for r in results) else None,
        "cited_expected_source": round(mean(r.cited_expected_source for r in answered), 3) if answered else None,
        "unverified_answers": sum(r.run.status == "unverified" for r in results),
        "repairs": sum(r.run.repairs for r in results),
        "mean_latency_s": round(mean(r.run.latency_s for r in results), 2),
        "mean_steps": round(mean(r.run.steps for r in results), 2),
        "total_cost_usd": round(sum(costs), 4) if costs else None,
    }
    return results, summary
