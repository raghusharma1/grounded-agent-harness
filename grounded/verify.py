"""Check an answer against the evidence the agent actually retrieved, before anyone sees it.

The rules are deliberately mechanical, so they cannot be talked around:

* every claim line (sentence block or bullet) carries at least one citation,
* every citation names a chunk a tool returned in this run, not one the model invented,
* or the answer is an explicit abstention.

Passing does not prove the answer is right. It proves every claim points at text
the model was shown, which is what makes the answer checkable by a person.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

CITATION = re.compile(r"\[([A-Za-z0-9_.\-]+#\d+)\]")
ABSTAIN = "INSUFFICIENT_EVIDENCE"


@dataclass
class Verdict:
    status: str  # "grounded" | "abstained" | "rejected"
    citations: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status != "rejected"


def _claim_lines(answer: str) -> list[str]:
    lines = []
    for raw in answer.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.endswith(":") or set(line) <= set("-*_ "):
            continue
        lines.append(line)
    return lines


def verify(answer: str, evidence: dict[str, str] | set[str]) -> Verdict:
    text = answer.strip()
    if not text:
        return Verdict("rejected", problems=["the answer is empty"])
    if text.startswith(ABSTAIN):
        return Verdict("abstained")

    citations = list(dict.fromkeys(CITATION.findall(text)))
    problems = []
    if not citations:
        problems.append("the answer has no citations")
    for cid in citations:
        if cid not in evidence:
            problems.append(f"[{cid}] was not returned by a tool in this run")
    for line in _claim_lines(text):
        if not CITATION.search(line):
            problems.append(f"this line has no citation: {line[:80]!r}")
    return Verdict("rejected" if problems else "grounded", citations, problems)
