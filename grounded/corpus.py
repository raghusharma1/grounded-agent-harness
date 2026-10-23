"""Load a folder of Markdown/text notes and split it into citable chunks.

Every chunk gets a stable id of the form ``<file stem>#<n>``. The agent cites
these ids, and the verifier checks them, so they must not change between runs
over the same files.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


@dataclass(frozen=True)
class Chunk:
    id: str
    source: str
    heading: str
    text: str

    def render(self) -> str:
        """The form the model sees: id and heading first, so it can cite precisely."""
        return f"[{self.id}] {self.heading}\n{self.text}"


def _sections(markdown: str) -> list[tuple[str, str]]:
    """Split on headings. Text before the first heading belongs to an empty heading."""
    sections: list[tuple[str, list[str]]] = [("", [])]
    for line in markdown.splitlines():
        m = HEADING.match(line)
        if m:
            sections.append((m.group(2).strip(), []))
        else:
            sections[-1][1].append(line)
    out = []
    for heading, lines in sections:
        body = "\n".join(lines).strip()
        if body:
            out.append((heading, body))
    return out


def _windows(text: str, max_words: int, overlap: int) -> list[str]:
    words = text.split()
    if len(words) <= max_words:
        return [text]
    step = max_words - overlap
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words) - overlap, step)]


def chunk_document(source: str, markdown: str, max_words: int = 160, overlap: int = 30) -> list[Chunk]:
    title = ""
    chunks: list[Chunk] = []
    for heading, body in _sections(markdown):
        title = title or heading
        label = heading if heading == title else f"{title} > {heading}"
        for piece in _windows(body, max_words, overlap):
            chunks.append(Chunk(f"{source}#{len(chunks) + 1}", source, label, piece))
    return chunks


def load_corpus(folder: str | Path, max_words: int = 160, overlap: int = 30) -> list[Chunk]:
    folder = Path(folder)
    files = sorted(p for p in folder.rglob("*") if p.suffix.lower() in {".md", ".txt"})
    if not files:
        raise FileNotFoundError(f"no .md or .txt files under {folder}")
    chunks: list[Chunk] = []
    for path in files:
        chunks.extend(chunk_document(path.stem, path.read_text(encoding="utf-8"), max_words, overlap))
    return chunks
