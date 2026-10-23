from pathlib import Path

import pytest

from grounded.corpus import load_corpus
from grounded.retrieval import HybridRetriever

KB = Path(__file__).resolve().parent.parent / "examples" / "kb"
CASES = KB.parent / "evals" / "questions.jsonl"


@pytest.fixture(scope="session")
def chunks():
    return load_corpus(KB)


@pytest.fixture(scope="session")
def retriever(chunks):
    return HybridRetriever(chunks)
