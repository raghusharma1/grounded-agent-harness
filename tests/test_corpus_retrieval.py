from conftest import CASES

from grounded.corpus import chunk_document
from grounded.evals import load_cases, retrieval_eval
from grounded.retrieval import HashingEmbedder


def test_chunks_have_stable_unique_ids(chunks):
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids))
    assert "caching#1" in ids
    assert all(c.text.strip() for c in chunks)


def test_long_sections_are_windowed_with_overlap():
    body = " ".join(f"w{i}" for i in range(400))
    parts = chunk_document("doc", f"# Title\n\n{body}", max_words=160, overlap=30)
    assert len(parts) == 3
    assert parts[0].text.split()[-30:] == parts[1].text.split()[:30]
    assert [p.id for p in parts] == ["doc#1", "doc#2", "doc#3"]


def test_headings_carry_the_document_title(chunks):
    eviction = next(c for c in chunks if "Eviction" in c.heading)
    assert eviction.heading == "Caching > Eviction policies"


def test_bm25_finds_exact_terms(retriever):
    top = retriever.bm25.search("token bucket refill rate", k=3)
    assert top[0].chunk.id.startswith("rate-limiting")


def test_hybrid_fuses_both_rank_lists(retriever):
    hits = retriever.search("what do virtual nodes improve on a hash ring", k=5)
    assert hits[0].chunk.source == "consistent-hashing"
    assert set(hits[0].ranks) == {"bm25", "vector"}
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_hashing_embedder_is_normalised_and_deterministic():
    e = HashingEmbedder(dim=64)
    a, b = e.embed(["cache eviction policy", "cache eviction policy"])
    assert abs(float(a @ a) - 1.0) < 1e-5
    assert (a == b).all()


def test_retrieval_eval_meets_the_bar(retriever):
    report = retrieval_eval(retriever, load_cases(CASES), k=5)
    assert report["hybrid"]["recall@5"] >= 0.9
    assert set(report) == {"bm25", "vector", "hybrid"}
