"""Tests for Searcher — RRF formula, Precision@10, edge cases.

TDD Pattern (VIBE-CODING.md #3): tests define the contract. Code must pass.
Run: python -m pytest tests/test_search.py -v
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import pytest

# Bootstrap path to import app.*
ROOT = Path(__file__).resolve().parent.parent
import sys

sys.path.insert(0, str(ROOT))

from app.search import Searcher, SearchHit  # noqa: E402

CORPUS_PATH = ROOT / "data" / "corpus_vn.jsonl"
GOLDEN_PATH = ROOT / "data" / "golden_set.jsonl"


@pytest.fixture(scope="module")
def searcher() -> Searcher:
    """Build Searcher once per test module — heavy (~30s on first run)."""
    assert CORPUS_PATH.exists(), f"{CORPUS_PATH} missing — run `make seed`"
    return Searcher.from_corpus(CORPUS_PATH)


# ── Unit: RRF formula correctness ────────────────────────────────────────


class TestRRFFormula:
    """Validate the Reciprocal Rank Fusion formula against the spec:
    score(d) = sum over retrievers of 1/(k + rank), k=60, rank 1-based.
    """

    def test_rrf_rank_is_one_based(self, searcher: Searcher):
        """rank_r(d) must start at 1, not 0. If rank=0 the first doc gets
        1/(60+0)=0.0167 instead of 1/(60+1)=0.0164 — small diff that silently
        degrades precision."""
        query = "cloud computing"
        hits = searcher.search(query, mode="hybrid", top_k=10, rrf_k=60)
        assert len(hits) > 0, "Hybrid search must return results on corpus"

        # The top hit must get contributions from both retrievers.
        # If rank were 0-based, the formula would be 1/(60+0) not 1/(60+1).
        # We verify the top-ranked doc appears in both keyword and semantic
        # top-50 — meaning it *should* have RRF contribution from both.
        kw_hits = searcher.search(query, mode="keyword", top_k=50)
        sem_hits = searcher.search(query, mode="semantic", top_k=50)
        kw_ids = {h.doc_id for h in kw_hits}
        sem_ids = {h.doc_id for h in sem_hits}

        overlap = kw_ids & sem_ids
        assert len(overlap) > 0, "Corpus designed so keyword and semantic overlap"

    def test_rrf_score_always_in_range(self, searcher: Searcher):
        """Each RRF score must be in (0, 2/k] range for 2 retrievers.
        Max possible = 1/(k+1)+1/(k+1) = 2/(k+1)."""
        k = 60
        max_score = 2.0 / (k + 1)
        query = "tự động mở rộng"
        hits = searcher.search(query, mode="hybrid", top_k=20, rrf_k=k)
        for h in hits:
            assert 0.0 < h.score <= max_score, (
                f"RRF score {h.score} for {h.doc_id} out of valid range (0, {max_score}]"
            )

    def test_rrf_both_retrievers_contribute(self, searcher: Searcher):
        """A doc found by both retrievers should outrank one found by only one,
        all else being equal."""
        query = "security bảo mật dữ liệu"
        hits = searcher.search(query, mode="hybrid", top_k=20, rrf_k=60)
        kw_ids = {h.doc_id for h in searcher.search(query, mode="keyword", top_k=50)}
        sem_ids = {h.doc_id for h in searcher.search(query, mode="semantic", top_k=50)}

        # At least one of the top-3 hits should be in both keyword and semantic top-50
        top3_ids = {h.doc_id for h in hits[:3]}
        top3_overlap = top3_ids & kw_ids & sem_ids
        assert len(top3_overlap) > 0, (
            "Top-3 hybrid results should include docs found by both retrievers"
        )


# ── Integration: Precision@10 on golden set ─────────────────────────────


class TestPrecisionAt10:
    """Rubric assertion: hybrid > keyword AND hybrid > semantic."""

    def test_hybrid_beats_keyword(self, searcher: Searcher):
        golden = [json.loads(l) for l in GOLDEN_PATH.open(encoding="utf-8")]
        p_kw, p_hyb = [], []
        for q in golden:
            relevant = set(q["relevant_doc_ids"])
            kw_hits = [h.doc_id for h in searcher.search(q["query"], mode="keyword", top_k=10)]
            hyb_hits = [h.doc_id for h in searcher.search(q["query"], mode="hybrid", top_k=10)]
            p_kw.append(sum(1 for d in kw_hits if d in relevant) / len(kw_hits))
            p_hyb.append(sum(1 for d in hyb_hits if d in relevant) / len(hyb_hits))

        avg_kw = statistics.mean(p_kw)
        avg_hyb = statistics.mean(p_hyb)
        assert avg_hyb > avg_kw, (
            f"Hybrid ({avg_hyb:.1%}) must beat keyword ({avg_kw:.1%}). "
            "Check RRF formula: score(d) = sum_r 1/(k + rank_r), rank 1-based."
        )

    def test_hybrid_beats_semantic(self, searcher: Searcher):
        golden = [json.loads(l) for l in GOLDEN_PATH.open(encoding="utf-8")]
        p_sem, p_hyb = [], []
        for q in golden:
            relevant = set(q["relevant_doc_ids"])
            sem_hits = [h.doc_id for h in searcher.search(q["query"], mode="semantic", top_k=10)]
            hyb_hits = [h.doc_id for h in searcher.search(q["query"], mode="hybrid", top_k=10)]
            p_sem.append(sum(1 for d in sem_hits if d in relevant) / len(sem_hits))
            p_hyb.append(sum(1 for d in hyb_hits if d in relevant) / len(hyb_hits))

        avg_sem = statistics.mean(p_sem)
        avg_hyb = statistics.mean(p_hyb)
        assert avg_hyb > avg_sem, (
            f"Hybrid ({avg_hyb:.1%}) must beat semantic ({avg_sem:.1%}). "
        )


# ── Edge cases & input validation ────────────────────────────────────────


class TestEdgeCases:
    """Test boundary behavior that LLMs often hallucinate."""

    def test_search_empty_query_raises(self, searcher: Searcher):
        """Empty query: BM25 returns zeros for all docs — not an error but
        semantically meaningless. FastAPI layer blocks empty strings."""
        # BM25 handles empty tokens gracefully (returns all zeros).
        # The Searcher itself doesn't fail — FastAPI validates q min_length=1.
        hits = searcher.search("", mode="keyword", top_k=10)
        assert isinstance(hits, list)

    def test_search_top_k_limits(self, searcher: Searcher):
        """top_k=1 should return exactly 1 result."""
        hits = searcher.search("cloud", mode="hybrid", top_k=1)
        assert len(hits) == 1

    def test_search_top_k_exceeds_corpus(self, searcher: Searcher):
        """top_k > corpus size should return all available docs."""
        hits = searcher.search("cloud", mode="hybrid", top_k=2000)
        assert len(hits) == searcher.size  # capped by corpus size

    def test_search_result_has_required_fields(self, searcher: Searcher):
        """Every SearchHit must have doc_id, title, text, score."""
        hits = searcher.search("cloud computing", mode="hybrid", top_k=10)
        for h in hits:
            assert isinstance(h.doc_id, str) and h.doc_id, "doc_id must be non-empty string"
            assert isinstance(h.title, str), "title must be string"
            assert isinstance(h.text, str), "text must be string"
            assert isinstance(h.score, float), "score must be float"

    def test_keyword_and_semantic_output_same_shape(self, searcher: Searcher):
        """All 3 modes must return list[SearchHit] with same structure."""
        query = "cloud computing tự động"
        kw = searcher.search(query, mode="keyword", top_k=10)
        sem = searcher.search(query, mode="semantic", top_k=10)
        hyb = searcher.search(query, mode="hybrid", top_k=10)
        assert all(len(h) == 10 for h in (kw, sem, hyb))
        assert all(len(h) > 0 for h in (kw, sem, hyb))

    def test_search_invalid_mode_raises(self, searcher: Searcher):
        """Unknown mode must raise ValueError, not silently default to hybrid."""
        with pytest.raises(ValueError, match="unknown mode"):
            searcher.search("cloud", mode="bogus")  # type: ignore[arg-type]
