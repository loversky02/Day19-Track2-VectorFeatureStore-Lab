"""Tests for FastAPI /search and /healthz endpoints.

TDD Pattern (VIBE-CODING.md #3): test the contract — response shape,
latency field, valid/invalid inputs, error states.

Run: python -m pytest tests/test_api.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.main import app, _searcher, CORPUS_PATH  # noqa: E402

client = TestClient(app)


@pytest.fixture(scope="module", autouse=True)
def ensure_searcher_loaded():
    """Trigger Searcher loading via lifespan before any tests run."""
    with client:  # triggers lifespan startup
        # Warm-up: first request loads Searcher (fastembed model download ~30s)
        r = client.get("/healthz")
        for _ in range(120):  # wait up to 120s for searcher to be ready
            if r.status_code == 200 and r.json().get("ready"):
                break
            r = client.get("/healthz")
        assert r.json().get("ready"), f"Searcher not ready after 120s: {r.json()}"
        assert r.json().get("n_docs") == 1000, f"Expected 1000 docs, got {r.json()}"
        yield


# ── Health endpoint ──────────────────────────────────────────────────────


class TestHealthz:
    """GET /healthz must return ready=True with n_docs after startup."""

    def test_healthz_ready(self):
        r = client.get("/healthz")
        assert r.status_code == 200
        data = r.json()
        assert data["ready"] is True
        assert data["n_docs"] == 1000

    def test_root_endpoint(self):
        r = client.get("/")
        assert r.status_code == 200
        data = r.json()
        assert "name" in data
        assert "endpoints" in data


# ── Search endpoint — happy path ────────────────────────────────────────


class TestSearchHappyPath:
    """Valid inputs must return SearchResponse with correct shape."""

    def test_search_keyword_returns_valid_response(self):
        r = client.get("/search", params={"q": "cloud computing", "mode": "keyword"})
        assert r.status_code == 200
        body = r.json()
        assert body["query"] == "cloud computing"
        assert body["mode"] == "keyword"
        assert body["top_k"] == 10
        assert "latency_ms" in body
        assert len(body["hits"]) == 10
        for hit in body["hits"]:
            assert "doc_id" in hit
            assert "title" in hit
            assert "text" in hit
            assert "score" in hit

    def test_search_semantic_returns_valid_response(self):
        r = client.get("/search", params={"q": "tự động mở rộng hạ tầng", "mode": "semantic"})
        assert r.status_code == 200
        body = r.json()
        assert body["mode"] == "semantic"
        assert len(body["hits"]) == 10

    def test_search_hybrid_returns_valid_response(self):
        r = client.get("/search", params={"q": "cloud computing security", "mode": "hybrid"})
        assert r.status_code == 200
        body = r.json()
        assert body["mode"] == "hybrid"
        assert len(body["hits"]) == 10
        # Hybrid should have latency < 200ms (warm) — rubric says <50ms after warm-up
        assert body["latency_ms"] < 200, f"Hybrid latency {body['latency_ms']}ms > 200ms"

    def test_search_default_mode_is_hybrid(self):
        r = client.get("/search", params={"q": "cloud"})
        assert r.status_code == 200
        body = r.json()
        assert body["mode"] == "hybrid"

    def test_search_top_k_custom(self):
        """top_k=3 should return exactly 3 hits."""
        r = client.get("/search", params={"q": "cloud", "top_k": 3})
        assert r.status_code == 200
        body = r.json()
        assert len(body["hits"]) == 3
        assert body["top_k"] == 3

    def test_search_rrf_k_custom(self):
        """Custom rrf_k parameter must be accepted."""
        r = client.get("/search", params={"q": "cloud", "mode": "hybrid", "rrf_k": 30})
        assert r.status_code == 200

    def test_search_vietnamese_query(self):
        """Tiếng Việt query must work (corpus is VN)."""
        r = client.get("/search", params={"q": "điện toán đám mây tự động mở rộng", "mode": "hybrid"})
        assert r.status_code == 200
        body = r.json()
        assert len(body["hits"]) > 0

    def test_search_latency_ms_is_positive(self):
        """latency_ms must be > 0 for any real search."""
        r = client.get("/search", params={"q": "cloud", "mode": "keyword"})
        assert r.status_code == 200
        assert r.json()["latency_ms"] > 0


# ── Search endpoint — error states ──────────────────────────────────────


class TestSearchErrors:
    """Invalid inputs must return appropriate HTTP errors."""

    def test_search_empty_query_rejected(self):
        r = client.get("/search", params={"q": "", "mode": "keyword"})
        assert r.status_code == 422  # FastAPI validation: min_length=1

    def test_search_missing_query_rejected(self):
        r = client.get("/search", params={"mode": "keyword"})
        assert r.status_code == 422

    def test_search_invalid_mode_rejected(self):
        r = client.get("/search", params={"q": "cloud", "mode": "garbage"})
        assert r.status_code == 422

    def test_search_invalid_top_k_rejected(self):
        r = client.get("/search", params={"q": "cloud", "top_k": 0})
        assert r.status_code == 422

    def test_search_top_k_too_large_rejected(self):
        r = client.get("/search", params={"q": "cloud", "top_k": 999})
        assert r.status_code == 422  # max is 100


# ── Latency benchmark (rubric check) ────────────────────────────────────


class TestLatencyRubric:
    """Rubric: Hybrid P99 server-side < 50ms after warm-up."""

    def test_hybrid_p99_under_50ms(self):
        """Run 50 queries × 3 reps = 150 calls to get reliable P99 estimate."""
        import json

        golden = [json.loads(l) for l in (ROOT / "data" / "golden_set.jsonl").open(encoding="utf-8")]
        latencies: list[float] = []
        for _ in range(3):
            for q in golden:
                r = client.get("/search", params={"q": q["query"], "mode": "hybrid"})
                assert r.status_code == 200
                latencies.append(r.json()["latency_ms"])

        latencies.sort()
        p99 = latencies[int(len(latencies) * 0.99)]
        assert p99 < 50, (
            f"Hybrid P99 = {p99:.1f}ms >= 50ms rubric threshold. "
            "Check: warm-up, cold fastembed, Qdrant mode."
        )
