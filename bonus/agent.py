"""HybridMemoryAgent — combine Feast Feature Store + Qdrant Vector Store.

Usage:
    agent = HybridMemoryAgent(seed_corpus_path="data/corpus_vn.jsonl")
    agent.remember("User read a doc about Kubernetes autoscaling", user_id="u_001")
    ctx = agent.recall("Kubernetes scaling", user_id="u_001")
    print(ctx)
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from rank_bm25 import BM25Okapi

from feast import FeatureStore

ROOT = Path(__file__).resolve().parent.parent
FEAST_DIR = ROOT / "app" / "feast_repo"


class HybridMemoryAgent:
    """Agent with episodic memory (Qdrant + BM25) and stable profile (Feast)."""

    def __init__(
        self,
        seed_corpus_path: Optional[str] = None,
        rrf_k: int = 60,
        top_k: int = 10,
    ):
        self.rrf_k = rrf_k
        self.top_k = top_k

        self.embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        self.qdrant = QdrantClient(":memory:")
        self.qdrant.create_collection(
            collection_name="bonus_memory",
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )

        self.feast_store = FeatureStore(repo_path=str(FEAST_DIR))

        self._doc_texts: list[str] = []
        self._doc_ids: list[str] = []
        self._bm25: Optional[BM25Okapi] = None
        self._next_id = 0

        if seed_corpus_path:
            self._seed_from_corpus(Path(seed_corpus_path))

    def _tokenize(self, text: str) -> list[str]:
        return text.lower().split()

    def _rebuild_bm25(self):
        if self._doc_texts:
            tokenized = [self._tokenize(t) for t in self._doc_texts]
            self._bm25 = BM25Okapi(tokenized)

    def _upsert_batch(self, texts: list[str], ids: list[str]):
        if not texts:
            return
        vectors = list(self.embedder.embed(texts))
        points = [
            PointStruct(
                id=self._next_id + i,
                vector=v.tolist(),
                payload={
                    "doc_id": ids[i],
                    "text": texts[i],
                },
            )
            for i, v in enumerate(vectors)
        ]
        self._next_id += len(points)
        self.qdrant.upsert(collection_name="bonus_memory", points=points)

    def _seed_from_corpus(self, corpus_path: Path):
        """Preload 1000 docs from corpus_vn.jsonl as initial episodic memory."""
        docs = [json.loads(line) for line in corpus_path.open(encoding="utf-8")]
        texts = [f"{d['title']}\n{d['text']}" for d in docs]
        ids = [d["doc_id"] for d in docs]

        self._doc_texts = list(texts)
        self._doc_ids = list(ids)
        self._rebuild_bm25()

        BATCH = 64
        for start in range(0, len(texts), BATCH):
            batch_texts = texts[start : start + BATCH]
            batch_ids = ids[start : start + BATCH]
            self._upsert_batch(batch_texts, batch_ids)

        print(f"Seeded {len(docs)} episodic memories from corpus")

    def remember(self, text: str, user_id: str = "u_001") -> None:
        """Add a new piece of episodic memory for a user.

        The text is embedded and upserted to Qdrant. The BM25 index is
        updated synchronously so the memory is immediately searchable.
        """
        memory_id = f"memory_{self._next_id}"
        self._doc_texts.append(text)
        self._doc_ids.append(memory_id)
        self._rebuild_bm25()
        self._upsert_batch([text], [memory_id])

    def _search_keyword(self, query: str, depth: int) -> list[tuple[str, float]]:
        if self._bm25 is None or not self._doc_ids:
            return []
        scores = self._bm25.get_scores(self._tokenize(query))
        ranked = sorted(
            enumerate(scores), key=lambda x: -x[1]
        )[:depth]
        return [(self._doc_ids[i], float(score)) for i, score in ranked]

    def _search_semantic(self, query: str, depth: int) -> list[tuple[str, float]]:
        q_vec = next(self.embedder.embed([query])).tolist()
        res = self.qdrant.query_points(
            collection_name="bonus_memory",
            query=q_vec,
            limit=depth,
        )
        return [
            (p.payload.get("doc_id", "?"), float(p.score))
            for p in res.points
        ]

    def _search_hybrid(self, query: str) -> list[str]:
        """RRF fusion over BM25 + vector."""
        depth = max(self.top_k * 5, 50)
        kw_ranked = self._search_keyword(query, depth)
        sem_ranked = self._search_semantic(query, depth)

        rrf: dict[str, float] = {}
        for rank, (doc_id, _score) in enumerate(kw_ranked, start=1):
            rrf[doc_id] = rrf.get(doc_id, 0.0) + 1.0 / (self.rrf_k + rank)
        for rank, (doc_id, _score) in enumerate(sem_ranked, start=1):
            rrf[doc_id] = rrf.get(doc_id, 0.0) + 1.0 / (self.rrf_k + rank)

        return [
            doc_id
            for doc_id, _ in sorted(rrf.items(), key=lambda kv: -kv[1])
        ][: self.top_k]

    def _get_user_profile(self, user_id: str) -> dict:
        """Fetch stable user profile from Feast online store."""
        REQUEST_FEATURES = [
            "user_profile_features:reading_speed_wpm",
            "user_profile_features:preferred_language",
            "user_profile_features:topic_affinity",
            "query_velocity_features:queries_last_hour",
            "query_velocity_features:distinct_topics_24h",
        ]
        result = self.feast_store.get_online_features(
            features=REQUEST_FEATURES,
            entity_rows=[{"user_id": user_id}],
        ).to_dict()
        # Feast returns {feature: [value]}, unpack
        return {
            k.replace("_features", "").replace("user_profile:", "").replace("query_velocity:", ""): v[0]
            for k, v in result.items()
        }

    def recall(self, query: str, user_id: str = "u_001") -> str:
        """Retrieve top-K memories + user profile → assembled context string."""
        t0 = time.perf_counter()
        top_ids = self._search_hybrid(query)
        profile = self._get_user_profile(user_id)
        elapsed = (time.perf_counter() - t0) * 1000

        return self._assemble_context(query, user_id, top_ids, profile, elapsed)

    def _assemble_context(
        self,
        query: str,
        user_id: str,
        top_ids: list[str],
        profile: dict,
        elapsed_ms: float,
    ) -> str:
        topic = profile.get("topic_affinity", "unknown")
        speed = profile.get("reading_speed_wpm", "unknown")
        lang = profile.get("preferred_language", "unknown")
        queries_1h = profile.get("queries_last_hour", 0)
        topics_24h = profile.get("distinct_topics_24h", 0)

        lines = [
            f"=== USER PROFILE ({user_id}) ===",
            f"  Preferred topic: {topic}",
            f"  Reading speed:   {speed} wpm",
            f"  Language:        {lang}",
            f"  Queries (1h):    {queries_1h}",
            f"  Topics (24h):    {topics_24h}",
            "",
            f"=== TOP MEMORIES for '{query}' ===",
        ]

        memory_texts = []
        for rank, doc_id in enumerate(top_ids, 1):
            idx = self._doc_ids.index(doc_id) if doc_id in self._doc_ids else -1
            snippet = (
                self._doc_texts[idx][:120] + "..."
                if idx >= 0 and len(self._doc_texts[idx]) > 120
                else (self._doc_texts[idx] if idx >= 0 else doc_id)
            )
            lines.append(f"  {rank}. [{doc_id}] {snippet}")
            memory_texts.append(snippet)

        lines.append("")
        lines.append(f"[recall latency: {elapsed_ms:.1f}ms]")

        return "\n".join(lines)
