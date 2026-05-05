# Architecture — Hybrid Memory Agent for Vietnamese Users

**Author:** Trần Đình Minh Vương (2A202600495)
**Type:** Solo work

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER (Vietnamese)                        │
│   "Tôi đã đọc gì về Kubernetes?" | "Recommend đọc gì tiếp?"     │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    HybridMemoryAgent.recall()                    │
│                                                                 │
│  ┌──────────────┐    ┌───────────────┐    ┌──────────────────┐  │
│  │ Feast Online │    │  Qdrant       │    │  RRF Fusion      │  │
│  │ Store        │    │  (in-memory)  │    │  k=60            │  │
│  │ (SQLite)     │    │               │    │                  │  │
│  │              │    │ ┌───────────┐ │    │  keyword +       │  │
│  │ user_profile │    │ │ BM25      │─┼───▶│  semantic ranks  │  │
│  │ query_veloc. │    │ │ (memory)  │ │    │  → top-K docs    │  │
│  │              │    │ └───────────┘ │    │                  │  │
│  │ get_online() │    │ ┌───────────┐ │    │  + weighted by   │  │
│  │  → stable    │    │ │ Vector    │─┼───▶│  topic_affinity  │  │
│  │    profile   │    │ │ (cosine)  │ │    │  from Feast      │  │
│  └──────┬───────┘    │ └───────────┘ │    └────────┬─────────┘  │
│         │            │  filtered by  │             │            │
│         │            │  user_id      │             │            │
│         │            └───────────────┘             │            │
│         │                  ▲                       │            │
│         │                  │ upsert                │            │
│         │            ┌─────┴────────┐              │            │
│         │            │  remember()  │              │            │
│         │            │  chunk text  │              │            │
│         │            │  → embed     │              │            │
│         │            └──────────────┘              │            │
│         │                                          │            │
│         └────────────────────┬─────────────────────┘            │
│                              │                                  │
│                              ▼                                  │
│              ┌──────────────────────────────┐                   │
│              │  CONTEXT ASSEMBLY            │                   │
│              │  "User prefers <topic>,      │                   │
│              │   reading <speed>wpm.        │                   │
│              │   Recent: <queries>.         │                   │
│              │   Top memories: <docs>."     │                   │
│              └──────────────────────────────┘                   │
│                              │                                  │
└──────────────────────────────┼──────────────────────────────────┘
                               │ assembled context string
                               ▼
                    ┌──────────────────────┐
                    │   LLM (future)       │
                    │   Final response     │
                    └──────────────────────┘
```

**Data flow summary:**
1. User query enters `recall()` → triggers parallel retrieval from Feast (stable profile) + Qdrant (episodic memory)
2. Qdrant search runs hybrid RRF (BM25 + vector) over user-specific memory, filtered by `user_id` payload
3. Feast `get_online_features()` returns `topic_affinity`, `reading_speed_wpm`, `queries_last_hour`
4. Context assembler merges profile + top-3 memories into a prompt-ready string
5. New memories enter via `remember()` → tokenize VN text → embed → upsert to Qdrant

---

## Architecture Decision 1 — Chunking Strategy

**Decision:** Per-message chunking with overlapping semantic windows.

Each user message / document is stored as 1 chunk with `title + text` concatenated. For documents > 500 words, split by sentence boundary with 1-sentence overlap between chunks. Each chunk maps to one Qdrant point with payload: `{user_id, chunk_id, parent_doc, timestamp}`.

**Tradeoff:**
| Approach | Retrieval Quality | Storage Cost | Context Window Efficiency |
|---|---|---|---|
| Per-message (chosen) | Medium — finds the exact message | Low — 1 point per message | High — precise context, no noise |
| Per-conversation | Low — whole conversation is too broad | Lowest | Low — wastes LLM token budget |
| Semantic-break (sentence-level) | High — finds exact passage | High — 5-10× more points | High — precise but index size explodes |
| Fixed-token (512-token windows) | Medium-High | Medium | Medium — good for large docs |

**Why per-message:** The user's memory use case (chat history, notes) has naturally bounded units — messages are already the right granularity. Semantic breaks would be overkill for Vietnamese (tokenizer accuracy varies) and fixed-token windows lose natural conversation boundaries. Per-message provides the best clarity-to-cost ratio for a POC.

---

## Architecture Decision 2 — Feature Schema

**Decision:** Tabular features in Feast FeatureStore, NOT embedding-based latent preferences.

Three feature views:
1. `user_profile` — TTL=30 days: `reading_speed_wpm`, `preferred_language`, `topic_affinity`
2. `query_velocity` — TTL=1 hour: `queries_last_hour`, `distinct_topics_24h`
3. `item_popularity` — TTL=24 hours: `click_count_24h`, `ctr_7d`, `avg_dwell_seconds`

**Tradeoff — Tabular vs Embedding-based:**
| Pattern | Query Speed | Interpretability | Cold-start | Update Cost |
|---|---|---|---|---|
| Tabular (chosen) | < 5ms P99 | High — every value is human-readable | Needs defaults | Low — just update SQLite/Redis row |
| Embedding (latent prefs from history) | 20-50ms (needs vector search) | Low — opaque vector, hard to debug | Needs N interactions to warm | High — re-embed all user history on profile change |

**Why tabular:** For the POC stage, `topic_affinity` as a categorical string and `reading_speed_wpm` as a scalar are immediately useful for personalizing recommendations. Embedding-based profiles would capture richer preferences (e.g., "user likes dense technical docs with code examples") but:
- Need > 50 user interactions before the embedding becomes meaningful → bad cold-start
- Cannot be inspected or audited → hard to explain "why this recommendation"
- Re-embedding on every profile change wastes compute

The tabular schema also maps 1-to-1 with the lab's existing Feast feature views, demonstrating explicit lab concept linkage (PIT join, TTL, materialize).

---

## Architecture Decision 3 — Freshness Strategy

**Decision:** Three-tier freshness aligned to data nature:

| Data Type | Freshness | Mechanism | Use Case |
|---|---|---|---|
| Episodic memory (new document read) | **Sub-second** | `remember()` calls `upsert` directly to Qdrant (in-memory, no batch delay) | "Tôi vừa đọc gì xong?" — user expects instant recall |
| Query velocity (recent activity) | **1-hour batch** | Feast `materialize-incremental` every hour from Parquet → SQLite online | "Tôi đang quan tâm gì gần đây?" — hourly window is sufficient for trend detection |
| Stable profile (reading speed, language, topic) | **Daily batch** | Feast materialize daily; user profile changes slowly | "Recommend đọc gì tiếp" — topic affinity doesn't change mid-session |

**Why not all sub-second:**
Streaming Push API (sub-second for everything) is over-engineered for this POC:
- Query velocity aggregated hourly is enough — user won't notice 1-hour lag on "what have I been into lately"
- Streaming infra (Kafka + Flink + Feast streaming) adds 3+ services beyond scope
- The cost/complexity ratio doesn't justify the marginal benefit of sub-second profile updates

**Rejected alternative — Daily batch for everything:**
I considered daily Feast materialize for all data (simplest), but rejected it because:
- User who just read a document expects instant recall in the next query
- Delaying episodic memory ingestion by 24h breaks the core promise of "personal assistant memory"
- The Qdrant in-memory upsert is already sub-second; adding a batch layer would introduce unnecessary latency

---

## Rejected Alternative

**Rejected: Storing episodic memory inside Feast as an embedding feature view.**

Rationale for considering: Unify everything under Feast — one storage, one query pattern, one materialize pipeline. Episodic memories would be rows in a Parquet table with `user_id`, `memory_embedding`, `timestamp`. Vector similarity search would use Feast's online store + external index.

**Why rejected — chose separate Qdrant instead:**

Feast is designed for **point-in-time feature lookups** ("give me feature X for entity Y at time T"), not for **approximate nearest neighbor search** over millions of embeddings. Pushing vector search into Feast would mean:
1. Feast online store (SQLite/Redis) has no ANN index → brute-force cosine over 100K memories = 200ms+, fails latency budget
2. Feast's materialize cycle (hourly/daily) applies TTL uniformly → episodic memory has per-user retention needs (user A deletes memory, user B keeps it)
3. Re-index cycle mismatch: profile features change weekly, episodic memories change per-minute. Tying both to the same materialize cadence forces sub-optimal compromise.

The separation follows the lab's own architecture: **Vector Store for retrieval, Feature Store for structured context**. This is the correct pattern from slide §7.

---

## Vietnamese-Context Considerations

1. **Tokenizer choice:** Vietnamese word segmentation is non-trivial — whitespace split fails on compound words like "học máy" (machine learning), "dữ liệu lớn" (big data). Options: `pyvi` (VNLP), `underthesea`, or whitespace-only. For this POC I chose **whitespace tokenizer** for BM25 + `fastembed` for semantic — matching the lab's approach. In production, switch to `underthesea.word_tokenize()` for BM25 and `bge-m3` (multilingual) for embeddings to improve Vietnamese paraphrase recall from 32% to 60%+.

2. **Code-switching handling:** Vietnamese users frequently mix English technical terms (e.g., "tôi muốn tìm hiểu về Kubernetes deployment"). The hybrid RRF approach naturally handles this: English keywords hit BM25, Vietnamese semantics hit vector. No special code-switching layer needed — the fusion itself compensates.

3. **Privacy / Decree-13 awareness:** Personal AI memory storing user conversations and reading history is sensitive under Vietnam's Decree 13/2023/NĐ-CP. The architecture separates `user_id` at Qdrant payload level (filtered queries) and at Feast entity level — no cross-user data leakage. For production, per-user collection + encryption at rest would be added.

4. **Phonetic typo resilience:** Vietnamese users on mobile frequently type without diacritics (e.g., "hoc may" instead of "học máy"). BM25 with diacritic-stripped tokens would improve recall. This is noted as a future enhancement — the current POC doesn't handle it.

---

## What This POC Doesn't Handle Yet

- **Multi-user isolation** at encryption level (current: payload filtering only, trivially bypassable with direct Qdrant API)
- **Memory CRUD** — no update/delete for episodic memories; only append
- **Memory decay / forgetting** — no TTL on episodic memories, infinite retention
- **LLM integration** — `recall()` returns context string but doesn't call an actual LLM for final response
- **Multi-device sync** — single-process, single-machine; no distributed Feast or Qdrant cluster
- **Phonetic typo normalization** — queries without diacritics will miss BM25 matches entirely
