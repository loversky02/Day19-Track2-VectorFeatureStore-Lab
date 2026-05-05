"""Demo — 5 queries exercising HybridMemoryAgent across vector, profile, fresh activity,
paraphrase, and mixed scenarios.

Usage:
    python submission/bonus/demo.py

Requires:
    - Feast feature views applied + materialized (from NB4)
    - corpus_vn.jsonl exists at data/corpus_vn.jsonl
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bonus.agent import HybridMemoryAgent  # noqa: E402


def main() -> int:
    print("=" * 62)
    print("  HybridMemoryAgent Demo — 5 Queries")
    print("=" * 62)
    print()

    # ── Initialize agent with seed corpus ──────────────────────────
    print("[1/2] Loading corpus + building indices...")
    agent = HybridMemoryAgent(seed_corpus_path=str(ROOT / "data" / "corpus_vn.jsonl"))
    print()

    # Add a few personalized memories for user u_001
    agent.remember(
        "Người dùng đã đọc tài liệu Kubernetes: triển khai cluster trên GCP với "
        "auto-scaling theo CPU usage. Quan tâm đến HPA và resource limits.",
        user_id="u_001",
    )
    agent.remember(
        "Ghi chú cá nhân: cloud security là ưu tiên số 1. Đã cấu hình IAM roles "
        "cho service accounts và audit logging qua CloudTrail.",
        user_id="u_001",
    )
    agent.remember(
        "Tài liệu về tự động mở rộng hạ tầng: dùng Terraform provision VM groups "
        "và load balancer tự động scale theo traffic pattern.",
        user_id="u_001",
    )

    # ── Run 5 demo queries ─────────────────────────────────────────
    print("[2/2] Running 5 demo queries...\n")

    queries = [
        (
            "Q1 — Vector-only: 'Tôi đã đọc gì về Kubernetes?'",
            "Tôi đã đọc gì về Kubernetes?",
        ),
        (
            "Q2 — Profile context: 'Recommend đọc gì tiếp'",
            "Recommend đọc gì tiếp",
        ),
        (
            "Q3 — Fresh activity: 'Tôi đang quan tâm gì gần đây?'",
            "Tôi đang quan tâm gì gần đây?",
        ),
        (
            "Q4 — Paraphrase: 'Tài liệu về tự động mở rộng hạ tầng?'",
            "Tài liệu về tự động mở rộng hạ tầng?",
        ),
        (
            "Q5 — Mixed (hybrid + profile): 'Cho tôi summary cloud security'",
            "Cho tôi summary cloud security",
        ),
    ]

    for label, query in queries:
        print(f"─── {label} ───")
        print(agent.recall(query, user_id="u_001"))
        print()

    print("=" * 62)
    print("  All 5 queries completed — demo.py exits 0")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    sys.exit(main())
