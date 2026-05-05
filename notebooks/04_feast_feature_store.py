# ---
# jupyter:
#   jupytext:
#     formats: py:percent
# ---

# %% [markdown]
# # NB4 — Feast Feature Store: 3 Feature Views
#
# **Stack:** Feast (LF AI&Data 2024+) + SQLite online store + Parquet offline.
# Maps to slide §6 (Feast Feature Store) + deliverable bullet 3.
#
# > Mục tiêu: định nghĩa 3 feature views, sinh dữ liệu vào offline store
# > (Parquet), `materialize` sang online store (SQLite), gọi
# > `get_online_features` < 10ms — đó là lookup latency rubric.

# %%
import _setup  # noqa: F401
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl

REPO_ROOT = Path(_setup.__file__).resolve().parent.parent
FEAST_DIR = REPO_ROOT / "app" / "feast_repo"
FEAST_DATA = FEAST_DIR / "data"
FEAST_DATA.mkdir(exist_ok=True)

# %% [markdown]
# ## 1. Sinh dữ liệu offline (Parquet) cho 3 feature views
#
# Trong production, dữ liệu này sẽ đến từ data warehouse (BigQuery/Snowflake/Delta).
# Ở lab, sinh từ corpus + synthetic user activity để học pattern materialize.

# %%
NOW = datetime.now(timezone.utc).replace(microsecond=0)


def make_user_profile(n_users: int = 100) -> pl.DataFrame:
    return pl.DataFrame({
        "user_id": [f"u_{i:03d}" for i in range(n_users)],
        "reading_speed_wpm": [180 + (i * 7) % 200 for i in range(n_users)],
        "preferred_language": ["vi" if i % 3 != 0 else "en" for i in range(n_users)],
        "topic_affinity": [
            ["ai_ml", "cloud", "security", "database", "devops"][i % 5]
            for i in range(n_users)
        ],
        "event_timestamp": [NOW - timedelta(hours=i % 48) for i in range(n_users)],
    })


def make_item_popularity(n_items: int = 1000) -> pl.DataFrame:
    return pl.DataFrame({
        "doc_id": [f"item_{i:04d}" for i in range(n_items)],
        "click_count_24h": [(i * 13) % 500 for i in range(n_items)],
        "ctr_7d": [round(((i * 7) % 100) / 100.0, 3) for i in range(n_items)],
        "avg_dwell_seconds": [10.0 + (i * 0.7) % 90 for i in range(n_items)],
        "event_timestamp": [NOW - timedelta(minutes=i % 720) for i in range(n_items)],
    })


def make_query_velocity(n_users: int = 100) -> pl.DataFrame:
    return pl.DataFrame({
        "user_id": [f"u_{i:03d}" for i in range(n_users)],
        "queries_last_hour": [(i * 11) % 50 for i in range(n_users)],
        "distinct_topics_24h": [1 + (i * 3) % 10 for i in range(n_users)],
        "event_timestamp": [NOW - timedelta(minutes=i % 30) for i in range(n_users)],
    })


make_user_profile().write_parquet(FEAST_DATA / "user_profile.parquet")
make_item_popularity().write_parquet(FEAST_DATA / "item_popularity.parquet")
make_query_velocity().write_parquet(FEAST_DATA / "query_velocity.parquet")
print(f"Wrote 3 Parquet sources to {FEAST_DATA}")
for p in sorted(FEAST_DATA.glob("*.parquet")):
    print(f"  {p.name}  {p.stat().st_size/1024:.1f} KB")

# %% [markdown]
# ## 2. `feast apply` — register 3 feature views với metadata registry
#
# `app/feast_repo/feature_views.py` đã định nghĩa 3 feature views (xem file đó).
# Chạy `feast apply` để Feast đọc file definition và ghi vào `registry.db`.

# %%
res = subprocess.run(
    ["feast", "apply"],
    cwd=str(FEAST_DIR),
    capture_output=True, text=True, check=False,
)
print("STDOUT:")
print(res.stdout)
if res.stderr:
    print("STDERR:")
    print(res.stderr)
assert res.returncode == 0, f"feast apply failed: {res.stderr}"

# %% [markdown]
# ## 3. `feast materialize-incremental` — load offline → online
#
# Feast scan offline store cho mọi sự kiện đến `now`, ghi giá trị mới nhất
# (per entity_key) vào online store. SQLite trong lite path; Redis trong docker path.

# %%
end_dt = NOW.strftime("%Y-%m-%dT%H:%M:%S")
res = subprocess.run(
    ["feast", "materialize-incremental", end_dt],
    cwd=str(FEAST_DIR),
    capture_output=True, text=True, check=False,
)
print(res.stdout[-1500:])
if res.stderr:
    print("STDERR (tail):")
    print(res.stderr[-500:])
assert res.returncode == 0, f"materialize failed: {res.stderr}"

# %% [markdown]
# ## 4. Online lookup — đo latency
#
# `get_online_features()` query online store cho 1 batch entity rows.
# Rubric threshold: P99 < 10ms cho lookup khi online store là SQLite local
# (Redis/Dynamo trong production sẽ < 5ms).

# %%
import time

from feast import FeatureStore

fs = FeatureStore(repo_path=str(FEAST_DIR))

REQUEST_FEATURES = [
    "user_profile_features:reading_speed_wpm",
    "user_profile_features:preferred_language",
    "user_profile_features:topic_affinity",
    "query_velocity_features:queries_last_hour",
    "query_velocity_features:distinct_topics_24h",
]

# Single lookup
t0 = time.perf_counter()
features = fs.get_online_features(
    features=REQUEST_FEATURES,
    entity_rows=[{"user_id": "u_001"}],
).to_dict()
single_latency_ms = (time.perf_counter() - t0) * 1000
print(f"Single lookup: {single_latency_ms:.2f}ms")
print({k: v[0] for k, v in features.items()})

# %% [markdown]
# ## 5. TODO — Batch latency benchmark (100 lookups, P99)

# %%
# Warmup — SQLite connections are lazy; first few calls hit connection setup
for _ in range(10):
    fs.get_online_features(
        features=REQUEST_FEATURES,
        entity_rows=[{"user_id": "u_000"}],
    )

latencies: list[float] = []
for i in range(100):
    user_id = f"u_{i:03d}"
    t0 = time.perf_counter()
    fs.get_online_features(
        features=REQUEST_FEATURES,
        entity_rows=[{"user_id": user_id}],
    ).to_dict()
    latencies.append((time.perf_counter() - t0) * 1000)

latencies.sort()
p50 = latencies[50]
p95 = latencies[95]
p99 = latencies[99]
print(f"Online lookup latency over 100 calls:")
print(f"  P50 = {p50:.2f}ms")
print(f"  P95 = {p95:.2f}ms")
print(f"  P99 = {p99:.2f}ms")

if p99 < 10:
    print(f"PASS — online lookup P99 < 10ms ({p99:.2f}ms)")
else:
    print(f"WARN — P99 = {p99:.2f}ms (SQLite trên macOS thường tốt hơn 5ms; Linux thường tốt hơn 1ms)")

# %% [markdown]
# ## 6. PIT join (offline) — đảm bảo no data leakage
#
# `get_historical_features` thực hiện Point-in-Time join: cho mỗi event row
# `(user_id, ts)`, lấy feature value tại ts đó (không dùng giá trị tương lai).
# Đây là cơ chế chính để tránh training-serving skew (deck §6).

# %%
import pandas as pd
entity_df = pd.DataFrame({
    "user_id": ["u_001", "u_002", "u_003"],
    "event_timestamp": [NOW, NOW, NOW],
})

historical = fs.get_historical_features(
    entity_df=entity_df,
    features=[
        "user_profile_features:reading_speed_wpm",
        "user_profile_features:topic_affinity",
    ],
).to_df()
print(historical)

# %% [markdown]
# ## Deliverable evidence
#
# 1. Output cell 2: 3 Parquet files generated.
# 2. Output cell 3: `feast apply` STDOUT showing "Created feature view <name>" × 3.
# 3. Output cell 4: `materialize` log showing rows materialized to online store.
# 4. Output cell 5: 1 online lookup result + latency.
# 5. Output cell 6: 100-lookup P50/P95/P99 + PASS line.
# 6. Output cell 7: PIT join DataFrame (3 rows × features).
#
# ---
#
# ## Vibe-coding callout
#
# **Delegate freely:** Feast feature view YAML / Python definitions follow strict
# patterns (entity → source → schema). AI nails this in 1 shot if you give it
# the schema. Cũng AI tốt cho synthetic data generators (`make_user_profile`).
#
# **Think hard yourself:** **TTL choices** trong feature_views.py — tại sao
# `user_profile_features` TTL=30 ngày nhưng `query_velocity_features` TTL=1 giờ?
# Nếu sai TTL: query_velocity với TTL=30d sẽ trả giá trị cũ → fraud detection
# bỏ lỡ tín hiệu real-time. **PIT join correctness** cũng là *think-hard* —
# nếu data leakage xảy ra, training accuracy đẹp nhưng prod tệ 20-30% (deck §6).
# Đừng để AI tự chọn TTL hay timestamp_field — bạn phải biết business semantics.
