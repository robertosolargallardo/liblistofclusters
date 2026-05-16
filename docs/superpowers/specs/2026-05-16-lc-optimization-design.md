# Design: query-latency optimization of liblistofclusters

**Date:** 2026-05-16
**Author:** Roberto Solar (with Claude assistance)
**Status:** Approved (pending spec review)
**Scope:** Single-threaded and batched kNN query throughput. Build time and
memory may grow modestly; correctness must stay exact.

---

## 1. Motivation

The current `bench/README.md` shows liblistofclusters competitive on
non-Euclidean exact search (5–8× sklearn at L1 D=8) but trailing
Faiss on Euclidean. The instructive data point:

- LC single-thread, D=8 Euclidean: **57 k QPS**
- Faiss `IndexFlatL2` (SIMD **brute force, no pruning**): **64 k QPS**
- Faiss `IndexIVFFlat` (nprobe=32): 350 k QPS
- Faiss `IndexHNSWFlat`: 358–491 k QPS

Brute force already matches us with no pruning at all. The bottleneck is
not "how many distances we prune"; it's **how cheaply each distance is
computed**. Faiss's `FlatL2` runs the inner loop as a batched SIMD/BLAS
kernel; we compute `d(q, c_i)` one cluster at a time through a
function-pointer-shaped call.

The same diagnostic at D=32: Faiss `FlatL2` = 97 k QPS, LC 1T = 14 k QPS.
Pruning is working; raw distance throughput is not.

This spec is a focused four-phase optimization aimed at QPS, plus a real
document-embedding benchmark to anchor the result claims.

## 2. Goals

1. ≥ 5× single-threaded QPS vs current LC at D=32 Euclidean, recall = 1.000.
2. Match Faiss `IVFFlat` at nprobe=32 on D=8 Euclidean using `batch_knn`
   at hardware threads, still exact.
3. Beat `hnswlib` at any operating point on real-document angular search
   while staying exact.
4. Keep / extend the existing lead on non-Euclidean metrics.

## 3. Non-goals

- Approximate search modes. LC stays exact in this spec.
- Out-of-RAM or disk-backed indexes.
- Distributed / multi-node indexes.
- Changes to the public `cluster_t`, `internal_object_t`, or `resultslist_t`
  types beyond strictly-additive members.

## 4. Architecture

Four changes, layered. Each is independently buildable, independently
testable, and independently turn-off-able for ablation.

```
                          ┌─────────────────────────────────────┐
                          │   listofclusters<O, Metric, m, ...> │
                          │                                     │
   build path:            │   ┌──────────────────────────────┐  │
   bulk_build(points) ──► │   │ FFT center selection (P4)    │  │
                          │   └──────────────┬───────────────┘  │
                          │                  ▼                  │
                          │   ┌──────────────────────────────┐  │
                          │   │ list<cluster_t>              │  │
                          │   │ centroids + buckets + radii  │  │ ◄── unchanged shape
                          │   └──────────────┬───────────────┘  │
                          │                  ▼                  │
                          │   ┌──────────────────────────────┐  │
                          │   │ centers_soa (P1)             │  │ ◄── NEW: contiguous
                          │   │ flat array, N_clusters × D   │  │     matrix of centers
                          │   └──────────────────────────────┘  │
                          │                                     │
                          │   ┌──────────────────────────────┐  │
                          │   │ aesa_table (P2, optional)    │  │ ◄── NEW: N × k anchor
                          │   │ k anchors + N×k dist table   │  │     distance table
                          │   └──────────────────────────────┘  │
                          └─────────────────────────────────────┘

   query path:
   knn_search(q, k)
     │
     ├─► (P2) if AESA enabled: compute d(q, anchor_i) for i in [0..k_anchors)
     │
     ├─► (P1) compute d(q, all centers) in one SIMD/batched pass
     │
     ├─► (P3) sort cluster indices by d(q, center) ascending
     │
     └─► walk clusters nearest-first:
           - early-terminate when (d_q_center + radius) ≤ shrinking_radius
           - filter bucket members via triangle inequality (SIMD where applicable)
           - if AESA enabled: skip members whose AESA lower bound exceeds shrinking_radius
```

Public API is preserved. `centers_soa` and `aesa_table` are private
members rebuilt by `bulk_build` and refreshed lazily after `insert` /
`remove`. A new `freeze()` method lets purely-online callers pay the
side-structure cost once.

## 5. Phase 1 — Batched-distance LC (`centers_soa`)

### 5.1 Data layout

```cpp
struct centers_soa_t {
    std::vector<double> data;   // N_clusters × D, row-major
    std::size_t         dim;    // D
    std::size_t         n;      // N_clusters (== _list.size() when fresh)
    bool                stale;  // true after insert/remove until next rebuild
};
centers_soa_t _centers;
```

- For compile-time-D `object_t = std::array<double, D>` the dim is known
  statically.
- For runtime-D `std::vector<double>`, dim is captured from the first
  inserted point and asserted thereafter.
- For non-vector objects (Levenshtein on strings, Hamming on bitsets)
  `centers_soa` is **not built**; queries fall back to the existing
  scalar loop. Driven by a `supports_batched_distance<M, O>` trait.

### 5.2 Build & refresh

- `bulk_build` populates `_centers` at the end (one O(N_clusters × D) copy).
- `insert` / `remove` set `stale = true` and do not touch the SoA.
- Query path lazily rebuilds when stale and the trait is true.
- `freeze()` is an explicit "build all side structures now" entry point
  for online workloads that never call `bulk_build`.

### 5.3 Batched distance kernel

```cpp
namespace metric::detail {
template <class Metric, class Object>
std::vector<double> batched_distance(
    const Metric& m,
    const Object& q,
    std::span<const double> centers_flat,
    std::size_t dim,
    std::size_t n);
}
```

Specializations:
- `metric::euclidean` — NEON (f64x2) / AVX2 (f64x4) inner kernel. Lifts
  the per-distance SIMD primitive (already shipped via Phase 5.7) into
  the outer loop.
- `metric::manhattan`, `metric::chebyshev`, `metric::minkowski<p>` —
  same shape, different reduce op.
- Default — scalar generic loop. Still a contiguous-memory win.

Trait `supports_batched_distance<M, O>` gates the specializations.

### 5.4 Query path

```cpp
auto d_centers = detail::batched_distance(_metric, q, _centers.data, dim, n);

// Phase 3 argsort:
std::vector<std::uint32_t> order(_centers.n);
std::iota(order.begin(), order.end(), 0u);
std::sort(order.begin(), order.end(),
          [&](auto a, auto b) noexcept { return d_centers[a] < d_centers[b]; });

for (std::uint32_t i : order) {
    const cluster_t& c = _list[i];
    const double d = d_centers[i];          // already computed
    // … existing centroid candidate + bucket walk, using d instead of recomputing
    if ((d + radius) <= c.radius()) break;
}
```

The bucket triangle-inequality filter is itself SIMD-vectorized as a
small kernel (`bucket_size` is a compile-time template parameter, so
the comparison loop unrolls exactly).

### 5.5 Costs

- **Memory:** `N_clusters × D × 8` bytes. For N=10 k, D=8: 64 KB. For
  N=100 k, D=32: 2.5 MB.
- **Build:** one extra copy after `bulk_build`; lazy refresh after
  online edits.
- **Query:** strictly faster than today on numeric metrics, equal on
  non-numeric.

## 6. Phase 2 — AESA-lite global pivot table

### 6.1 The bound

For any metric `d`, any anchor `a`, any point `p`, query `q`:

```
d(q, p) ≥ |d(q, a) − d(p, a)|
```

With `k` anchors the lower bound is `LB(q, p) = max_i |d(q, a_i) − d(p, a_i)|`.
If `LB > current_radius`, skip the candidate. The bound is monotone in
`k`.

### 6.2 Data layout

```cpp
struct aesa_table_t {
    std::size_t                                       k_anchors;   // 0 == disabled
    std::vector<object_t>                             anchors;
    std::vector<double>                               dists;       // N × k, row-major
    std::vector<std::uint32_t>                        row_to_id;
    std::unordered_map<std::uint32_t, std::uint32_t>  id_to_row;
    bool                                              stale;
};
aesa_table_t _aesa;
```

### 6.3 Anchor selection

Farthest-First Traversal seeded from a random point:

1. Pick `anchor_0` uniformly from indexed points.
2. For `i in [1, k)`: pick the point maximizing `min_{j<i} d(p, a_j)`.
   Tie-break by id.
3. Stop at `k_anchors`.

Cost: `O(N · k)` distance calls at build time. Anchors are **fixed
once chosen** — online `insert` does not rerun FFT; instead it appends
`k` distances per new point. Online `remove` marks the row dead via a
bitset; compaction happens on the next rebuild.

### 6.4 Query path

Inside `knn_search`, before the cluster walk:

```cpp
std::array<double, k_anchors> dqa;
for (size_t i = 0; i < k_anchors; ++i)
    dqa[i] = _metric(q, _aesa.anchors[i].object());
```

Inside the bucket walk, after the existing centroid-distance filter:

```cpp
if (k_anchors > 0) {
    const double* row = &_aesa.dists[id_to_row[m.id()] * k_anchors];
    double lb = 0.0;
    for (size_t i = 0; i < k_anchors; ++i) {
        const double diff = std::abs(dqa[i] - row[i]);
        if (diff > lb) lb = diff;
    }
    if (lb >= radius) continue;
}
const double md = _metric(q, m.object());
```

The inner LB loop is small (`k ≤ 32`) and trivially SIMD-vectorizable
on doubles. The `id_to_row` unordered_map lookup is one hash per
candidate that survives the centroid-distance filter; cheaper than the
`d(q, m)` call it gates, but not free. If profiling shows it on the
hot path we add a `uint32_t aesa_row` field directly to
`internal_object_t` and drop the map lookup. Deferred to that
measurement.

### 6.5 Knob & defaults

Constructor argument `k_anchors`, default `0` (off). Suggested defaults
in docs:

| Regime | k_anchors |
|---|---|
| D ≤ 8, cheap metric | 0–4 |
| D ∈ [16, 64], cheap metric | 8 |
| D ≥ 64 or expensive metric | 16 |
| Non-numeric (Levenshtein, generic) | 16–32 |

### 6.6 Costs

- **Memory:** `N · k · 8` bytes. N=100 k, k=16 → 13 MB; N=1 M, k=32 →
  256 MB. Documented; capped by `k_anchors`.
- **Build:** O(N · k) extra distance calls.
- **Query:** k extra distance calls per query, plus a tight scalar/SIMD
  LB loop per candidate.

## 7. Phase 3 — Nearest-first cluster ordering

Today `knn_search` / `range_search` walk `_list` in insertion order.
With `d_centers[]` already computed by Phase 1, argsort cluster indices
by `d_centers[i]` ascending and walk in that order. The shrinking
radius then drops faster, so the centroid-distance early-termination
check `(d_centers[i] − radius) > cluster.radius` prunes far clusters
before their buckets are touched.

Cost: one O(N_clusters · log N_clusters) sort per query — sub-µs for
typical N_clusters. The sort is dwarfed by the query work it saves.

For metrics where `supports_batched_distance` is false, we still pay
the `d(q, c_i)` distances up front to drive the ordering. This is the
same work as today's per-cluster centroid distance, just hoisted out
of the bucket walk, so per-cluster cost is unchanged and we still gain
the ordering benefit.

Range search applies the ordering too; the win is smaller (no
shrinking radius) but it costs nothing.

## 8. Phase 4 — Farthest-First Traversal in `bulk_build`

Today's `bulk_build` picks the first unassigned point as each new
center. Replace with FFT seeded from a random unassigned point:

```cpp
std::size_t c_idx = _list.empty()
    ? pick_random_unassigned(assigned, rng)
    : pick_farthest_unassigned(objs, assigned, _list, _metric);
```

`pick_farthest_unassigned` maintains, for each unassigned `u`,
`min_j d(u, center_j)` incrementally. Adding a new center costs one
distance call per unassigned point (not all-pairs). Total bulk_build
distance count stays `O((N/m) · N)`, ~2× constant factor over today's
"first unassigned" path.

For very large N a sample-based FFT fallback evaluates the criterion
over a uniform random sample (4096 points). Engaged when N exceeds
~100 k. Standard FFT clustering tradeoff.

Tighter, more separated clusters compound with P1/P2/P3:
- Smaller per-cluster radii → stronger centroid bound.
- Better-spread centers → fewer overlapping balls → nearest-first
  ordering picks correctly more often.

### API

```cpp
enum class build_strategy { first_unassigned, farthest_first };

void bulk_build(const std::vector<object_t>&, const std::vector<uint32_t>&,
                build_strategy = build_strategy::farthest_first,
                bool use_pivots = false);
```

Default is FFT. The `first_unassigned` path stays for paper-faithful
behavior and bench A/B.

## 9. Composition

| Phase | What it does | Helps which regime |
|---|---|---|
| P1 centers_soa + SIMD | Batched centroid distance | Numeric metrics, all D |
| P2 AESA-lite | Anchor-table candidate filter | All metrics; biggest at high D and expensive metrics |
| P3 nearest-first | Visit closest cluster first | All metrics, all D — biggest absolute win on kNN |
| P4 FFT centers | Tighter, more separated clusters | All metrics, all D — multiplies P1/P2/P3 |

Each has an off switch (trait for P1, `k_anchors=0` for P2,
implicit-with-P1 for P3 with a scalar fallback otherwise,
`build_strategy::first_unassigned` for P4).

### 9.1 Removal of existing per-cluster pivots (Phase 5.6)

The library ships an opt-in `use_pivots` flag in `bulk_build` that
stores one per-cluster pivot plus per-bucket-member pivot distances
(`cluster_t::_pivot`, `internal_object_t::pivot_distance`). Phase 5.6
documented it as a **negative result**: query-time wins didn't pay for
the per-cluster bookkeeping, so the default has been `false` since it
landed.

AESA-lite (P2) is a strictly more general version of the same idea — k
**global** anchors shared across all clusters, with one flat k×N
table, prunes candidates the per-cluster pivot bound also prunes (and
more). Keeping both means carrying dead opt-in infrastructure for no
benefit. As part of this work we **remove** the per-cluster pivot
machinery:

- Drop `use_pivots` parameter from `bulk_build`.
- Drop `_pivot`, `has_pivot()`, `pivot()`, `set_pivot()` from
  `cluster_t`.
- Drop `pivot_distance` from `internal_object_t`.
- Drop the `c.has_pivot()` branches in `knn_search` /
  `range_search` / `explore`.
- Drop the corresponding test cases and bench rows.

Net effect: smaller `cluster_t` and `internal_object_t`, simpler
query paths, no orphan API. The removal lands in the same series of
commits as P2 — never a window where both the old and new pivot
mechanisms are in tree.

### 9.2 Removal of the two-tier block index (Phase 5.8)

The same logic applies to `_blocks`, `block_t`, `build_block_index()`,
and the tiered branches in `knn_search` / `range_search`. Phase 5.8
documented the block index as a negative result ("Empirically this
does not help (and sometimes slightly hurts) workloads where
consecutive clusters aren't spatially close, which is the typical
case under LC's order-dependent invariant"). It is opt-in dead code
under the same definition.

As part of this work we also remove:

- `block_t` struct.
- `_blocks` member.
- `build_block_index()` method.
- The `if (tiered)` / `if (!_blocks.empty())` branches in
  `range_search` and `knn_search`.
- Related tests and the bench row for the blocks variant.

P1 and P3 together — batched centroid distances, then walk in
nearest-first order — supersede whatever the block index was trying
to do (one distance call pruning many clusters at once). Removal
happens alongside P1 to keep the query-path branches simple from the
start.

## 10. Testing

### 10.1 Correctness

```
tests/test_smoke.cc additions:
  - test_centers_soa_consistency       — P1 only, lazy rebuild after inserts
  - test_aesa_lower_bound_correctness  — P2 only, sweep k_anchors ∈ {0, 4, 16, 32}
  - test_nearest_first_same_results    — P3 only, vs insertion-order walk
  - test_fft_build_same_results        — P4 only, vs first_unassigned
  - test_phase_matrix                  — full cross product of P1/P2/P3/P4 on/off
                                         32 configs × 2 D values × 50 queries = 3200 brute checks
```

Two property-style sanity checks:
- `LB ≤ true distance` on random pairs.
- `centers_soa == list centroids` (pointwise) after rebuild.

CI runs both `tests/test_smoke` (ASan+UBSan) and `tests/test_smoke_nosan`
(release). Any phase that regresses any test blocks merge.

### 10.2 Benchmark gates (synthetic, controlled)

Measured QPS gain, no recall regression. Each phase must hit its gate
to land; otherwise revert and document.

| Phase | Min gain (single-threaded) | Tested at |
|---|---|---|
| P1 | ≥ 2× vs current LC 1T, Euclidean D=8 | N=10k, D=8, k=10 |
| P1 | ≥ 3× vs current LC 1T, Euclidean D=32 | N=10k, D=32, k=10 |
| P2 (k=16) | ≥ 1.5× vs P1-only, L1 D=32 | N=10k, D=32, k=10 |
| P3 | ≥ 1.3× vs P1-only | N=10k, D=8 and D=32 |
| P4 | ≥ 1.2× vs first_unassigned, holding P1+P2+P3 on | N=10k, all D |
| Combined | ≥ 5× vs current LC 1T on Euclidean D=32 | N=10k |
| Combined batched | Match Faiss IVFFlat nprobe=32 on D=8 | N=10k, HW threads |

Bench also reports recall (must = 1.000 for every exact config) and
bytes/point memory overhead.

### 10.3 Ablation sweep

- N ∈ {1k, 10k, 100k}
- D ∈ {8, 32, 128}
- metric ∈ {euclidean, manhattan, chebyshev, canberra, angular}
- k_anchors (P2) ∈ {0, 8, 16, 32}
- build_strategy ∈ {first_unassigned, farthest_first}

CSV + regenerated plots committed.

## 11. Phase ordering & stop-points

Order: **P1 → P3 → P4 → P2**. P1 is the highest-confidence single win
and unblocks P3 for free. P3 and P4 are cheap compounders. P2 has the
largest engineering surface and is deferred until we know we need it
for the high-D / non-numeric regimes.

After each phase:
1. Profile if gate isn't hit.
2. One or two obvious fixes.
3. If still no, revert the phase, document the negative result in the
   bench README. Match the precedent set by Phase 5.6 (pivot-augmented)
   and Phase 5.8 (block index).

## 12. Headline benchmark — real document embeddings

### 12.1 Corpus build (`bench/data/build_corpus.py`)

Reproducible, hermetic after first run.

1. **Pinned arXiv IDs** (~50, baked into the script):
   - ~15 metric indexing / ANN classics
   - ~10 modern vector search systems
   - ~10 embedding model papers
   - ~10 contrastive / dense retrieval
   - ~5 general ML filler

   Stored as `bench/data/arxiv_ids.txt` (committed).

2. **Download**: `bench/data/pdfs/<id>.pdf` via the `arxiv` package.
   Polite (1 s between requests), resumable. SHA256 recorded on first
   download for a reproducibility receipt. Gitignored.

3. **Text extract**: `bench/data/text/<id>.txt` via `pypdf`. References
   stripped via regex; pages under 200 non-whitespace chars dropped.

4. **Chunk**: `bench/data/chunks.jsonl` — 512-char sliding window, 64
   char overlap. Each record `{id, arxiv_id, text}`. Target ~6 k–10 k
   chunks total.

5. **Embed**: `bench/data/embeddings.npy` `(N, 768) float32`.
   `sentence-transformers` loading `jinaai/jina-embeddings-v2-base-en`,
   pinned to a specific HuggingFace revision (recorded in the script).
   Batch 32, CPU, checkpoint every 1000 chunks. L2-normalized at write
   time. Gitignored.

Idempotent: each stage skips work whose output already exists.

### 12.2 Bench harness (`bench/compare_documents.py`)

1. Load `embeddings.npy` → `(N, 768)` L2-normalized float32.
2. Split: `random.seed(42)`, 200 queries, rest as DB.
3. Ground truth: numpy brute force.
4. Compete:
   - `liblistofclusters_baseline` (pre-P1)
   - `liblistofclusters_p1` (centers_soa)
   - `liblistofclusters_p1_p3` (+ nearest-first)
   - `liblistofclusters_p1_p3_p4` (+ FFT)
   - `liblistofclusters_full` (+ AESA-lite, k=16)
   - `liblistofclusters_full_batched` (HW threads)
   - `faiss.FlatIP`
   - `faiss.IndexIVFFlat` (nprobe ∈ {4, 10, 32, 100})
   - `faiss.IndexHNSWFlat` (efSearch ∈ {16, 32, 64, 128})
   - `hnswlib` (ef ∈ {16, 32, 64, 128})
   - `sklearn.BallTree`
   - `numpy brute`
5. Per method: build_time_s, index_bytes, qps, recall_at_10.
6. Outputs: `bench/compare_documents.csv`, `bench/compare_documents_pareto.png`,
   `bench/compare_documents_dist.png`.

### 12.3 README placement

This becomes the **headline result** in `bench/README.md`. The
synthetic sweep moves to a "Controlled ablation: uniform-random"
section below.

### 12.4 CI policy

Not per-PR (PDF download + 120 MB model is too heavy). Separate
workflow `.github/workflows/bench_documents.yml`:
- `workflow_dispatch` + weekly cron
- Caches HF model and `bench/data/` between runs
- Artifacts uploaded; weekly cron commits to `bench-results` branch
  for trend lines

`ci.yml` (C++ tests + Python tests) continues to run per-PR. No
slowdown there.

### 12.5 Acceptance gates for the document benchmark

| Gate | Threshold |
|---|---|
| Full LC vs current LC at angular | ≥ 5× QPS, 1T, recall = 1.000 |
| Full LC batched vs current LC at angular | ≥ 8× QPS at HW threads |
| Full LC vs Faiss IVFFlat at angular | Match QPS at recall ≥ 0.95 |
| Full LC vs hnswlib at angular | Beat at any operating point at recall = 1.000 |
| Bench first-run on clean checkout | ≤ 20 min on a laptop |
| Bench second-run | ≤ 30 s to plot |

### 12.6 Reproducibility receipts (committed)

- `bench/data/arxiv_ids.txt`
- `bench/data/MODEL_REVISION.txt`
- `bench/data/CHUNKING.md`
- `bench/data/.gitignore` covering `pdfs/`, `text/`, `embeddings.npy`,
  `chunks.jsonl`

## 13. Definition of done

A merged `dev → master` PR whose description quotes:
- Single number per regime: combined-config QPS vs current LC 1T and
  vs Faiss IVFFlat / hnswlib / sklearn, at N=10 k and N=100 k
  (synthetic).
- Headline numbers on the real-document corpus (Section 12).
- Updated `bench/README.md`, regenerated plots, CHANGELOG entry
  referencing phase tags.
- Any reverted phase explicitly documented (with the negative-result
  bench numbers) in the bench README.

## 14. Risks & mitigations

- **SIMD specialization grows the header surface.** Mitigation: gate
  via the trait, keep specializations in `metric::detail`, give every
  metric a working scalar default.
- **AESA memory blow-up at large N.** Mitigation: `k_anchors=0`
  default, document the cost, surface index size in the bench.
- **FFT bulk_build slower at very large N.** Mitigation: incremental
  min-distance maintenance; sample fallback above N=100 k.
- **PDF source mutability.** Mitigation: per-PDF SHA receipt; warn but
  do not block on drift.
- **Jina model mutability on HuggingFace.** Mitigation: pin to a
  specific HF revision; record the SHA in
  `bench/data/MODEL_REVISION.txt`.
- **A phase fails its gate.** Mitigation: revert and document. Memory
  rule applies: fix the bug you uncover, don't punt — and don't merge
  a regression for "we'll get to it later."

