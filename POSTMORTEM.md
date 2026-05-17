# liblistofclusters — Postmortem

**Status:** Archived 2026-05-17. Final tag: `v0.1-archived` at commit `ebc2d27`.

## What this was

A C++23 header-only implementation of the List of Clusters (LC) metric-space index
(Chávez & Navarro, 2005), built as a generic kNN library with a `Metric` concept,
batched SIMD kernels (NEON / AVX2), a BLAS fast path (Apple Accelerate /
OpenBLAS / MKL), Python bindings (nanobind), and a comparative bench against
Faiss FlatIP / IVFFlat and hnswlib.

## What we measured

**Real Jina v2 corpus, D=768, N=11,332 chunks, 200 queries, k=10, exact recall:**

| method                | recall | batched HW QPS | online 1T QPS |
|-----------------------|-------:|---------------:|--------------:|
| LC (this library)     | 1.000  | ~40,000        | ~1,850        |
| faiss.FlatIP          | 1.000  | ~47,000        | ~2,150        |
| faiss.IVFFlat np=10   | 0.93   | ~44,000        | ~15,400       |
| hnswlib ef=64         | 0.99   | ~17,500        | ~2,800        |

**Synthetic D=512, N=256k, unit-random unit vectors:**

| method                | recall | HW QPS |
|-----------------------|-------:|-------:|
| LC                    | 1.000  |    73  |
| faiss.FlatIP          | 1.000  | 3,174  |
| hnswlib ef=128        | 0.06   | 4,201  |
| faiss.IVFFlat np=64   | 0.28   | 1,678  |

At D≥128 with unit-normalized data, concentration of measure breaks every
approximate index except FlatIP. The synthetic bench was where LC's
PCA verify-style filter was *supposed* to win — instead it was 43× slower
than the brute-force baseline.

## Why the project was wound down

Every LC-specific filter we tried at D≥128 was either bypassed or net-negative:

- **TI walk at high D**: distance concentration kills pruning. Bypassed via
  `if (D >= 128)` topk-no-walk path.
- **AESA-on-PCA cluster filter (batch)**: dead code — its `proj_enabled` and
  `!will_topk_path` gates turned out to be mutually exclusive (both require
  `D ≥ 128`). Deleted in 476b70f.
- **AESA-on-PCA online filter**: A/B showed +22 µs/query overhead with zero
  recall delta. Deleted in 84b99fd.
- **PCA verify-style pre-filter (N≥50k)**: 43× slower than Faiss FlatIP HW at
  N=256k. Per-query O(N log N) sort of LB pairs + cache-cold scattered verify
  swamped the projected-sgemm savings. Deleted in ebc2d27.

After cleanup, `batch_knn` at D≥128 is essentially `cblas_sgemm` for the Q×N
inner-product matrix + per-query heap top-k + a per-result binary search to map
`row_idx → (cluster_idx, bucket_idx)` for result emission. The
List-of-Clusters layout exists only as storage; the algorithm contributes
nothing. The honest framing: at high D this library is FlatIP with cluster
bookkeeping overhead.

LC could plausibly still be useful at low D (D ≲ 32) where triangle-inequality
bounds actually discriminate, or with non-Euclidean / non-BLAS-able metrics
(Manhattan, edit distance, custom predicates) — both of which the `Metric`
concept supports — but we never benched those regimes and modern ANN work
overwhelmingly targets high-D dense embeddings, so the project no longer has
a workload to optimize for.

## What's reusable

- `include/listofclusters/detail/batched_distance.hh` — NEON/AVX2 batched
  pairwise distance kernels + BLAS `cblas_sgemm` Euclidean and IP-only paths.
- `include/listofclusters/metric/` — the `Metric` concept + a registry of
  builtins (euclidean, manhattan, chebyshev, minkowski, hamming, angular,
  jaccard, canberra, levenshtein).
- `bench/` — side-by-side comparison harness against Faiss FlatIP/IVF and
  hnswlib, plus a Jina-v2 corpus build script (`bench/data/build_corpus.py`).
- `python/` — nanobind + scikit-build-core wrapper template.
- The cleanup arc itself (see git log from `463e609..ebc2d27`) is a case study
  in "what looks like an algorithmic win in the paper does nothing at the D
  that modern ANN actually uses."

## Reference for future kNN work

- **Dense high-D embeddings (D ≳ 128)**: use Faiss FlatIP (exact) or Faiss
  IVFPQ / HNSW (approximate). Do not reach for LC-style metric-space indexes —
  concentration of measure makes triangle-inequality bounds useless.
- **Low D or arbitrary metrics where BLAS doesn't apply**: kd-tree, ball-tree,
  or LC are reasonable starting points. Bench before committing.
- **Anything else**: always include FlatIP as a baseline. If FlatIP is fast
  enough at your N, no fancy index is needed.
