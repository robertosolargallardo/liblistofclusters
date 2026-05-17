# LC Query-Latency Optimization — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans`. Each phase file lists its tasks with checkbox steps.

**Goal.** Beat the current LC query-latency benchmark by ≥5× single-thread at D=32 Euclidean while staying exact, and ship a real-document headline benchmark (arXiv PDFs → Jina v2 embeddings → angular search).

**Spec.** [`docs/superpowers/specs/2026-05-16-lc-optimization-design.md`](../../specs/2026-05-16-lc-optimization-design.md)

**Tech stack.** C++23 (concepts, `std::span`, `[[no_unique_address]]`), NEON/AVX2 intrinsics, ASan+UBSan smoke tests, `make`/CMake builds, nanobind Python bindings, `sentence-transformers` for the document corpus.

## Phase ordering

The spec specifies query-time order P2 → P1 → P3 → walk and implementation order P1 → P3 → P4 → P2. The plan refines the implementation order so the public API surface contracts before it grows:

| Plan phase | Spec mapping | What changes | API impact |
|---|---|---|---|
| [Phase 0](phase-0-remove-negative-results.md) | §9.1 + §9.2 | Remove `use_pivots` and block index machinery | **shrinks** public API |
| [Phase 1](phase-1-centers-soa.md) | §5 + §7 (P1 + P3) | `centers_soa` + scalar batched distance + nearest-first ordering | adds `freeze()` |
| [Phase 2](phase-2-simd.md) | §5.3 | NEON/AVX2 batched-distance specializations | none (internal) |
| [Phase 3](phase-3-fft.md) | §8 (P4) | FFT in `bulk_build` (default flipped) | adds `build_strategy` enum |
| [Phase 4](phase-4-aesa.md) | §6 (P2) | AESA-lite global anchor table | adds `build_aesa(k)` |
| [Phase 5](phase-5-bench-ablation.md) | §10.3 | Synthetic ablation rows + bytes/point | bench-only |
| [Phase 6](phase-6-document-bench.md) | §12 | arXiv + Jina v2 headline benchmark | bench-only |

The motivation for putting removals first: between Phase 3 (FFT) and Phase 4 (AESA, in the original spec ordering), `bulk_build`'s signature would have changed twice (gain `build_strategy`, lose `use_pivots`). Removing `use_pivots` in Phase 0 means each subsequent signature change is monotone.

## Cross-phase bench gates (from spec §10.2 / §12.5)

Each phase ends with a measurement step. Missing the gate triggers profile-and-fix; if the gate still misses, revert the phase and document the negative result (memory rule: don't keep opt-in dead code).

| After phase | Workload | Gate |
|---|---|---|
| Phase 1 (scalar SoA + nearest-first) | N=10k, D=8 Euclidean, 1T | ≥1.5× QPS vs current LC |
| Phase 2 (SIMD) | N=10k, D=8 Euclidean, 1T | ≥2× QPS vs current LC |
| Phase 2 (SIMD) | N=10k, D=32 Euclidean, 1T | ≥3× QPS vs current LC |
| Phase 3 (FFT) | N=10k, all D, holding P1+P2 | ≥1.2× QPS vs `first_unassigned` |
| Phase 4 (AESA k=16) | N=10k, D=32 L1, holding P1+P2+P3 | ≥1.5× QPS |
| All combined | N=10k, D=32 Euclidean, 1T | ≥5× vs current LC |
| All combined | N=10k, D=8 Euclidean, batched HW | match Faiss IVFFlat nprobe=32 |
| Phase 6 doc bench | arXiv corpus, angular, 1T | ≥5× vs current LC |
| Phase 6 doc bench | arXiv corpus, angular, batched | ≥8× vs current LC |
| Phase 6 doc bench | arXiv corpus, angular | match Faiss IVFFlat at recall ≥0.95 |
| Phase 6 doc bench | arXiv corpus, angular | beat hnswlib at any recall = 1.000 op-point |

## Correctness invariants (all phases)

- `tests/test_smoke` passes under ASan+UBSan.
- `python/tests/` passes after each phase that touches the bindings.
- Recall@k against brute force = 1.000 for every exact LC config.
- No phase regresses `test_knn_matches_brute_force`, `test_batch_knn_matches_serial`, or `test_range_search`.

## References

See [`references.md`](references.md) for citations to the original algorithms (LC, AESA, FFT clustering, etc.), modern ANN systems (Faiss, HNSW, ScaNN), and the embedding model used for the document benchmark (Jina v2).
