"""Online (one-query-at-a-time) variant of compare_documents.py.

Tests the per-query latency regime: each method is called Q times with a
single query, not once with a batch. This is the realistic ANN serving
scenario where queries arrive interactively.

Key difference from compare_documents.py:
 - LC uses `idx.knn(q, k)` (per-query path) not `idx.batch_knn(Q, k)`.
 - Faiss / hnswlib search calls are looped per-query, not batched.
 - Threading: methods still parallelize internally if they can, but
   there's no batch-level Q parallelism — only per-call internal SIMD.

Reports MEDIAN per-query latency in microseconds and the equivalent QPS.
"""
from __future__ import annotations

import argparse
import csv
import statistics
import time
from pathlib import Path

import numpy as np

from compare_documents import (
    build_listofclusters, build_faiss_flatip, build_faiss_ivf, build_hnswlib,
    recall, _set_faiss_threads,
)

DATA = Path(__file__).parent / "data"
EMB = DATA / "embeddings.npy"


def load_embeddings() -> np.ndarray:
    if not EMB.exists():
        raise SystemExit("embeddings.npy missing - run bench/data/build_corpus.py first")
    return np.load(EMB).astype(np.float32)


def split(emb: np.ndarray, n_queries: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(emb.shape[0])
    return emb[idx[n_queries:]], emb[idx[:n_queries]]


# --- Per-query callers ------------------------------------------------------

def run_listofclusters_online(idx, queries, k):
    out = np.zeros((len(queries), k), dtype=np.uint32)
    latencies_us = []
    for i, q in enumerate(queries):
        q_c = np.ascontiguousarray(q, dtype=np.float64)
        t0 = time.perf_counter()
        nbrs, _dists = idx.knn(q_c, k=k)
        latencies_us.append((time.perf_counter() - t0) * 1e6)
        n = min(len(nbrs), k)
        out[i, :n] = nbrs[:n]
        if n < k:
            out[i, n:] = nbrs[-1] if n > 0 else 0
    return out, latencies_us


def run_faiss_flatip_online(index, queries, k, threads):
    _set_faiss_threads(threads)
    out = np.zeros((len(queries), k), dtype=np.int64)
    latencies_us = []
    for i, q in enumerate(queries):
        q_c = q.reshape(1, -1).astype(np.float32)
        t0 = time.perf_counter()
        _, ids = index.search(q_c, k)
        latencies_us.append((time.perf_counter() - t0) * 1e6)
        out[i] = ids[0]
    return out, latencies_us


def run_faiss_ivf_online(index, queries, k, threads):
    _set_faiss_threads(threads)
    out = np.zeros((len(queries), k), dtype=np.int64)
    latencies_us = []
    for i, q in enumerate(queries):
        q_c = q.reshape(1, -1).astype(np.float32)
        t0 = time.perf_counter()
        _, ids = index.search(q_c, k)
        latencies_us.append((time.perf_counter() - t0) * 1e6)
        out[i] = ids[0]
    return out, latencies_us


def run_hnswlib_online(index, queries, k, threads):
    n = threads if threads > 0 else 0
    out = np.zeros((len(queries), k), dtype=np.int64)
    latencies_us = []
    for i, q in enumerate(queries):
        q_c = q.reshape(1, -1).astype(np.float32)
        t0 = time.perf_counter()
        ids, _ = index.knn_query(q_c, k=k, num_threads=n)
        latencies_us.append((time.perf_counter() - t0) * 1e6)
        out[i] = ids[0]
    return out, latencies_us


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", type=int, default=200)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=10, help="Warm-up queries before timing.")
    args = ap.parse_args()

    emb = load_embeddings()
    n_q = min(args.queries, emb.shape[0] // 4)
    db, queries = split(emb, n_q)
    print(f"loaded {emb.shape[0]} embeddings, dim={emb.shape[1]}, db={db.shape[0]}, q={queries.shape[0]}")
    print(f"online mode: {n_q} sequential single-query calls per method (warmup={args.warmup})")

    # Build once.
    print("building indexes...")
    builds = {}
    bt0 = time.perf_counter()
    builds["lc"] = build_listofclusters(db);                                            t_lc = time.perf_counter() - bt0
    bt0 = time.perf_counter()
    builds["faiss_flat"] = build_faiss_flatip(db);                                      t_ff = time.perf_counter() - bt0
    bt0 = time.perf_counter()
    builds["faiss_ivf10"] = build_faiss_ivf(db, 10);                                    t_iv10 = time.perf_counter() - bt0
    bt0 = time.perf_counter()
    builds["faiss_ivf32"] = build_faiss_ivf(db, 32);                                    t_iv32 = time.perf_counter() - bt0
    bt0 = time.perf_counter()
    builds["hnsw_64"]  = build_hnswlib(db, 64);                                         t_h64  = time.perf_counter() - bt0
    bt0 = time.perf_counter()
    builds["hnsw_128"] = build_hnswlib(db, 128);                                        t_h128 = time.perf_counter() - bt0
    print(f"  LC {t_lc:.2f}s, FlatIP {t_ff:.3f}s, IVF10 {t_iv10:.3f}s, IVF32 {t_iv32:.3f}s, HNSW64 {t_h64:.2f}s, HNSW128 {t_h128:.2f}s")

    # Ground truth via Faiss FlatIP (exact).
    gt_arr = np.zeros((queries.shape[0], args.k), dtype=np.int64)
    for i, q in enumerate(queries):
        _, ids = builds["faiss_flat"].search(q.reshape(1, -1).astype(np.float32), args.k)
        gt_arr[i] = ids[0]
    truth = gt_arr

    # Warm up: run a few queries through each method to fill caches.
    warm = queries[:args.warmup]

    methods = [
        ("LC, 1T",                       lambda Q: run_listofclusters_online(builds["lc"], Q, args.k),                t_lc),
        ("faiss.FlatIP 1T",              lambda Q: run_faiss_flatip_online(builds["faiss_flat"], Q, args.k, 1),       t_ff),
        ("faiss.FlatIP HW",              lambda Q: run_faiss_flatip_online(builds["faiss_flat"], Q, args.k, 0),       t_ff),
        ("faiss.IVFFlat np=10 1T",       lambda Q: run_faiss_ivf_online(builds["faiss_ivf10"], Q, args.k, 1),         t_iv10),
        ("faiss.IVFFlat np=32 1T",       lambda Q: run_faiss_ivf_online(builds["faiss_ivf32"], Q, args.k, 1),         t_iv32),
        ("hnswlib ef=64 1T",             lambda Q: run_hnswlib_online(builds["hnsw_64"], Q, args.k, 1),                t_h64),
        ("hnswlib ef=128 1T",            lambda Q: run_hnswlib_online(builds["hnsw_128"], Q, args.k, 1),               t_h128),
    ]

    print(f"\n{'method':<28s}  {'recall@k':>9s}  {'p50 (µs)':>10s}  {'p95 (µs)':>10s}  {'mean QPS':>10s}  {'build s':>8s}")
    rows = []
    for name, fn, build_s in methods:
        try:
            fn(warm)  # warm-up
            got, lat = fn(queries)
            lat_sorted = sorted(lat)
            p50 = statistics.median(lat_sorted)
            p95 = lat_sorted[int(0.95 * len(lat_sorted))]
            mean_qps = 1e6 / statistics.mean(lat_sorted)
            r = recall(np.asarray(got), truth)
            rows.append({"method": name, "recall": float(r),
                         "p50_us": float(p50), "p95_us": float(p95),
                         "qps": float(mean_qps), "build_s": float(build_s)})
            print(f"  {name:<28s}  {r:9.3f}  {p50:10.1f}  {p95:10.1f}  {mean_qps:10.1f}  {build_s:8.3f}")
        except Exception as e:
            print(f"  {name:<28s}  skipped: {e}")

    csv_path = Path(__file__).parent / "compare_documents_online.csv"
    with csv_path.open("w") as f:
        w = csv.DictWriter(f, fieldnames=["method", "recall", "p50_us", "p95_us", "qps", "build_s"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nwrote {csv_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
