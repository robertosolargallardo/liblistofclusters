"""Synthetic-data bench: L2-normalized random unit vectors at large N, D.

The default workload (N=256k, D=512) targets the regime where LC's PCA
pre-filter is expected to start winning over the full BLAS sgemm path.
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from compare_documents import (
    build_listofclusters, run_listofclusters,
    build_faiss_flatip, run_faiss_flatip,
    build_faiss_ivf, run_faiss_ivf,
    build_hnswlib, run_hnswlib,
    recall, time_call,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=256_000)
    ap.add_argument("--d", type=int, default=512)
    ap.add_argument("--queries", type=int, default=200)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-lc", action="store_true", help="Skip LC build (slow at large N).")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    print(f"generating N={args.n}, D={args.d}, queries={args.queries}...")
    t0 = time.perf_counter()
    data = rng.standard_normal((args.n + args.queries, args.d), dtype=np.float32)
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    data /= norms
    db = np.ascontiguousarray(data[:args.n])
    queries = np.ascontiguousarray(data[args.n:args.n + args.queries])
    print(f"  generated in {time.perf_counter() - t0:.2f}s "
          f"(corpus = {db.nbytes / 1e6:.0f} MB)")

    # Ground truth via Faiss FlatIP (exact).
    print("computing ground truth (faiss.FlatIP)...")
    t0 = time.perf_counter()
    gt_index = build_faiss_flatip(db)
    truth = run_faiss_flatip(gt_index, queries, args.k, threads=0)
    print(f"  ground truth in {time.perf_counter() - t0:.2f}s")

    # Build all indexes (timed separately from queries).
    builds = {"faiss_flat": gt_index}
    print("building faiss.FlatIP...",      flush=True); bt0 = time.perf_counter()
    builds["faiss_flat"]   = build_faiss_flatip(db);  t_ff = time.perf_counter() - bt0
    print(f"  built in {t_ff:.2f}s",       flush=True)
    print("building faiss.IVF nprobe=32...", flush=True); bt0 = time.perf_counter()
    builds["faiss_ivf32"]  = build_faiss_ivf(db, 32); t_iv32 = time.perf_counter() - bt0
    print(f"  built in {t_iv32:.2f}s",     flush=True)
    print("building faiss.IVF nprobe=64...", flush=True); bt0 = time.perf_counter()
    builds["faiss_ivf64"]  = build_faiss_ivf(db, 64); t_iv64 = time.perf_counter() - bt0
    print(f"  built in {t_iv64:.2f}s",     flush=True)
    print("building hnswlib ef=64...",     flush=True); bt0 = time.perf_counter()
    builds["hnsw_64"]      = build_hnswlib(db, 64);   t_h64  = time.perf_counter() - bt0
    print(f"  built in {t_h64:.2f}s",      flush=True)
    print("building hnswlib ef=128...",    flush=True); bt0 = time.perf_counter()
    builds["hnsw_128"]     = build_hnswlib(db, 128);  t_h128 = time.perf_counter() - bt0
    print(f"  built in {t_h128:.2f}s",     flush=True)

    t_lc = 0.0
    if not args.skip_lc:
        print(f"building LC at N={args.n}... (this can take minutes)")
        bt0 = time.perf_counter()
        builds["lc"] = build_listofclusters(db)
        t_lc = time.perf_counter() - bt0
        print(f"  LC built in {t_lc:.2f}s")

    methods = []
    if not args.skip_lc:
        methods += [
            ("LC, 1T",                 lambda: run_listofclusters(builds["lc"], queries, args.k, nthreads=1),         t_lc),
            ("LC, HW threads",         lambda: run_listofclusters(builds["lc"], queries, args.k, nthreads=0),         t_lc),
        ]
    methods += [
        ("faiss.FlatIP 1T",            lambda: run_faiss_flatip(builds["faiss_flat"], queries, args.k, threads=1),    t_ff),
        ("faiss.FlatIP HW",            lambda: run_faiss_flatip(builds["faiss_flat"], queries, args.k, threads=0),    t_ff),
        ("faiss.IVFFlat np=32 HW",     lambda: run_faiss_ivf(builds["faiss_ivf32"], queries, args.k, threads=0),      t_iv32),
        ("faiss.IVFFlat np=64 HW",     lambda: run_faiss_ivf(builds["faiss_ivf64"], queries, args.k, threads=0),      t_iv64),
        ("hnswlib ef=64 HW",           lambda: run_hnswlib(builds["hnsw_64"], queries, args.k, threads=0),            t_h64),
        ("hnswlib ef=128 HW",          lambda: run_hnswlib(builds["hnsw_128"], queries, args.k, threads=0),           t_h128),
    ]

    print(f"\n{'method':<30s}  {'recall@k':>9s}  {'qps':>10s}  {'query s':>8s}  {'build s':>8s}")
    rows = []
    for name, fn, build_s in methods:
        try:
            got, elapsed = time_call(fn)
            qps = queries.shape[0] / elapsed
            r = recall(np.asarray(got), truth)
            rows.append({"method": name, "recall": float(r), "qps": float(qps),
                         "sec": float(elapsed), "build_s": float(build_s)})
            print(f"  {name:<30s}  {r:9.3f}  {qps:10.1f}  {elapsed:8.3f}  {build_s:8.3f}")
        except Exception as e:
            print(f"  {name:<30s}  skipped: {e}")

    csv_path = Path(__file__).parent / "compare_synthetic.csv"
    with csv_path.open("w") as f:
        w = csv.DictWriter(f, fieldnames=["method", "recall", "qps", "sec", "build_s"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nwrote {csv_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
