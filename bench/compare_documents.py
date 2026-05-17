"""Real-document benchmark: angular search on arXiv chunks (Jina v2, 768-dim).

Methods compared:
  liblistofclusters (multiple LC configs incl. AESA-on/off)
  faiss.IndexFlatIP        (SIMD brute force on L2-normalized inputs)
  faiss.IndexIVFFlat       (nprobe sweep)
  hnswlib                  (ef sweep, cosine space)
  numpy brute              (ground truth)

Outputs:
  bench/compare_documents.csv
  bench/compare_documents_pareto.png

Usage:
  python bench/compare_documents.py [--queries 200] [--k 10]

LC's bindings currently expose euclidean/manhattan/chebyshev/canberra.
For angular search on L2-normalized embeddings we use 'euclidean' as a
ranking-equivalent stand-in (d_eucl(a,b)^2 = 2 - 2*cos(a,b) on unit vectors,
so argsort by Euclidean = argsort by angular for normalized inputs).
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

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


def brute_truth(db: np.ndarray, q: np.ndarray, k: int) -> np.ndarray:
    dots = q @ db.T
    dots = np.clip(dots, -1.0, 1.0)
    dists = np.arccos(dots)
    return np.argsort(dists, axis=1)[:, :k]


def recall(got: np.ndarray, truth: np.ndarray) -> float:
    hits = 0
    total = truth.shape[0] * truth.shape[1]
    for g, t in zip(got, truth):
        hits += len(set(int(x) for x in g) & set(int(x) for x in t))
    return hits / total


def time_call(fn):
    t0 = time.perf_counter()
    out = fn()
    t1 = time.perf_counter()
    return out, t1 - t0


def build_listofclusters(db):
    import listofclusters
    idx = listofclusters.Index(metric="euclidean")
    ids = np.arange(db.shape[0], dtype=np.uint32)
    db_c = np.ascontiguousarray(db, dtype=np.float64)
    idx.bulk_build(db_c, ids)
    idx.freeze()
    return idx


def run_listofclusters(idx, queries, k, *, nthreads=1):
    q_c = np.ascontiguousarray(queries, dtype=np.float64)
    nbrs2d, _ = idx.batch_knn(q_c, k=k, nthreads=nthreads)
    out = np.zeros((len(queries), k), dtype=np.uint32)
    for i, row in enumerate(nbrs2d):
        n = min(len(row), k)
        out[i, :n] = row[:n]
        if n < k:
            out[i, n:] = row[-1] if n > 0 else 0
    return out


def _set_faiss_threads(n):
    import faiss
    # n=0 means "use default" (omp_get_max_threads). For 1-thread comparison
    # we pin to 1; for HW-thread comparison we leave it at the default.
    if n > 0:
        faiss.omp_set_num_threads(n)
    else:
        # Restore default to avoid leaking a setting across runs.
        import os
        try:
            faiss.omp_set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 0)) or
                                       (os.cpu_count() or 1))
        except Exception:
            pass


def build_faiss_flatip(db):
    import faiss
    index = faiss.IndexFlatIP(db.shape[1])
    index.add(db)
    return index


def run_faiss_flatip(index, queries, k, threads=1):
    _set_faiss_threads(threads)
    return index.search(queries, k)[1]


def build_faiss_ivf(db, nprobe):
    import faiss
    nlist = max(4, int(np.sqrt(db.shape[0])))
    quant = faiss.IndexFlatIP(db.shape[1])
    index = faiss.IndexIVFFlat(quant, db.shape[1], nlist, faiss.METRIC_INNER_PRODUCT)
    index.train(db)
    index.add(db)
    index.nprobe = min(nprobe, nlist)
    return index


def run_faiss_ivf(index, queries, k, threads=1):
    _set_faiss_threads(threads)
    return index.search(queries, k)[1]


def build_hnswlib(db, ef):
    import hnswlib
    index = hnswlib.Index(space="cosine", dim=db.shape[1])
    index.init_index(max_elements=db.shape[0], ef_construction=200, M=16)
    index.add_items(db, num_threads=0)
    index.set_ef(ef)
    return index


def run_hnswlib(index, queries, k, threads=1):
    n = threads if threads > 0 else 0
    return index.knn_query(queries, k=k, num_threads=n)[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", type=int, default=200)
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    emb = load_embeddings()
    n_q = min(args.queries, emb.shape[0] // 4)
    db, queries = split(emb, n_q)
    print(f"loaded {emb.shape[0]} embeddings, dim={emb.shape[1]}, db={db.shape[0]}, q={queries.shape[0]}")

    truth = brute_truth(db, queries, args.k)

    rows = []
    # Build indexes once; time only the QUERY phase. Build times reported
    # separately so we don't penalize methods with expensive builds (LC's
    # bulk_build is O(N²/m) vs Faiss FlatIP's O(N) memcpy).
    print(f"\n{'method':<30s}  {'recall@k':>9s}  {'qps':>10s}  {'query s':>8s}  {'build s':>8s}")

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

    methods = [
        ("LC, 1T",                       lambda: run_listofclusters(builds["lc"], queries, args.k, nthreads=1),         t_lc),
        ("LC, HW threads",               lambda: run_listofclusters(builds["lc"], queries, args.k, nthreads=0),         t_lc),
        ("faiss.FlatIP 1T",              lambda: run_faiss_flatip(builds["faiss_flat"], queries, args.k, threads=1),    t_ff),
        ("faiss.FlatIP HW",              lambda: run_faiss_flatip(builds["faiss_flat"], queries, args.k, threads=0),    t_ff),
        ("faiss.IVFFlat np=10 1T",       lambda: run_faiss_ivf(builds["faiss_ivf10"], queries, args.k, threads=1),      t_iv10),
        ("faiss.IVFFlat np=10 HW",       lambda: run_faiss_ivf(builds["faiss_ivf10"], queries, args.k, threads=0),      t_iv10),
        ("faiss.IVFFlat np=32 1T",       lambda: run_faiss_ivf(builds["faiss_ivf32"], queries, args.k, threads=1),      t_iv32),
        ("faiss.IVFFlat np=32 HW",       lambda: run_faiss_ivf(builds["faiss_ivf32"], queries, args.k, threads=0),      t_iv32),
        ("hnswlib ef=64 1T",             lambda: run_hnswlib(builds["hnsw_64"], queries, args.k, threads=1),            t_h64),
        ("hnswlib ef=64 HW",             lambda: run_hnswlib(builds["hnsw_64"], queries, args.k, threads=0),            t_h64),
        ("hnswlib ef=128 1T",            lambda: run_hnswlib(builds["hnsw_128"], queries, args.k, threads=1),           t_h128),
        ("hnswlib ef=128 HW",            lambda: run_hnswlib(builds["hnsw_128"], queries, args.k, threads=0),           t_h128),
    ]
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

    csv_path = Path(__file__).parent / "compare_documents.csv"
    with csv_path.open("w") as f:
        w = csv.DictWriter(f, fieldnames=["method", "recall", "qps", "sec", "build_s"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nwrote {csv_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 6))
        for r in rows:
            ax.scatter(r["recall"], r["qps"], s=80)
            ax.annotate(r["method"], (r["recall"], r["qps"]), fontsize=7, alpha=0.8)
        ax.set_xlabel("Recall@k")
        ax.set_ylabel("QPS (log)")
        ax.set_yscale("log")
        ax.set_title(f"arXiv + Jina v2 angular search (N={db.shape[0]}, D={db.shape[1]}, k={args.k})")
        fig.tight_layout()
        png = Path(__file__).parent / "compare_documents_pareto.png"
        fig.savefig(png, dpi=120)
        print(f"wrote {png}")
    except ImportError:
        print("matplotlib not available; skipping plot")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
