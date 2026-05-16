#!/usr/bin/env python3
"""Pareto comparison of liblistofclusters against the standard ANN libraries.

Runs every method available at multiple parameter settings, measures
(recall@k, queries/sec) per setting, writes results to a CSV, and renders a
log-scale recall-vs-throughput plot in the ann-benchmarks style.

What it compares (when each library is installed):

  - liblistofclusters  (this library, via the Python wrapper)
  - Faiss IndexIVFFlat (Meta)
  - Faiss IndexHNSWFlat
  - Annoy              (Spotify)
  - hnswlib            (the canonical HNSW)
  - sklearn KDTree     (low-D exact)
  - sklearn BallTree   (general-metric exact)
  - Brute force        (numpy, the reference)

Usage:
  pip install numpy matplotlib pandas faiss-cpu annoy scikit-learn hnswlib
  python bench/compare.py --n 10000 --d 8 --queries 200 --k 10
  open bench/compare.png

Optional flags:
  --gen {uniform,clustered}   data distribution (default uniform)
  --csv PATH                  output CSV path (default bench/compare.csv)
  --plot PATH                 output PNG path (default bench/compare.png)
  --include lc,faiss,...      comma-separated subset of libraries

Methods/parameters can be edited at the top of this file - the structure is
deliberately simple so adding a new library is one block.
"""
from __future__ import annotations

import argparse
import csv
import importlib
import time
from dataclasses import dataclass

import numpy as np


# --------------------------------------------------------------------------- #
# Data + ground truth
# --------------------------------------------------------------------------- #

def make_data(n: int, d: int, gen: str, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if gen == "clustered":
        n_clusters = 64
        sigma = 0.05
        centers = rng.uniform(-1.0, 1.0, size=(n_clusters, d))
        which = rng.integers(0, n_clusters, size=n)
        noise = rng.normal(0.0, sigma, size=(n, d))
        return centers[which] + noise
    return rng.uniform(-1.0, 1.0, size=(n, d))


def brute_topk(db: np.ndarray, queries: np.ndarray, k: int) -> np.ndarray:
    """Returns (Q, k) array of true nearest-neighbor ids per query."""
    # Vectorized squared L2 (avoid sqrt - relative order is the same).
    d2 = ((db[None, :, :] - queries[:, None, :]) ** 2).sum(axis=2)
    return np.argpartition(d2, kth=k, axis=1)[:, :k]


def recall_at_k(got: list[list[int]], gt: np.ndarray) -> float:
    """Mean fraction of true neighbors recovered, averaged over queries."""
    hits = 0
    total = 0
    for i, g in enumerate(gt):
        truth = set(int(x) for x in g)
        for gi in got[i]:
            if int(gi) in truth:
                hits += 1
        total += len(truth)
    return hits / total if total else 0.0


# --------------------------------------------------------------------------- #
# Method runners. Each returns a list of (params, recall, qps) tuples.
# --------------------------------------------------------------------------- #

@dataclass
class Result:
    method: str
    params: str
    recall: float
    qps: float
    notes: str = ""


def _time_queries(fn, queries, label):
    """fn(q) -> list[int] of neighbor ids. Times the whole query batch."""
    got = []
    t0 = time.perf_counter()
    for q in queries:
        got.append(fn(q))
    elapsed = time.perf_counter() - t0
    qps = len(queries) / elapsed if elapsed > 0 else float("inf")
    return got, qps


def run_brute(db, queries, k, gt) -> list[Result]:
    def query(q):
        d2 = ((db - q) ** 2).sum(axis=1)
        return np.argpartition(d2, kth=k)[:k].tolist()
    got, qps = _time_queries(query, queries, "brute")
    return [Result("brute", "numpy", recall_at_k(got, gt), qps)]


def run_sklearn(db, queries, k, gt) -> list[Result]:
    try:
        from sklearn.neighbors import KDTree, BallTree
    except ImportError:
        return []
    results = []
    for cls, name in [(KDTree, "sklearn.KDTree"), (BallTree, "sklearn.BallTree")]:
        for leaf in (16, 32, 64):
            tree = cls(db, leaf_size=leaf)
            def query(q, tree=tree):
                _, idx = tree.query(q.reshape(1, -1), k=k)
                return idx[0].tolist()
            got, qps = _time_queries(query, queries, name)
            results.append(Result(name, f"leaf={leaf}", recall_at_k(got, gt), qps))
    return results


def run_faiss(db, queries, k, gt) -> list[Result]:
    try:
        import faiss
    except ImportError:
        return []
    results = []
    n, d = db.shape
    db32 = db.astype(np.float32)
    q32 = queries.astype(np.float32)

    # IndexFlatL2 (exact brute force inside Faiss - SIMD-optimised).
    idx = faiss.IndexFlatL2(d)
    idx.add(db32)
    t0 = time.perf_counter()
    _, I = idx.search(q32, k)
    qps = len(queries) / max(time.perf_counter() - t0, 1e-9)
    results.append(Result("faiss.FlatL2", "exact",
                          recall_at_k(I.tolist(), gt), qps))

    # IndexIVFFlat sweeps nprobe.
    nlist = max(int(np.sqrt(n)), 4)
    quantizer = faiss.IndexFlatL2(d)
    ivf = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
    ivf.train(db32)
    ivf.add(db32)
    for nprobe in (1, 4, 10, 32, nlist):
        ivf.nprobe = int(nprobe)
        t0 = time.perf_counter()
        _, I = ivf.search(q32, k)
        qps = len(queries) / max(time.perf_counter() - t0, 1e-9)
        results.append(Result("faiss.IVFFlat", f"nlist={nlist} nprobe={nprobe}",
                              recall_at_k(I.tolist(), gt), qps))

    # IndexHNSWFlat sweeps efSearch.
    hnsw = faiss.IndexHNSWFlat(d, 16, faiss.METRIC_L2)
    hnsw.hnsw.efConstruction = 200
    hnsw.add(db32)
    for ef in (16, 32, 64, 128, 256):
        hnsw.hnsw.efSearch = int(ef)
        t0 = time.perf_counter()
        _, I = hnsw.search(q32, k)
        qps = len(queries) / max(time.perf_counter() - t0, 1e-9)
        results.append(Result("faiss.HNSW", f"M=16 ef={ef}",
                              recall_at_k(I.tolist(), gt), qps))
    return results


def run_hnswlib(db, queries, k, gt) -> list[Result]:
    try:
        import hnswlib
    except ImportError:
        return []
    results = []
    n, d = db.shape
    p = hnswlib.Index(space="l2", dim=d)
    p.init_index(max_elements=n, ef_construction=200, M=16)
    p.add_items(db, np.arange(n))
    for ef in (16, 32, 64, 128, 256):
        p.set_ef(int(ef))
        def query(q, p=p):
            ids, _ = p.knn_query(q.reshape(1, -1), k=k)
            return ids[0].tolist()
        got, qps = _time_queries(query, queries, "hnswlib")
        results.append(Result("hnswlib", f"M=16 ef={ef}",
                              recall_at_k(got, gt), qps))
    return results


def run_annoy(db, queries, k, gt) -> list[Result]:
    """Spotify's Annoy. NOTE: annoy 1.17.3 (latest on PyPI as of 2024) has a
    known bug on Python 3.13 / numpy 2.x where queries always return a list
    of one item, breaking recall. Use Python 3.11/3.12 if you want annoy in
    the comparison. We sanity-check on a known query and skip if broken."""
    try:
        from annoy import AnnoyIndex
    except ImportError:
        return []

    # Sanity probe: a query for an inserted item should return that item.
    probe = AnnoyIndex(4, "euclidean")
    probe.add_item(0, [0.0, 0.0, 0.0, 0.0])
    probe.add_item(1, [1.0, 1.0, 1.0, 1.0])
    probe.build(5)
    if 0 not in probe.get_nns_by_vector([0.0, 0.0, 0.0, 0.0], 2):
        print("  annoy is broken on this Python/numpy combo; skipping")
        return []

    results = []
    n, d = db.shape
    for n_trees in (10, 50, 100):
        idx = AnnoyIndex(d, "euclidean")
        for i, v in enumerate(db):
            idx.add_item(i, v.tolist())
        idx.build(n_trees)
        for search_k in (-1, 100, 1000):
            def query(q, idx=idx, search_k=search_k):
                return idx.get_nns_by_vector(q.tolist(), k, search_k=search_k)
            got, qps = _time_queries(query, queries, "annoy")
            label = f"trees={n_trees} search_k={search_k}"
            results.append(Result("annoy", label, recall_at_k(got, gt), qps))
    return results


def run_lc(db, queries, k, gt) -> list[Result]:
    """Our own library, via the Python wrapper."""
    try:
        from listofclusters import Index
    except ImportError:
        return []
    results = []
    ids = np.arange(len(db), dtype=np.uint32)
    # Default LC (incremental insert).
    for metric in ("euclidean",):
        idx = Index(metric=metric)
        idx.bulk_build(db, ids)
        def query(q, idx=idx):
            nbrs, _ = idx.knn(q, k=k)
            return list(nbrs)
        got, qps = _time_queries(query, queries, "lc")
        results.append(Result("liblistofclusters", f"metric={metric} 1T",
                              recall_at_k(got, gt), qps))
        # Batched (multi-threaded).
        t0 = time.perf_counter()
        batch_nbrs, _ = idx.batch_knn(queries, k=k, nthreads=0)
        qps = len(queries) / max(time.perf_counter() - t0, 1e-9)
        got = [list(b) for b in batch_nbrs]
        results.append(Result("liblistofclusters", f"metric={metric} batch HW threads",
                              recall_at_k(got, gt), qps))
    return results


RUNNERS = {
    "brute":   run_brute,
    "sklearn": run_sklearn,
    "faiss":   run_faiss,
    "hnswlib": run_hnswlib,
    "annoy":   run_annoy,
    "lc":      run_lc,
}


# --------------------------------------------------------------------------- #
# Plot
# --------------------------------------------------------------------------- #

def plot_pareto(results: list[Result], path: str, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_method: dict[str, list[Result]] = {}
    for r in results:
        by_method.setdefault(r.method, []).append(r)

    fig, ax = plt.subplots(figsize=(9, 6.5))
    cmap = plt.get_cmap("tab10")
    for i, (name, rs) in enumerate(sorted(by_method.items())):
        xs = [r.recall for r in rs]
        ys = [r.qps for r in rs]
        ax.scatter(xs, ys, s=60, color=cmap(i % 10), label=name, edgecolors="black", linewidths=0.5)
    ax.set_xlabel("Recall@k (1.0 = exact)")
    ax.set_ylabel("Queries per second")
    ax.set_yscale("log")
    ax.set_xlim(-0.02, 1.02)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_title(title)
    ax.legend(loc="lower left", framealpha=0.9, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Pareto comparison of ANN libraries.")
    parser.add_argument("--n",       type=int, default=10000)
    parser.add_argument("--d",       type=int, default=8)
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--k",       type=int, default=10)
    parser.add_argument("--gen",     choices=["uniform", "clustered"], default="uniform")
    parser.add_argument("--csv",     default="bench/compare.csv")
    parser.add_argument("--plot",    default="bench/compare.png")
    parser.add_argument("--include", default=",".join(RUNNERS.keys()),
                        help="comma-separated subset of methods (default: all)")
    args = parser.parse_args()

    db      = make_data(args.n, args.d, args.gen, seed=42)
    queries = make_data(args.queries, args.d, args.gen, seed=9999)

    print(f"Computing brute-force ground truth (n={args.n}, q={args.queries}, k={args.k})...")
    gt = brute_topk(db, queries, args.k)

    libs = [x.strip() for x in args.include.split(",")]
    all_results: list[Result] = []
    for name in libs:
        runner = RUNNERS.get(name)
        if runner is None:
            print(f"  unknown library: {name}, skipping")
            continue
        print(f"Running {name}...")
        rs = runner(db, queries, args.k, gt)
        if not rs:
            print(f"  {name}: not installed, skipping")
            continue
        for r in rs:
            print(f"  {r.method:28s} {r.params:30s} recall={r.recall:.3f} qps={r.qps:8.1f}")
            all_results.append(r)

    with open(args.csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "params", "recall", "qps"])
        for r in all_results:
            w.writerow([r.method, r.params, f"{r.recall:.4f}", f"{r.qps:.2f}"])
    print(f"\nCSV: {args.csv}")

    title = f"ANN comparison: N={args.n} D={args.d} k={args.k} gen={args.gen}"
    plot_pareto(all_results, args.plot, title)
    print(f"Plot: {args.plot}")


if __name__ == "__main__":
    main()
