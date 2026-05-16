#!/usr/bin/env python3
"""Pareto comparison of liblistofclusters against the standard ANN libraries.

Sweeps over (dataset size N, dimensionality D, metric). For each
(N, D, metric) cell, runs every applicable library at multiple parameter
settings, measures (recall@k, queries/sec), writes results to CSV, and
renders a per-metric grid plot in the ann-benchmarks style.

What it compares (when each library is installed):

  - liblistofclusters  (this library, via the Python wrapper)
  - Faiss IndexFlatL2  (SIMD brute force)
  - Faiss IndexIVFFlat
  - Faiss IndexHNSWFlat
  - hnswlib            (the canonical HNSW)
  - sklearn KDTree     (low-D exact)
  - sklearn BallTree   (general-metric exact, supports any of many metrics)
  - Annoy              (Spotify; auto-skipped if broken on the runtime)
  - Brute force        (numpy reference)

Usage with uv (recommended):

  # One-shot, no venv needed - uv resolves and runs in an ephemeral env:
  uv run --with numpy,matplotlib,scikit-learn,faiss-cpu,hnswlib,annoy \
         --with ./python \
         bench/compare.py --n 10000 --d 8 --metric euclidean

  # Persistent venv:
  uv venv && source .venv/bin/activate
  uv pip install numpy matplotlib scikit-learn faiss-cpu hnswlib annoy ./python
  python bench/compare.py --n 10000,50000 --d 8,32 --metric euclidean,manhattan,angular

Plain pip works too:
  pip install numpy matplotlib scikit-learn faiss-cpu hnswlib annoy
  pip install ./python    # for liblistofclusters
  python bench/compare.py --n 10000 --d 8 --metric euclidean

Sweep examples (libraries that don't support a metric are skipped, plot
is still produced from whatever does):
  python bench/compare.py --n 5000,10000,50000 --d 8,32,128 --metric euclidean
  python bench/compare.py --n 10000 --d 32 --metric manhattan
  python bench/compare.py --n 10000 --d 32 --metric angular

Outputs:
  bench/compare.csv             every (method, params, N, D, metric, recall, qps) row
  bench/compare_<metric>.png    grid plot (rows: N, cols: D) for that metric
"""
from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass, asdict

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


def pairwise_distance(db: np.ndarray, q: np.ndarray, metric: str) -> np.ndarray:
    """Compute distance from one query to every db row under the given metric."""
    if metric == "euclidean":
        return np.linalg.norm(db - q, axis=1)
    if metric == "manhattan":
        return np.abs(db - q).sum(axis=1)
    if metric == "chebyshev":
        return np.max(np.abs(db - q), axis=1)
    if metric == "angular":
        # Angular distance on unit sphere; works for any non-zero vectors.
        a_n = q / max(np.linalg.norm(q), 1e-12)
        b_n = db / np.linalg.norm(db, axis=1, keepdims=True).clip(min=1e-12)
        cos = np.clip(b_n @ a_n, -1.0, 1.0)
        return np.arccos(cos)
    raise ValueError(f"unsupported metric: {metric}")


def brute_topk(db: np.ndarray, queries: np.ndarray, k: int, metric: str) -> np.ndarray:
    """(Q, k) array of true nearest-neighbor ids per query under `metric`."""
    gt = np.empty((len(queries), k), dtype=np.int64)
    for i, q in enumerate(queries):
        d = pairwise_distance(db, q, metric)
        gt[i] = np.argpartition(d, kth=k)[:k]
    return gt


def recall_at_k(got: list[list[int]], gt: np.ndarray) -> float:
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
# Method runners. Each returns a list of Result for one (db, queries, metric).
# --------------------------------------------------------------------------- #

@dataclass
class Result:
    method: str
    params: str
    n: int
    d: int
    metric: str
    recall: float
    qps: float


def _time_queries(fn, queries):
    got = []
    t0 = time.perf_counter()
    for q in queries:
        got.append(fn(q))
    elapsed = time.perf_counter() - t0
    qps = len(queries) / elapsed if elapsed > 0 else float("inf")
    return got, qps


def run_brute(db, queries, k, gt, metric) -> list[Result]:
    n, d = db.shape
    def query(q):
        dist = pairwise_distance(db, q, metric)
        return np.argpartition(dist, kth=k)[:k].tolist()
    got, qps = _time_queries(query, queries)
    return [Result("brute", f"numpy/{metric}", n, d, metric, recall_at_k(got, gt), qps)]


def run_sklearn(db, queries, k, gt, metric) -> list[Result]:
    try:
        from sklearn.neighbors import KDTree, BallTree
    except ImportError:
        return []
    # KDTree only supports Minkowski (L_p); BallTree supports many.
    kd_metrics = {"euclidean": "euclidean", "manhattan": "manhattan", "chebyshev": "chebyshev"}
    bt_metrics = {"euclidean": "euclidean", "manhattan": "manhattan", "chebyshev": "chebyshev"}
    # sklearn doesn't ship "angular"; skip cleanly.
    results = []
    n, d = db.shape
    if metric in kd_metrics:
        for leaf in (32, 64):
            tree = KDTree(db, leaf_size=leaf, metric=kd_metrics[metric])
            def query(q, tree=tree):
                _, idx = tree.query(q.reshape(1, -1), k=k)
                return idx[0].tolist()
            got, qps = _time_queries(query, queries)
            results.append(Result("sklearn.KDTree", f"leaf={leaf}/{metric}",
                                   n, d, metric, recall_at_k(got, gt), qps))
    if metric in bt_metrics:
        for leaf in (32, 64):
            tree = BallTree(db, leaf_size=leaf, metric=bt_metrics[metric])
            def query(q, tree=tree):
                _, idx = tree.query(q.reshape(1, -1), k=k)
                return idx[0].tolist()
            got, qps = _time_queries(query, queries)
            results.append(Result("sklearn.BallTree", f"leaf={leaf}/{metric}",
                                   n, d, metric, recall_at_k(got, gt), qps))
    return results


def run_faiss(db, queries, k, gt, metric) -> list[Result]:
    try:
        import faiss
    except ImportError:
        return []
    # Faiss natively supports L2 and inner product. For angular we normalize
    # to unit sphere and use IP; recall is computed against angular ground
    # truth so the comparison is fair.
    n, d = db.shape
    results = []

    if metric == "euclidean":
        db32 = db.astype(np.float32)
        q32  = queries.astype(np.float32)
        # FlatL2.
        idx = faiss.IndexFlatL2(d)
        idx.add(db32)
        t0 = time.perf_counter()
        _, I = idx.search(q32, k)
        qps = len(queries) / max(time.perf_counter() - t0, 1e-9)
        results.append(Result("faiss.FlatL2", "exact", n, d, metric,
                              recall_at_k(I.tolist(), gt), qps))
        # IVFFlat.
        nlist = max(int(np.sqrt(n)), 4)
        quantizer = faiss.IndexFlatL2(d)
        ivf = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
        ivf.train(db32); ivf.add(db32)
        for nprobe in (1, 4, 10, 32, nlist):
            ivf.nprobe = int(nprobe)
            t0 = time.perf_counter()
            _, I = ivf.search(q32, k)
            qps = len(queries) / max(time.perf_counter() - t0, 1e-9)
            results.append(Result("faiss.IVFFlat", f"nlist={nlist} nprobe={nprobe}",
                                  n, d, metric, recall_at_k(I.tolist(), gt), qps))
        # HNSW.
        hnsw = faiss.IndexHNSWFlat(d, 16, faiss.METRIC_L2)
        hnsw.hnsw.efConstruction = 200
        hnsw.add(db32)
        for ef in (16, 32, 64, 128, 256):
            hnsw.hnsw.efSearch = int(ef)
            t0 = time.perf_counter()
            _, I = hnsw.search(q32, k)
            qps = len(queries) / max(time.perf_counter() - t0, 1e-9)
            results.append(Result("faiss.HNSW", f"M=16 ef={ef}", n, d, metric,
                                  recall_at_k(I.tolist(), gt), qps))
    elif metric == "angular":
        # Normalize then use FlatIP (a.k.a. cosine similarity).
        db32 = db.astype(np.float32)
        q32  = queries.astype(np.float32)
        faiss.normalize_L2(db32); faiss.normalize_L2(q32)
        idx = faiss.IndexFlatIP(d)
        idx.add(db32)
        t0 = time.perf_counter()
        _, I = idx.search(q32, k)
        qps = len(queries) / max(time.perf_counter() - t0, 1e-9)
        results.append(Result("faiss.FlatIP (cos)", "exact", n, d, metric,
                              recall_at_k(I.tolist(), gt), qps))
    # Faiss doesn't natively do L1; skip.
    return results


def run_hnswlib(db, queries, k, gt, metric) -> list[Result]:
    try:
        import hnswlib
    except ImportError:
        return []
    n, d = db.shape
    space_map = {"euclidean": "l2", "angular": "cosine"}
    if metric not in space_map:
        return []
    space = space_map[metric]
    p = hnswlib.Index(space=space, dim=d)
    p.init_index(max_elements=n, ef_construction=200, M=16)
    p.add_items(db, np.arange(n))
    results = []
    for ef in (16, 32, 64, 128, 256):
        p.set_ef(int(ef))
        def query(q, p=p):
            ids, _ = p.knn_query(q.reshape(1, -1), k=k)
            return ids[0].tolist()
        got, qps = _time_queries(query, queries)
        results.append(Result("hnswlib", f"{space}/M=16 ef={ef}",
                              n, d, metric, recall_at_k(got, gt), qps))
    return results


def run_annoy(db, queries, k, gt, metric) -> list[Result]:
    try:
        from annoy import AnnoyIndex
    except ImportError:
        return []
    space_map = {"euclidean": "euclidean", "manhattan": "manhattan", "angular": "angular"}
    if metric not in space_map:
        return []
    space = space_map[metric]
    # Sanity probe in the actual space (annoy 1.17.3 is broken on Py3.13
    # and the failure mode is "every query returns a 1-element list", which
    # we detect by asking for 2 neighbors and checking the count).
    probe = AnnoyIndex(4, space)
    probe.add_item(0, [1.0, 0.0, 0.0, 0.0])
    probe.add_item(1, [0.0, 1.0, 0.0, 0.0])
    probe.add_item(2, [0.0, 0.0, 1.0, 0.0])
    probe.build(5)
    probe_result = probe.get_nns_by_vector([1.0, 0.0, 0.0, 0.0], 3)
    if len(probe_result) < 2:
        print(f"  annoy is broken for space={space} on this runtime; skipping")
        return []

    n, d = db.shape
    results = []
    for n_trees in (10, 50, 100):
        idx = AnnoyIndex(d, space)
        for i, v in enumerate(db):
            idx.add_item(i, v.tolist())
        idx.build(n_trees)
        for search_k in (-1, 1000):
            def query(q, idx=idx, search_k=search_k):
                return idx.get_nns_by_vector(q.tolist(), k, search_k=search_k)
            got, qps = _time_queries(query, queries)
            results.append(Result("annoy", f"{space}/trees={n_trees} search_k={search_k}",
                                   n, d, metric, recall_at_k(got, gt), qps))
    return results


def run_lc(db, queries, k, gt, metric) -> list[Result]:
    """Our library, via the Python wrapper. Supported metrics are pulled
    from the wrapper's supported_metrics() at runtime."""
    try:
        from listofclusters import Index, supported_metrics
    except ImportError:
        return []
    if metric not in supported_metrics():
        return []
    n, d = db.shape
    ids = np.arange(n, dtype=np.uint32)
    idx = Index(metric=metric)
    idx.bulk_build(db, ids)
    results = []
    # 1T
    def query(q, idx=idx):
        nbrs, _ = idx.knn(q, k=k)
        return list(nbrs)
    got, qps = _time_queries(query, queries)
    results.append(Result("liblistofclusters", f"{metric}/1T",
                          n, d, metric, recall_at_k(got, gt), qps))
    # Batch HW threads
    t0 = time.perf_counter()
    batch_nbrs, _ = idx.batch_knn(queries, k=k, nthreads=0)
    qps = len(queries) / max(time.perf_counter() - t0, 1e-9)
    got = [list(b) for b in batch_nbrs]
    results.append(Result("liblistofclusters", f"{metric}/batch HW threads",
                          n, d, metric, recall_at_k(got, gt), qps))
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

def plot_grid(results: list[Result], metric: str, path: str):
    """One figure for `metric`, with subplots laid out by (N rows, D cols)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ns = sorted({r.n for r in results if r.metric == metric})
    ds = sorted({r.d for r in results if r.metric == metric})
    if not ns or not ds:
        print(f"  no results for metric={metric}, skipping plot")
        return

    fig, axes = plt.subplots(len(ns), len(ds),
                              figsize=(5.5 * len(ds), 4.5 * len(ns)),
                              squeeze=False, sharex=True)
    cmap = plt.get_cmap("tab10")
    # Stable color per method across subplots.
    methods = sorted({r.method for r in results if r.metric == metric})
    color_of = {m: cmap(i % 10) for i, m in enumerate(methods)}

    for ri, n in enumerate(ns):
        for ci, d in enumerate(ds):
            ax = axes[ri][ci]
            cell = [r for r in results if r.metric == metric and r.n == n and r.d == d]
            for m in methods:
                xs = [r.recall for r in cell if r.method == m]
                ys = [r.qps    for r in cell if r.method == m]
                if not xs: continue
                ax.scatter(xs, ys, s=55, color=color_of[m], label=m,
                            edgecolors="black", linewidths=0.5)
            ax.set_title(f"N={n}  D={d}", fontsize=10)
            ax.set_yscale("log")
            ax.set_xlim(-0.02, 1.02)
            ax.grid(True, which="both", alpha=0.3)
            if ri == len(ns) - 1: ax.set_xlabel("Recall@k")
            if ci == 0:          ax.set_ylabel("Queries/sec (log)")

    # One legend for the whole figure.
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=color_of[m],
                            markeredgecolor="black", markersize=8, label=m)
               for m in methods]
    fig.legend(handles=handles, loc="upper center", ncol=min(len(methods), 4),
               bbox_to_anchor=(0.5, 1.02), fontsize=9)
    fig.suptitle(f"ANN library comparison - metric={metric}", y=1.05)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_str_list(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(description="ANN library comparison sweep.")
    parser.add_argument("--n",       default="10000",
                        help="dataset sizes, comma-separated (e.g. 10000,100000)")
    parser.add_argument("--d",       default="8",
                        help="dimensionalities, comma-separated (e.g. 8,32,128)")
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--k",       type=int, default=10)
    parser.add_argument("--metric",  default="euclidean",
                        help="comma-separated: euclidean,manhattan,angular,chebyshev")
    parser.add_argument("--gen",     choices=["uniform", "clustered"], default="uniform")
    parser.add_argument("--csv",     default="bench/compare.csv")
    parser.add_argument("--plot-prefix", default="bench/compare",
                        help="output PNG basename; metric is appended: <prefix>_<metric>.png")
    parser.add_argument("--include", default=",".join(RUNNERS.keys()),
                        help="comma-separated subset of libraries (default: all)")
    args = parser.parse_args()

    ns      = parse_int_list(args.n)
    ds      = parse_int_list(args.d)
    metrics = parse_str_list(args.metric)
    libs    = parse_str_list(args.include)

    all_results: list[Result] = []

    for metric in metrics:
        for n in ns:
            for d in ds:
                db      = make_data(n, d, args.gen, seed=42)
                queries = make_data(args.queries, d, args.gen, seed=9999)
                print(f"\n=== N={n} D={d} metric={metric} ({args.gen}) ===")
                print("  ground truth...")
                gt = brute_topk(db, queries, args.k, metric)
                for lib in libs:
                    runner = RUNNERS.get(lib)
                    if runner is None:
                        continue
                    rs = runner(db, queries, args.k, gt, metric)
                    if not rs:
                        continue
                    for r in rs:
                        print(f"  {r.method:24s} {r.params:38s} recall={r.recall:.3f} qps={r.qps:9.1f}")
                        all_results.append(r)

    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(Result("", "", 0, 0, "", 0.0, 0.0)).keys()))
        w.writeheader()
        for r in all_results:
            w.writerow(asdict(r))
    print(f"\nCSV: {args.csv}")

    for metric in metrics:
        path = f"{args.plot_prefix}_{metric}.png"
        plot_grid(all_results, metric, path)
        print(f"Plot: {path}")


if __name__ == "__main__":
    main()
