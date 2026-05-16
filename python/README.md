# listofclusters (Python wrapper)

Python bindings for [liblistofclusters](..), an exact metric-space
nearest-neighbor index. Built with [nanobind](https://github.com/wjakob/nanobind)
+ [scikit-build-core](https://github.com/scikit-build/scikit-build-core).

## Install

```sh
# from this directory:
pip install .
```

Requires Python ≥ 3.11, a C++23 compiler, and CMake ≥ 3.20.

## Quickstart

```python
import numpy as np
from listofclusters import Index, available_metrics, supported_metrics

db  = np.random.uniform(-1, 1, size=(10_000, 8))
ids = np.arange(len(db), dtype=np.uint32)

idx = Index(metric="euclidean")
idx.bulk_build(db, ids)

q = np.random.uniform(-1, 1, size=8)
nbrs, dists = idx.knn(q, k=10)

Q = np.random.uniform(-1, 1, size=(100, 8))
batch_nbrs, batch_dists = idx.batch_knn(Q, k=10, nthreads=0)  # 0 = HW threads
```

## API

| Operation | Call |
|---|---|
| Online insert | `idx.insert(v, id)` |
| Batch insert | `idx.insert_batch(V, ids)` |
| Online remove | `idx.remove(v, id)` |
| Batch remove | `idx.remove_batch(V, ids)` |
| Bulk build (replaces state) | `idx.bulk_build(V, ids, use_pivots=False)` |
| kNN query | `nbrs, dists = idx.knn(q, k=10)` |
| Range query | `nbrs, dists = idx.range(q, radius)` |
| Parallel batch kNN | `nbrs, dists = idx.batch_knn(Q, k=10, nthreads=0)` |
| Clear | `idx.clear()` |
| Cluster count | `len(idx)` / `idx.cluster_count` |
| Metric name | `idx.metric` |

Module-level:
- `available_metrics()` — full list of metrics in the C++ library as
  `(name, description, domain)` tuples
- `supported_metrics()` — subset that this wrapper can instantiate
  (currently: euclidean, manhattan, chebyshev, canberra)

## What's not bound yet

- Custom Python-side metrics (you must use a C++ metric from the registry)
- `bucket_size` is fixed at 20 in this build
- `range_search` returns Python lists, not NumPy arrays (knn does too — easy
  follow-up)
- HC block index opt-in
- Hamming / Levenshtein metrics (not numeric vectors)
- SIMD-specialized Euclidean (requires fixed-D)

## Tests

```sh
pip install pytest
pytest tests/ -v
```
