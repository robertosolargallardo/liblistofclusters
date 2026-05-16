"""liblistofclusters Python bindings.

Exact metric-space nearest-neighbor index. See README for the speed/recall
positioning vs HNSW and brute-force baselines.

Quickstart
----------

    >>> import numpy as np
    >>> from listofclusters import Index, available_metrics, supported_metrics
    >>>
    >>> db = np.random.uniform(-1, 1, size=(10_000, 8))
    >>> ids = np.arange(len(db), dtype=np.uint32)
    >>>
    >>> idx = Index(metric="euclidean")
    >>> idx.bulk_build(db, ids)
    >>>
    >>> q = np.random.uniform(-1, 1, size=8)
    >>> nbrs, dists = idx.knn(q, k=10)
    >>>
    >>> Q = np.random.uniform(-1, 1, size=(100, 8))
    >>> batch_nbrs, batch_dists = idx.batch_knn(Q, k=10, nthreads=0)
"""

from ._listofclusters import (
    Index,
    available_metrics,
    supported_metrics,
)

__all__ = ["Index", "available_metrics", "supported_metrics"]
