"""Basic smoke tests for the listofclusters Python wrapper.

Run with: python3 -m pytest python/tests/ -v
"""
import numpy as np
import pytest

from listofclusters import Index, available_metrics, supported_metrics


def test_available_metrics_is_non_empty():
    metrics = available_metrics()
    assert len(metrics) > 0
    names = [m[0] for m in metrics]
    assert "euclidean" in names
    assert "levenshtein" in names


def test_supported_metrics_subset_of_available():
    avail = {m[0] for m in available_metrics()}
    for s in supported_metrics():
        assert s in avail, f"{s} not in available_metrics"


def test_unknown_metric_raises():
    with pytest.raises(ValueError):
        Index(metric="cosine_similarity")  # not a metric


def test_empty_index():
    idx = Index(metric="euclidean")
    assert idx.empty
    assert len(idx) == 0
    assert idx.metric == "euclidean"


def test_online_insert_and_knn():
    rng = np.random.default_rng(42)
    db = rng.uniform(-1.0, 1.0, size=(64, 8))
    idx = Index(metric="euclidean")
    for i, v in enumerate(db):
        idx.insert(v, np.uint32(i))
    # 1-NN of any point is itself (distance 0).
    nbrs, dists = idx.knn(db[7], k=1)
    assert 7 in nbrs


def test_batch_insert_and_batch_knn():
    rng = np.random.default_rng(0)
    db = rng.uniform(-1.0, 1.0, size=(300, 6))
    ids = np.arange(len(db), dtype=np.uint32)

    idx = Index(metric="euclidean")
    idx.insert_batch(db, ids)
    assert not idx.empty

    Q = rng.uniform(-1.0, 1.0, size=(10, 6))
    batch_nbrs, batch_dists = idx.batch_knn(Q, k=5, nthreads=2)
    assert len(batch_nbrs) == 10
    assert all(len(r) <= 5 for r in batch_nbrs)


def test_knn_matches_brute_force():
    rng = np.random.default_rng(11)
    db = rng.uniform(-1.0, 1.0, size=(200, 4))
    ids = np.arange(len(db), dtype=np.uint32)

    idx = Index(metric="euclidean")
    idx.bulk_build(db, ids)

    for _ in range(5):
        q = rng.uniform(-1.0, 1.0, size=4)
        nbrs, dists = idx.knn(q, k=5)

        # Brute-force ground truth.
        bf = np.linalg.norm(db - q, axis=1)
        expected = np.argsort(bf)[:5]

        # All ground-truth ids must be in the returned set.
        nbrs_set = set(nbrs)
        for e in expected:
            assert e in nbrs_set, f"{e} missing from {nbrs_set}"


def test_range_search():
    rng = np.random.default_rng(123)
    db = rng.uniform(0.0, 1.0, size=(50, 4))
    ids = np.arange(len(db), dtype=np.uint32)

    idx = Index(metric="euclidean")
    idx.insert_batch(db, ids)

    q = db[0]
    nbrs, dists = idx.range(q, radius=0.5)
    # Ground truth.
    truth = np.where(np.linalg.norm(db - q, axis=1) <= 0.5)[0]
    nbrs_set = set(nbrs)
    for t in truth:
        assert t in nbrs_set


def test_remove():
    rng = np.random.default_rng(7)
    db = rng.uniform(-1.0, 1.0, size=(20, 4))
    ids = np.arange(len(db), dtype=np.uint32)
    idx = Index(metric="euclidean")
    idx.insert_batch(db, ids)

    # Remove half via batch.
    idx.remove_batch(db[:10], ids[:10])

    # Removed ids should no longer be findable as 1-NN of themselves.
    for i in range(10):
        nbrs, _ = idx.knn(db[i], k=1)
        assert i not in nbrs


def test_freeze_amortizes_first_query():
    rng = np.random.default_rng(11)
    db = rng.uniform(-1.0, 1.0, size=(200, 4))
    ids = np.arange(len(db), dtype=np.uint32)

    idx = Index(metric="euclidean")
    idx.bulk_build(db, ids)
    idx.freeze()  # idempotent — bulk_build already refreshed

    for _ in range(5):
        q = rng.uniform(-1.0, 1.0, size=4)
        nbrs, _ = idx.knn(q, k=5)
        bf = np.linalg.norm(db - q, axis=1)
        expected = np.argsort(bf)[:5]
        for e in expected:
            assert e in set(nbrs), f"frozen-index knn missed expected NN {e}"


def test_each_supported_metric_constructible():
    for name in supported_metrics():
        idx = Index(metric=name)
        assert idx.metric == name
        # Insert a couple of points and run a query.
        idx.insert(np.array([0.1, 0.2, 0.3, 0.4]), np.uint32(0))
        idx.insert(np.array([0.4, 0.3, 0.2, 0.1]), np.uint32(1))
        nbrs, _ = idx.knn(np.array([0.1, 0.2, 0.3, 0.4]), k=1)
        assert 0 in nbrs
