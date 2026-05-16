# liblistofclusters

Header-only C++23 template library implementing the [List of Clusters][lc]
metric-space index (Chávez & Navarro, 2005), with a Python wrapper.

## Layout

```
include/listofclusters/   Public headers (the library itself)
tests/                    Unit tests using synthetic data, no external deps
bench/                    Benchmarks (in-tree C++ + cross-library Python). See bench/README.md
python/                   Python wrapper via nanobind
third_party/hnswlib/      Vendored HNSW reference (MIT) used in the bench
```

## Build

```sh
make test         # build + run the test suite (no external deps)
make bench        # build + run the C++ bench
make all          # everything
```

Or with CMake:

```sh
cmake -S . -B build -DLISTOFCLUSTERS_BUILD_BENCH=ON
cmake --build build
ctest --test-dir build
```

## Library use

```cpp
#include <listofclusters/listofclusters.hh>
#include <listofclusters/metrics.hh>

using vec_t = std::vector<double>;

metric::listofclusters<vec_t, metric::euclidean, /*bucket_size=*/20, /*overflow=*/80> idx;

// Online: one at a time.
idx.insert(point, id);
auto results = idx.knn_search(query, query_id, /*k=*/10);

// Batch (sequential):
idx.insert(points, ids);
idx.remove(points_to_drop, ids_to_drop);

// Batch query, parallel across hardware threads:
auto batch = idx.batch_knn(queries, /*start_qid=*/0, /*k=*/10);

// Bulk build (replaces existing state):
idx.bulk_build(all_points, all_ids);
```

## Built-in metrics

`<listofclusters/metrics.hh>` ships stateless functors. **Every one of these
is a true metric** — non-negative, identity, symmetric, triangle inequality
— which is what LC's pruning correctness depends on.

| Functor | Distance | Domain |
|---|---|---|
| `metric::euclidean` | L2 = `sqrt(Σ (a_i − b_i)²)` | real vectors |
| `metric::manhattan` | L1 = `Σ ǀa_i − b_iǀ` | real vectors |
| `metric::chebyshev` | L∞ = `max ǀa_i − b_iǀ` | real vectors |
| `metric::minkowski<p>` | Lp = `(Σ ǀa_i − b_iǀᵖ)^(1/p)`, `p ≥ 1` | real vectors |
| `metric::hamming` | count of positions where `a_i ≠ b_i` | any equality-comparable |
| `metric::angular` | `arccos(⟨a,b⟩ / (ǀaǀ ǀbǀ))` | non-zero vectors |
| `metric::jaccard` | `1 − ǀA∩Bǀ / ǀA∪Bǀ` on binary vectors | sparse sets |
| `metric::canberra` | `Σ ǀa_i − b_iǀ / (ǀa_iǀ + ǀb_iǀ)`, `0/0 ↦ 0` | non-negative reals |
| `metric::levenshtein` | edit distance (insert/delete/substitute) | strings, integer sequences |

The set is also exposed as a compile-time registry,
`metric::available_metrics[]` (with `name`, `description`, `domain` for each).
Useful for consumers (e.g. the future Python wrapper) that need to
enumerate what's available without hand-listing.

### What's deliberately NOT here

These are commonly mistaken for metrics but are *not* — using them with this
library will silently produce wrong results:

- **Squared Euclidean** (L2 without the `sqrt`): violates the triangle inequality.
- **Cosine "distance"** defined as `1 − cos(a, b)`: not a metric in general.
  Use `metric::angular` instead.
- **KL divergence**, **Bregman divergences**: asymmetric.
- **Dot product**, **inner product** (larger = closer): not a distance at all.

If you need to plug in your own metric, write a stateless functor that
satisfies `metric::Metric<M, O>` (see `glob.hh` for the formal axioms) and
pass it as the `distance_t` template parameter.

[lc]: https://users.dcc.uchile.cl/~gnavarro/abstracts/prl04.html
