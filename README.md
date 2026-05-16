# liblistofclusters

Header-only C++23 template library implementing the [List of Clusters][lc]
metric-space index (Chávez & Navarro, 2005), with example applications and
(in progress) a Python wrapper.

> **Status:** undergoing a phased refactor from research code into a
> production-quality library. See `tests/` for current behavioral guarantees.

## Layout

```
include/listofclusters/   Public headers (the library itself)
tests/                    Unit tests using synthetic data, no external deps
examples/
  cache/                  Indexed lookup + caching layer
  dbscan/                 DBSCAN clustering via LC range queries
  baseline_brute_force/   O(n²) brute-force baseline (does not use the library)
bench/                    Benchmarks (placeholder)
python/                   Python wrapper (placeholder)
```

## Build

```sh
make test         # build + run the test suite (no external deps)
make examples     # build the example apps (requires libarmadillo)
make all          # everything
```

Or with CMake:

```sh
cmake -S . -B build -DLISTOFCLUSTERS_BUILD_EXAMPLES=ON
cmake --build build
ctest --test-dir build
```

## Library use

```cpp
#include <listofclusters/listofclusters.hh>

using vec_t = std::vector<double>;
double euclid(vec_t a, vec_t b) { /* ... */ }

metric::listofclusters<vec_t, euclid, /*bucket_size=*/4, /*overflow=*/10> idx;
idx.insert(point, id);
auto results = idx.knn_search(query, query_id, k);
```

[lc]: https://users.dcc.uchile.cl/~gnavarro/abstracts/prl04.html
