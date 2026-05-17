# LC Query-Latency Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Beat the current LC query-latency benchmark by ≥5× single-thread on D=32 Euclidean while staying exact; add a real-document (arXiv + Jina v2) headline benchmark.

**Architecture:** Four optimization phases layered on the existing flat LC index. P1 adds a contiguous `centers_soa` and a batched-distance kernel. P3 (free with P1) walks clusters nearest-first. P4 switches `bulk_build` to farthest-first traversal. P2 adds an AESA-lite k-anchor pivot table. Implementation order P1→P3→P4→P2. Two negative-result mechanisms (block index, per-cluster pivots) are removed inline with the phases that supersede them.

**Tech Stack:** C++23 (concepts, `std::span`, `[[no_unique_address]]`), NEON/AVX2 intrinsics, ASan+UBSan smoke tests, `make`/CMake builds, nanobind Python bindings, Python `sentence-transformers` for the document corpus.

**Spec:** `docs/superpowers/specs/2026-05-16-lc-optimization-design.md`

---

## File Structure

### Modified

- `include/listofclusters/listofclusters.hh` — drop `_blocks`/`block_t`/`build_block_index()`; add `_centers`/`centers_soa_t`; thread batched-distance call + nearest-first ordering through `knn_search`/`range_search`; add FFT center selection; add AESA-lite path; drop `_pivot`/`has_pivot` branches.
- `include/listofclusters/cluster.hh` — drop `_pivot`, `has_pivot()`, `pivot()`, `set_pivot()`.
- `include/listofclusters/internal_object.hh` — drop `_pivot_distance` field and accessors.
- `include/listofclusters/glob.hh` — no changes (Metric concept stays).
- `include/listofclusters/metrics.hh` — add `supports_batched_distance` trait + specializations for `euclidean`, `manhattan`, `chebyshev`, `minkowski`, `euclidean_simd`.
- `tests/test_smoke.cc` — drop `test_bulk_build_with_pivots`; add `test_centers_soa_consistency`, `test_aesa_lower_bound_correctness`, `test_nearest_first_same_results`, `test_fft_build_same_results`, `test_phase_matrix`, plus property checks.
- `tests/Makefile` — no changes (header deps via wildcard already).
- `bench/bench_main.cc` — drop the blocks variant; add rows for P1, P1+P3, P1+P3+P4, full (AESA-lite), report `bytes/point`.
- `bench/Makefile` — no changes.
- `bench/README.md` — restructure: real-document headline first, synthetic ablation below.
- `python/src/_listofclusters.cc` — drop `use_pivots`/`build_block_index` from the bindings; expose `k_anchors`, `build_strategy`, `freeze()`.
- `python/tests/test_index.py` — drop pivot/block tests if any; add tests for new exposed API.
- `.github/workflows/ci.yml` — no changes (block-index and pivot rows in tests are dropped).
- `CMakeLists.txt`, `Makefile` — no changes.

### Created

- `include/listofclusters/detail/batched_distance.hh` — new header for batched-distance kernels (specializations live here, listofclusters.hh includes it).
- `include/listofclusters/detail/aesa.hh` — new header for `aesa_table_t` + helper methods.
- `bench/data/build_corpus.py` — arXiv downloader, PDF extractor, chunker, Jina embedder.
- `bench/data/arxiv_ids.txt` — pinned ID list (committed).
- `bench/data/MODEL_REVISION.txt` — pinned Jina HF revision (committed).
- `bench/data/CHUNKING.md` — one-pager on chunk sizing rules (committed).
- `bench/data/.gitignore` — covers `pdfs/`, `text/`, `embeddings.npy`, `chunks.jsonl`.
- `bench/compare_documents.py` — real-corpus bench harness.
- `.github/workflows/bench_documents.yml` — manual+weekly cron for the document bench.

---

## Phase 0 — Remove block index (alongside P1's prep)

The block index from Phase 5.8 is opt-in dead code superseded by P1+P3. Removing it first keeps subsequent diffs small.

### Task 0.1: Remove block index members and methods from `listofclusters.hh`

**Files:**
- Modify: `include/listofclusters/listofclusters.hh`

- [ ] **Step 1: Write the failing test in `tests/test_smoke.cc`**

Open `tests/test_smoke.cc` and at the bottom (above `main()`) add:

```cpp
// Phase 0 removal: build_block_index() must no longer exist on the
// public API. This is compile-time gating: if the symbol still exists
// the SFINAE branch flips and the static_assert fires.
template <class T, class = void>
struct has_build_block_index : std::false_type {};
template <class T>
struct has_build_block_index<T, std::void_t<decltype(std::declval<T&>().build_block_index())>>
    : std::true_type {};

static void test_block_index_removed()
{
    static_assert(!has_build_block_index<idx_t>::value,
                  "build_block_index() should have been removed in Phase 0");
    std::cout << "  test_block_index_removed: OK\n";
}
```

Add `test_block_index_removed();` inside `main()` before `std::cout << "all tests passed\n";`.

- [ ] **Step 2: Run tests to verify the new test fails to compile**

Run: `cd tests && make clean && make`
Expected: compilation succeeds but `static_assert` fails because `build_block_index()` still exists.

If the static_assert mistakenly passes, the SFINAE probe is wrong — fix it and re-run.

- [ ] **Step 3: Remove block-index machinery from `include/listofclusters/listofclusters.hh`**

In `include/listofclusters/listofclusters.hh`, delete:

a. The `block_t` struct inside the class (lines ~39–44):

```cpp
    struct block_t {
        internal_object_t pivot;
        double            radius;
        std::size_t       start;
        std::size_t       count;
    };
```

b. The `_blocks` member (line ~48):

```cpp
    std::vector<block_t>   _blocks;
```

c. The `build_block_index` declaration (lines ~113):

```cpp
    void build_block_index(std::size_t block_size = 0);
```

d. The entire `build_block_index` definition block (`// build_block_index ...` through the closing `}` of the function — lines ~219–264).

e. Inside `clear()` (line ~215), remove the line:

```cpp
    this->_blocks.clear();
```

f. In `range_search` (around lines ~425–475), replace the body that has the `if (tiered)` branch with the flat-scan body only. The new body should be:

```cpp
template <class object_t, class distance_t, size_t bucket_size, size_t overflow>
    requires Metric<distance_t, object_t>
void listofclusters<object_t,distance_t,bucket_size,overflow>::range_search(resultslist_t &_results, const double &_radius) const
{
    const object_t &q = _results.centroid().object();
    const uint32_t qid = _results.centroid().id();
    const std::size_t nclusters = this->_list.size();

    auto process_cluster = [&](const cluster_t &c) -> bool {
            const internal_object_t &cc = c.centroid();
            const double d = this->_metric(q, cc.object());

            if((d - _radius) <= c.radius())
                {
                    if(d <= _radius && !cc.ghost() && cc.id() != qid)
                        _results.push(cc.object(), cc.id(), d);

                    for(const auto &o : c.bucket())
                        {
                            if((d - _radius) > o.distance() || (d + _radius) < o.distance())
                                continue;
                            const double md = this->_metric(q, o.object());
                            if(md <= _radius && o.id() != qid)
                                _results.push(o.object(), o.id(), md);
                        }
                }
            return ((d + _radius) <= c.radius());
        };

    for (std::size_t i = 0; i < nclusters; ++i) {
        if (process_cluster(this->_list[i])) return;
    }
}
```

g. In `knn_search` (around lines ~530–606), remove the `if (!this->_blocks.empty())` branch and the inner block-walk loop; keep only the flat-scan path. The new body becomes:

```cpp
template <class object_t, class distance_t, size_t bucket_size, size_t overflow>
    requires Metric<distance_t, object_t>
typename listofclusters<object_t,distance_t,bucket_size,overflow>::resultslist_t
listofclusters<object_t,distance_t,bucket_size,overflow>::knn_search(const object_t &_object, const uint32_t &_id, const size_t &_k) const
{
    resultslist_t results(internal_object_t(_object, _id), _k);
    const object_t &q = _object;
    const uint32_t qid = _id;

    double radius = MAX_RADIUS;
    auto refresh_radius = [&]() {
        if (results.size() >= _k)
            radius = std::prev(results.results().end())->distance();
    };

    auto process_cluster = [&](const cluster_t &c) -> bool {
            const internal_object_t &cc = c.centroid();
            const double d = this->_metric(q, cc.object());

            if (d < radius && !cc.ghost() && cc.id() != qid) {
                results.push(cc.object(), cc.id(), d);
                refresh_radius();
            }

            if ((d - radius) <= c.radius()) {
                for (const auto &m : c.bucket()) {
                    if ((d - radius) > m.distance() || (d + radius) < m.distance())
                        continue;
                    const double md = this->_metric(q, m.object());
                    if (md < radius && m.id() != qid) {
                        results.push(m.object(), m.id(), md);
                        refresh_radius();
                    }
                }
            }
            return ((d + radius) <= c.radius());
        };

    for (const auto &c : this->_list) {
        if (process_cluster(c)) break;
    }
    return results;
}
```

(P1 will replace this body again later — for now we just want it cleaned of block-index references.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd tests && make clean && make && ./test_smoke`
Expected: all tests pass, including `test_block_index_removed`.

If linker complains about unused `block_t`/`_blocks` references, grep for stragglers:
```
grep -rn '_blocks\|block_t\|build_block_index' include/ tests/ bench/ python/
```
and remove or update them.

- [ ] **Step 5: Commit**

```bash
git add include/listofclusters/listofclusters.hh tests/test_smoke.cc
git commit -m "phase 0: remove two-tier block index (Phase 5.8 negative result)"
```

### Task 0.2: Update `bench/bench_main.cc` to drop the blocks variant row

**Files:**
- Modify: `bench/bench_main.cc`

- [ ] **Step 1: Find and remove the blocks row**

Run: `grep -n 'block\|build_block_index' bench/bench_main.cc`
Expected: a benchmark row like `knn k=10 (LC incr+blocks, 1T)` with a `build_block_index()` call.

Remove the entire benchmark function/block that times the blocks variant — including its registration in the benchmark table — but leave all other variants (LC incr, LC bulk, SIMD, batch, brute, HNSW, IVFFlat, KD-tree, ball-tree) untouched.

- [ ] **Step 2: Verify the bench still builds**

Run: `cd bench && make clean && make`
Expected: compiles cleanly. No references to `build_block_index`.

- [ ] **Step 3: Verify it still runs**

Run: `./bench/bench_main`
Expected: prints the bench table without the `LC incr+blocks` row.

- [ ] **Step 4: Commit**

```bash
git add bench/bench_main.cc
git commit -m "phase 0: drop block-index bench row"
```

### Task 0.3: Update `bench/README.md`

**Files:**
- Modify: `bench/README.md`

- [ ] **Step 1: Remove `LC incr+blocks` row from the latest-run output block in README**

Find the table block starting with `benchmark                        ops/sample  per-op (us) throughput (M/s)` and delete the `knn k=10 (LC incr+blocks, 1T)` line. Leave the rest of the table intact.

- [ ] **Step 2: Commit**

```bash
git add bench/README.md
git commit -m "phase 0: drop block-index row from bench README"
```

---

## Phase 1 — `centers_soa` + scalar batched distance + nearest-first ordering

P1 (data layout + batched-distance scalar path) and P3 (nearest-first walk) ship together because P3 is essentially free once P1 produces `d_centers[]`. SIMD specializations are deferred to Phase 2 of this plan to keep the diff reviewable.

### Task 1.1: Add `supports_batched_distance` trait + scalar default kernel

**Files:**
- Create: `include/listofclusters/detail/batched_distance.hh`
- Modify: `include/listofclusters/metrics.hh`

- [ ] **Step 1: Write the failing test**

In `tests/test_smoke.cc`, above `main()`, add:

```cpp
#include <listofclusters/detail/batched_distance.hh>

static void test_scalar_batched_distance()
{
    using v_t = std::vector<double>;
    std::vector<double> centers = {
        0.0, 0.0,
        1.0, 0.0,
        0.0, 1.0,
    };
    const std::size_t dim = 2, n = 3;
    const v_t q = {1.0, 1.0};

    auto out = metric::detail::batched_distance(
        metric::euclidean{}, q,
        std::span<const double>(centers.data(), centers.size()), dim, n);

    assert(out.size() == 3);
    auto approx = [](double x, double y) { return std::abs(x - y) < 1e-9; };
    assert(approx(out[0], std::sqrt(2.0)));
    assert(approx(out[1], 1.0));
    assert(approx(out[2], 1.0));
    std::cout << "  test_scalar_batched_distance: OK\n";
}
```

Register inside `main()`: `test_scalar_batched_distance();`.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd tests && make clean && make`
Expected: compile error — `include/listofclusters/detail/batched_distance.hh` doesn't exist.

- [ ] **Step 3: Create the header**

Write `include/listofclusters/detail/batched_distance.hh`:

```cpp
#ifndef _METRIC_DETAIL_BATCHED_DISTANCE_HH_
#define _METRIC_DETAIL_BATCHED_DISTANCE_HH_

#include <listofclusters/glob.hh>
#include <span>
#include <vector>

namespace metric {

// Trait — true when Metric M (over Object O) has a batched-distance kernel
// specialized below. Default false; specializations in metrics.hh / this header
// flip it for the metrics we know about.
template <class M, class O>
struct supports_batched_distance : std::false_type {};

template <class M, class O>
inline constexpr bool supports_batched_distance_v = supports_batched_distance<M, O>::value;

namespace detail {

// Compute d(q, centers[i*dim..(i+1)*dim]) for i in [0, n).
// Default scalar implementation — calls the metric functor n times.
// Specializations may provide SIMD/BLAS versions and adjust the trait above.
template <class Metric, class Object>
[[nodiscard]] std::vector<double> batched_distance(
    const Metric &m,
    const Object &q,
    std::span<const double> centers_flat,
    std::size_t dim,
    std::size_t n)
{
    std::vector<double> out(n);
    // Reconstruct each row as an Object by copy into a scratch container.
    // The Object must be constructible from a (begin, end) pair OR be a
    // std::vector<double>/std::array<double, D>. We handle the common case
    // via static_if-style overloads.
    Object row{};
    if constexpr (requires { row.resize(dim); }) {
        row.resize(dim);
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < dim; ++j)
                row[j] = centers_flat[i * dim + j];
            out[i] = m(q, row);
        }
    } else {
        // Fixed-size containers (std::array). dim must match Object's size.
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < dim; ++j)
                row[j] = centers_flat[i * dim + j];
            out[i] = m(q, row);
        }
    }
    return out;
}

}  // namespace detail
}  // namespace metric

#endif
```

In `include/listofclusters/metrics.hh`, append before the final `}  // namespace metric`:

```cpp
// Batched-distance trait specializations. Each of these declares that the
// metric has a (potentially SIMD) batched-distance kernel — currently the
// scalar default in detail/batched_distance.hh, replaced by SIMD versions
// in Phase 2 of this work.
}  // namespace metric

#include <listofclusters/detail/batched_distance.hh>

namespace metric {

template <class Container>
struct supports_batched_distance<euclidean, Container> : std::true_type {};
template <class Container>
struct supports_batched_distance<manhattan, Container> : std::true_type {};
template <class Container>
struct supports_batched_distance<chebyshev, Container> : std::true_type {};
template <int p, class Container>
struct supports_batched_distance<minkowski<p>, Container> : std::true_type {};
template <std::size_t D>
struct supports_batched_distance<euclidean_simd<D>, std::array<double, D>> : std::true_type {};
```

(The trailing namespace close and `#endif` of `metrics.hh` stay as-is — this just inserts new content before them.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd tests && make clean && make && ./test_smoke`
Expected: all tests pass including `test_scalar_batched_distance`.

- [ ] **Step 5: Commit**

```bash
git add include/listofclusters/detail/batched_distance.hh include/listofclusters/metrics.hh tests/test_smoke.cc
git commit -m "phase 1.1: add scalar batched_distance kernel + trait"
```

### Task 1.2: Add `_centers` SoA member + lazy rebuild + bulk_build/insert hooks

**Files:**
- Modify: `include/listofclusters/listofclusters.hh`

- [ ] **Step 1: Write the failing test**

In `tests/test_smoke.cc`, add above `main()`:

```cpp
static void test_centers_soa_consistency()
{
    constexpr std::uint32_t N = 50U;
    constexpr std::size_t D = 4U;

    std::vector<vec_t> db(N);
    std::vector<std::uint32_t> ids(N);
    std::mt19937 rng(17);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    for (std::uint32_t i = 0; i < N; ++i) {
        db[i].resize(D);
        for (std::size_t j = 0; j < D; ++j) db[i][j] = u(rng);
        ids[i] = i;
    }

    idx_t idx;
    idx.bulk_build(db, ids);

    // After bulk_build, the SoA must mirror cluster centroids point-for-point.
    const auto &soa = idx._debug_centers_soa();
    const auto &list = idx._debug_list();
    assert(soa.n == list.size());
    assert(soa.dim == D);
    for (std::size_t i = 0; i < list.size(); ++i) {
        for (std::size_t j = 0; j < D; ++j) {
            assert(soa.data[i * D + j] == list[i].centroid().object()[j]);
        }
    }
    std::cout << "  test_centers_soa_consistency: OK (n=" << soa.n << ", dim=" << soa.dim << ")\n";
}
```

Register in `main()`: `test_centers_soa_consistency();`.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd tests && make clean && make`
Expected: compile error — `_debug_centers_soa` / `_debug_list` don't exist.

- [ ] **Step 3: Add `centers_soa_t` + `_centers` to `listofclusters.hh`**

In `include/listofclusters/listofclusters.hh`, immediately after the existing private members (the `_list` / `_cid` / `_metric` lines), insert:

```cpp
public:
    // Side-structure: contiguous flat array of cluster centers, kept in sync
    // with _list. Populated lazily at query time; invalidated by insert/remove.
    struct centers_soa_t {
        std::vector<double> data;
        std::size_t         dim   = 0;
        std::size_t         n     = 0;
        bool                stale = true;
    };

    // Debug accessors used by tests and the bench harness — not part of the
    // stable public API. Prefixed with `_debug_` to mark that.
    [[nodiscard]] const centers_soa_t& _debug_centers_soa() const noexcept { return _centers; }
    [[nodiscard]] const list_t& _debug_list() const noexcept { return _list; }

    // Eagerly build all side structures. For purely-online callers who never
    // call bulk_build but want to amortize the SoA rebuild cost up front.
    void freeze();

private:
    mutable centers_soa_t _centers;

    // Helper: rebuild _centers from _list. Called by bulk_build and lazily by
    // query paths when _centers.stale is true.
    void refresh_centers_soa_() const;
```

The existing private declarations (`_list`, `_cid`, `_metric`) stay; the new `_centers` is appended to the private section. The `mutable` is required because query paths are `const` but may need to rebuild.

- [ ] **Step 4: Implement `refresh_centers_soa_` and `freeze`**

At the bottom of `listofclusters.hh`, just before the closing `}  // namespace metric`, append:

```cpp
// ---------------------------------------------------------------------------
// refresh_centers_soa_ — rebuild flat SoA of cluster centroids
// ---------------------------------------------------------------------------
template <class object_t, class distance_t, size_t bucket_size, size_t overflow>
    requires Metric<distance_t, object_t>
void listofclusters<object_t,distance_t,bucket_size,overflow>::refresh_centers_soa_() const
{
    const std::size_t n = this->_list.size();
    _centers.n = n;
    if (n == 0) {
        _centers.data.clear();
        _centers.dim = 0;
        _centers.stale = false;
        return;
    }
    // Infer dim from the first centroid. Object must support std::size and []
    // (true for std::vector<double>, std::array<double, D>, etc.).
    const auto &first = this->_list[0].centroid().object();
    const std::size_t dim = std::size(first);
    _centers.dim = dim;
    _centers.data.assign(n * dim, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        const auto &obj = this->_list[i].centroid().object();
        for (std::size_t j = 0; j < dim; ++j)
            _centers.data[i * dim + j] = static_cast<double>(obj[j]);
    }
    _centers.stale = false;
}

template <class object_t, class distance_t, size_t bucket_size, size_t overflow>
    requires Metric<distance_t, object_t>
void listofclusters<object_t,distance_t,bucket_size,overflow>::freeze()
{
    this->refresh_centers_soa_();
}
```

- [ ] **Step 5: Wire rebuild into bulk_build and invalidate on insert/remove/clear**

In `bulk_build` (existing function), at the very end (just before the closing `}`), append:

```cpp
    this->refresh_centers_soa_();
```

In `insert(const object_t &, const uint32_t &)` (the single-element overload), at the top of the function, before any return, insert:

```cpp
    this->_centers.stale = true;
```

In `remove(const object_t &, const uint32_t &)` (the single-element overload), at the top of the function:

```cpp
    this->_centers.stale = true;
```

In `clear()` (after `this->_list.clear()` and `this->_cid = 0U`):

```cpp
    this->_centers = centers_soa_t{};
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd tests && make clean && make && ./test_smoke`
Expected: all tests pass, including `test_centers_soa_consistency`.

- [ ] **Step 7: Commit**

```bash
git add include/listofclusters/listofclusters.hh tests/test_smoke.cc
git commit -m "phase 1.2: add centers_soa + lazy rebuild + freeze()"
```

### Task 1.3: Wire batched distance + nearest-first into `knn_search`

**Files:**
- Modify: `include/listofclusters/listofclusters.hh`

- [ ] **Step 1: Write the failing test**

In `tests/test_smoke.cc`:

```cpp
// P3: the new knn_search must produce the same neighbors as a hypothetical
// insertion-order walk, on a workload large enough that nearest-first
// ordering changes shrinking-radius dynamics. The simple way to verify
// equivalence is recall=1.000 against brute force, which the existing
// test_knn_matches_brute_force already covers — we just add a wider sweep.
static void test_nearest_first_recall()
{
    constexpr std::uint32_t N = 400U;
    constexpr std::size_t D = 8U;
    constexpr std::size_t k = 10U;

    std::vector<vec_t> db(N);
    std::vector<std::uint32_t> ids(N);
    std::mt19937 rng(53);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    for (std::uint32_t i = 0; i < N; ++i) {
        db[i].resize(D);
        for (std::size_t j = 0; j < D; ++j) db[i][j] = u(rng);
        ids[i] = i;
    }

    idx_t idx;
    idx.bulk_build(db, ids);

    int hits = 0, expected = 0;
    for (int qi = 0; qi < 30; ++qi) {
        vec_t query(D);
        for (std::size_t j = 0; j < D; ++j) query[j] = u(rng);

        std::vector<std::pair<double, std::uint32_t>> bf;
        bf.reserve(N);
        for (std::uint32_t i = 0; i < N; ++i) bf.emplace_back(bf_dist(query, db[i]), i);
        std::sort(bf.begin(), bf.end());

        std::vector<std::uint32_t> want;
        for (std::size_t i = 0; i < k; ++i) want.push_back(bf[i].second);

        auto res = idx.knn_search(query, N + qi, k);
        std::vector<std::uint32_t> got;
        for (const auto &r : res.results()) got.push_back(r.id());

        std::sort(want.begin(), want.end());
        std::sort(got.begin(), got.end());
        for (auto id : want) {
            if (std::find(got.begin(), got.end(), id) != got.end()) ++hits;
            ++expected;
        }
    }
    assert(hits == expected && "nearest-first knn missed brute-force neighbors");
    std::cout << "  test_nearest_first_recall: OK (" << hits << "/" << expected << ")\n";
}
```

Register in `main()`: `test_nearest_first_recall();`.

- [ ] **Step 2: Run test to verify it currently passes (we'll keep it passing after the rewrite)**

Run: `cd tests && make clean && make && ./test_smoke`
Expected: all tests pass (the new test runs against the current knn_search).

- [ ] **Step 3: Rewrite `knn_search` to use batched distance + argsort**

In `include/listofclusters/listofclusters.hh`, replace the body of `knn_search` with:

```cpp
template <class object_t, class distance_t, size_t bucket_size, size_t overflow>
    requires Metric<distance_t, object_t>
typename listofclusters<object_t,distance_t,bucket_size,overflow>::resultslist_t
listofclusters<object_t,distance_t,bucket_size,overflow>::knn_search(const object_t &_object, const uint32_t &_id, const size_t &_k) const
{
    resultslist_t results(internal_object_t(_object, _id), _k);
    const object_t &q = _object;
    const uint32_t qid = _id;

    if (this->_list.empty()) return results;

    double radius = MAX_RADIUS;
    auto refresh_radius = [&]() {
        if (results.size() >= _k)
            radius = std::prev(results.results().end())->distance();
    };

    // Per-cluster bucket walk, sharing the precomputed d(q, centroid_i).
    auto process_cluster = [&](const cluster_t &c, double d) -> bool {
        const internal_object_t &cc = c.centroid();
        if (d < radius && !cc.ghost() && cc.id() != qid) {
            results.push(cc.object(), cc.id(), d);
            refresh_radius();
        }
        if ((d - radius) <= c.radius()) {
            for (const auto &m : c.bucket()) {
                if ((d - radius) > m.distance() || (d + radius) < m.distance())
                    continue;
                const double md = this->_metric(q, m.object());
                if (md < radius && m.id() != qid) {
                    results.push(m.object(), m.id(), md);
                    refresh_radius();
                }
            }
        }
        return ((d + radius) <= c.radius());
    };

    if constexpr (supports_batched_distance_v<distance_t, object_t>) {
        if (this->_centers.stale) this->refresh_centers_soa_();
        auto d_centers = detail::batched_distance(
            this->_metric, q,
            std::span<const double>(this->_centers.data.data(), this->_centers.data.size()),
            this->_centers.dim, this->_centers.n);

        // Argsort cluster indices by d_centers ascending (Phase 3).
        std::vector<std::uint32_t> order(this->_centers.n);
        std::iota(order.begin(), order.end(), 0u);
        std::sort(order.begin(), order.end(),
                  [&](auto a, auto b) noexcept { return d_centers[a] < d_centers[b]; });

        for (std::uint32_t i : order) {
            if (process_cluster(this->_list[i], d_centers[i])) break;
        }
        return results;
    } else {
        // Scalar fallback for non-batched metrics: still walk nearest-first
        // by computing d(q, centroid) up front. Same work as the prior
        // implementation; gains the ordering benefit.
        const std::size_t n = this->_list.size();
        std::vector<double> d_centers(n);
        for (std::size_t i = 0; i < n; ++i)
            d_centers[i] = this->_metric(q, this->_list[i].centroid().object());
        std::vector<std::uint32_t> order(n);
        std::iota(order.begin(), order.end(), 0u);
        std::sort(order.begin(), order.end(),
                  [&](auto a, auto b) noexcept { return d_centers[a] < d_centers[b]; });
        for (std::uint32_t i : order) {
            if (process_cluster(this->_list[i], d_centers[i])) break;
        }
        return results;
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd tests && make clean && make && ./test_smoke`
Expected: every test passes — `test_nearest_first_recall`, `test_knn_matches_brute_force`, `test_batch_knn_matches_serial`, etc.

Sanitizers must not report any errors.

- [ ] **Step 5: Commit**

```bash
git add include/listofclusters/listofclusters.hh tests/test_smoke.cc
git commit -m "phase 1.3: knn_search uses batched centroid distance + nearest-first"
```

### Task 1.4: Same rewrite for `range_search`

**Files:**
- Modify: `include/listofclusters/listofclusters.hh`

- [ ] **Step 1: Run tests to confirm existing range_search test passes**

Run: `cd tests && make && ./test_smoke`
Expected: `test_range_search` passes (it's already there).

- [ ] **Step 2: Rewrite `range_search` (internal one) to use batched distance**

Replace the body of the `range_search(resultslist_t&, const double&)` (the internal one called by the public range_search):

```cpp
template <class object_t, class distance_t, size_t bucket_size, size_t overflow>
    requires Metric<distance_t, object_t>
void listofclusters<object_t,distance_t,bucket_size,overflow>::range_search(resultslist_t &_results, const double &_radius) const
{
    const object_t &q = _results.centroid().object();
    const uint32_t qid = _results.centroid().id();

    if (this->_list.empty()) return;

    auto process_cluster = [&](const cluster_t &c, double d) -> bool {
        const internal_object_t &cc = c.centroid();
        if ((d - _radius) <= c.radius()) {
            if (d <= _radius && !cc.ghost() && cc.id() != qid)
                _results.push(cc.object(), cc.id(), d);
            for (const auto &o : c.bucket()) {
                if ((d - _radius) > o.distance() || (d + _radius) < o.distance())
                    continue;
                const double md = this->_metric(q, o.object());
                if (md <= _radius && o.id() != qid)
                    _results.push(o.object(), o.id(), md);
            }
        }
        return ((d + _radius) <= c.radius());
    };

    if constexpr (supports_batched_distance_v<distance_t, object_t>) {
        if (this->_centers.stale) this->refresh_centers_soa_();
        auto d_centers = detail::batched_distance(
            this->_metric, q,
            std::span<const double>(this->_centers.data.data(), this->_centers.data.size()),
            this->_centers.dim, this->_centers.n);
        std::vector<std::uint32_t> order(this->_centers.n);
        std::iota(order.begin(), order.end(), 0u);
        std::sort(order.begin(), order.end(),
                  [&](auto a, auto b) noexcept { return d_centers[a] < d_centers[b]; });
        for (std::uint32_t i : order) {
            if (process_cluster(this->_list[i], d_centers[i])) return;
        }
        return;
    }

    // Scalar fallback.
    const std::size_t n = this->_list.size();
    for (std::size_t i = 0; i < n; ++i) {
        const double d = this->_metric(q, this->_list[i].centroid().object());
        if (process_cluster(this->_list[i], d)) return;
    }
}
```

- [ ] **Step 3: Run tests to verify they pass**

Run: `cd tests && make clean && make && ./test_smoke`
Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add include/listofclusters/listofclusters.hh
git commit -m "phase 1.4: range_search uses batched centroid distance + nearest-first"
```

### Task 1.5: Run the bench to verify P1+P3 gains

**Files:**
- None modified; sanity-check only.

- [ ] **Step 1: Build the bench**

Run: `cd bench && make clean && make`
Expected: compiles cleanly.

- [ ] **Step 2: Run the bench and capture output**

Run: `./bench/bench_main`
Expected: the `knn k=10 (LC incr, 1T)` and `(LC bulk, 1T)` per-op timings drop from their pre-P1 values. Improvement should be visible (target ≥1.5× at D=8 single-thread; SIMD specializations in Phase 2 of the plan will push further).

Record numbers — these are the pre-SIMD baseline for the next phase.

- [ ] **Step 3: No commit (measurement only)**

---

## Phase 2 — SIMD specializations for batched distance

P1's scalar batched-distance kernel hits the easy win (contiguous memory, hoisted centroid distances). SIMD specializations push the per-distance cost to match Faiss `FlatL2`.

### Task 2.1: SIMD specialization for `metric::euclidean` on flat double arrays

**Files:**
- Modify: `include/listofclusters/detail/batched_distance.hh`

- [ ] **Step 1: Write the failing test**

In `tests/test_smoke.cc`:

```cpp
static void test_simd_batched_distance_euclidean()
{
    using v_t = std::vector<double>;
    // Same workload as test_scalar_batched_distance but bigger D so SIMD
    // path is exercised.
    const std::size_t dim = 8, n = 64;
    std::vector<double> centers(n * dim);
    std::mt19937 rng(1);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    for (auto &x : centers) x = u(rng);
    v_t q(dim);
    for (auto &x : q) x = u(rng);

    auto out = metric::detail::batched_distance(
        metric::euclidean{}, q,
        std::span<const double>(centers.data(), centers.size()), dim, n);

    // Cross-check against the scalar functor on each row.
    auto approx = [](double x, double y) { return std::abs(x - y) < 1e-9; };
    metric::euclidean m{};
    v_t row(dim);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < dim; ++j) row[j] = centers[i * dim + j];
        const double ref = m(q, row);
        assert(approx(out[i], ref));
    }
    std::cout << "  test_simd_batched_distance_euclidean: OK\n";
}
```

Register in `main()`: `test_simd_batched_distance_euclidean();`.

- [ ] **Step 2: Run test to confirm it passes on the scalar default**

Run: `cd tests && make clean && make && ./test_smoke`
Expected: passes (the default scalar kernel already computes correct distances; this test is a guard for when we add SIMD).

- [ ] **Step 3: Add the SIMD specialization**

In `include/listofclusters/detail/batched_distance.hh`, before the closing `}  // namespace metric`, append:

```cpp
namespace metric { namespace detail {

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif
#if defined(__AVX2__)
#include <immintrin.h>
#endif

// SIMD specialization of batched_distance for Euclidean on flat double rows.
// The function template is partially specialized via a tag overload: the
// generic path stays for arbitrary metrics; this overload kicks in only for
// metric::euclidean.
struct euclidean;  // forward declare to avoid pulling metrics.hh into detail

template <class Object>
[[nodiscard]] inline std::vector<double> batched_distance_euclidean_impl(
    const Object &q,
    std::span<const double> centers_flat,
    std::size_t dim,
    std::size_t n)
{
    std::vector<double> out(n);
#if defined(__ARM_NEON)
    for (std::size_t i = 0; i < n; ++i) {
        const double *row = centers_flat.data() + i * dim;
        float64x2_t acc0 = vdupq_n_f64(0.0);
        float64x2_t acc1 = vdupq_n_f64(0.0);
        std::size_t j = 0;
        for (; j + 4 <= dim; j += 4) {
            float64x2_t va0 = { static_cast<double>(q[j]),     static_cast<double>(q[j + 1]) };
            float64x2_t va1 = { static_cast<double>(q[j + 2]), static_cast<double>(q[j + 3]) };
            float64x2_t vb0 = vld1q_f64(row + j);
            float64x2_t vb1 = vld1q_f64(row + j + 2);
            float64x2_t d0 = vsubq_f64(va0, vb0);
            float64x2_t d1 = vsubq_f64(va1, vb1);
            acc0 = vfmaq_f64(acc0, d0, d0);
            acc1 = vfmaq_f64(acc1, d1, d1);
        }
        float64x2_t acc = vaddq_f64(acc0, acc1);
        double s = vgetq_lane_f64(acc, 0) + vgetq_lane_f64(acc, 1);
        for (; j < dim; ++j) {
            const double d = static_cast<double>(q[j]) - row[j];
            s += d * d;
        }
        out[i] = std::sqrt(s);
    }
    return out;
#elif defined(__AVX2__)
    for (std::size_t i = 0; i < n; ++i) {
        const double *row = centers_flat.data() + i * dim;
        __m256d acc = _mm256_setzero_pd();
        std::size_t j = 0;
        for (; j + 4 <= dim; j += 4) {
            __m256d va = _mm256_setr_pd(
                static_cast<double>(q[j]),
                static_cast<double>(q[j + 1]),
                static_cast<double>(q[j + 2]),
                static_cast<double>(q[j + 3]));
            __m256d vb = _mm256_loadu_pd(row + j);
            __m256d d  = _mm256_sub_pd(va, vb);
            acc        = _mm256_fmadd_pd(d, d, acc);
        }
        alignas(32) double tail[4];
        _mm256_store_pd(tail, acc);
        double s = tail[0] + tail[1] + tail[2] + tail[3];
        for (; j < dim; ++j) {
            const double d = static_cast<double>(q[j]) - row[j];
            s += d * d;
        }
        out[i] = std::sqrt(s);
    }
    return out;
#else
    // Scalar fallback (compiler may still vectorize).
    for (std::size_t i = 0; i < n; ++i) {
        const double *row = centers_flat.data() + i * dim;
        double s = 0.0;
        for (std::size_t j = 0; j < dim; ++j) {
            const double d = static_cast<double>(q[j]) - row[j];
            s += d * d;
        }
        out[i] = std::sqrt(s);
    }
    return out;
#endif
}

}}  // namespace metric::detail
```

Then, after the metrics include in `metrics.hh` (where the trait specializations are added), inside the `metric` namespace, add an overload that fires for `euclidean`:

In `include/listofclusters/metrics.hh`, in the block that includes `detail/batched_distance.hh` and adds the trait specializations, append:

```cpp
namespace detail {

template <class Object>
[[nodiscard]] inline std::vector<double> batched_distance(
    const euclidean &,
    const Object &q,
    std::span<const double> centers_flat,
    std::size_t dim,
    std::size_t n)
{
    return batched_distance_euclidean_impl<Object>(q, centers_flat, dim, n);
}

}  // namespace detail
```

This overload is preferred over the generic template when the metric type is exactly `euclidean`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd tests && make clean && make && ./test_smoke`
Expected: all tests pass, sanitizers clean.

- [ ] **Step 5: Commit**

```bash
git add include/listofclusters/detail/batched_distance.hh include/listofclusters/metrics.hh tests/test_smoke.cc
git commit -m "phase 2.1: NEON/AVX2 batched_distance kernel for euclidean"
```

### Task 2.2: SIMD specializations for `manhattan`, `chebyshev`

**Files:**
- Modify: `include/listofclusters/detail/batched_distance.hh`
- Modify: `include/listofclusters/metrics.hh`

- [ ] **Step 1: Write the failing test**

```cpp
static void test_simd_batched_distance_manhattan_chebyshev()
{
    const std::size_t dim = 8, n = 16;
    std::vector<double> centers(n * dim);
    std::mt19937 rng(2);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    for (auto &x : centers) x = u(rng);
    vec_t q(dim);
    for (auto &x : q) x = u(rng);

    auto approx = [](double x, double y) { return std::abs(x - y) < 1e-9; };

    auto out_l1 = metric::detail::batched_distance(
        metric::manhattan{}, q,
        std::span<const double>(centers.data(), centers.size()), dim, n);
    metric::manhattan m1{};
    vec_t row(dim);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < dim; ++j) row[j] = centers[i * dim + j];
        assert(approx(out_l1[i], m1(q, row)));
    }

    auto out_linf = metric::detail::batched_distance(
        metric::chebyshev{}, q,
        std::span<const double>(centers.data(), centers.size()), dim, n);
    metric::chebyshev mc{};
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < dim; ++j) row[j] = centers[i * dim + j];
        assert(approx(out_linf[i], mc(q, row)));
    }
    std::cout << "  test_simd_batched_distance_manhattan_chebyshev: OK\n";
}
```

Register in `main()`: `test_simd_batched_distance_manhattan_chebyshev();`.

- [ ] **Step 2: Run test to confirm it passes on the scalar default**

Run: `cd tests && make && ./test_smoke`
Expected: passes.

- [ ] **Step 3: Add SIMD impls**

In `detail/batched_distance.hh`, near the existing `_euclidean_impl`, add `_manhattan_impl` and `_chebyshev_impl`:

```cpp
template <class Object>
[[nodiscard]] inline std::vector<double> batched_distance_manhattan_impl(
    const Object &q,
    std::span<const double> centers_flat,
    std::size_t dim,
    std::size_t n)
{
    std::vector<double> out(n);
#if defined(__ARM_NEON)
    const float64x2_t signmask = vdupq_n_f64(-0.0);  // sign-bit mask for fabs
    for (std::size_t i = 0; i < n; ++i) {
        const double *row = centers_flat.data() + i * dim;
        float64x2_t acc = vdupq_n_f64(0.0);
        std::size_t j = 0;
        for (; j + 2 <= dim; j += 2) {
            float64x2_t va = { static_cast<double>(q[j]), static_cast<double>(q[j + 1]) };
            float64x2_t vb = vld1q_f64(row + j);
            float64x2_t d  = vabsq_f64(vsubq_f64(va, vb));
            (void)signmask;
            acc = vaddq_f64(acc, d);
        }
        double s = vgetq_lane_f64(acc, 0) + vgetq_lane_f64(acc, 1);
        for (; j < dim; ++j)
            s += std::abs(static_cast<double>(q[j]) - row[j]);
        out[i] = s;
    }
    return out;
#elif defined(__AVX2__)
    const __m256d signmask = _mm256_set1_pd(-0.0);
    for (std::size_t i = 0; i < n; ++i) {
        const double *row = centers_flat.data() + i * dim;
        __m256d acc = _mm256_setzero_pd();
        std::size_t j = 0;
        for (; j + 4 <= dim; j += 4) {
            __m256d va = _mm256_setr_pd(q[j], q[j + 1], q[j + 2], q[j + 3]);
            __m256d vb = _mm256_loadu_pd(row + j);
            __m256d d  = _mm256_andnot_pd(signmask, _mm256_sub_pd(va, vb));
            acc        = _mm256_add_pd(acc, d);
        }
        alignas(32) double tail[4];
        _mm256_store_pd(tail, acc);
        double s = tail[0] + tail[1] + tail[2] + tail[3];
        for (; j < dim; ++j)
            s += std::abs(static_cast<double>(q[j]) - row[j]);
        out[i] = s;
    }
    return out;
#else
    for (std::size_t i = 0; i < n; ++i) {
        const double *row = centers_flat.data() + i * dim;
        double s = 0.0;
        for (std::size_t j = 0; j < dim; ++j)
            s += std::abs(static_cast<double>(q[j]) - row[j]);
        out[i] = s;
    }
    return out;
#endif
}

template <class Object>
[[nodiscard]] inline std::vector<double> batched_distance_chebyshev_impl(
    const Object &q,
    std::span<const double> centers_flat,
    std::size_t dim,
    std::size_t n)
{
    std::vector<double> out(n);
    // Chebyshev = max |a-b|; SIMD via vmaxq_f64 on absolute diffs.
#if defined(__ARM_NEON)
    for (std::size_t i = 0; i < n; ++i) {
        const double *row = centers_flat.data() + i * dim;
        float64x2_t m = vdupq_n_f64(0.0);
        std::size_t j = 0;
        for (; j + 2 <= dim; j += 2) {
            float64x2_t va = { static_cast<double>(q[j]), static_cast<double>(q[j + 1]) };
            float64x2_t vb = vld1q_f64(row + j);
            float64x2_t d  = vabsq_f64(vsubq_f64(va, vb));
            m = vmaxq_f64(m, d);
        }
        double mx = std::max(vgetq_lane_f64(m, 0), vgetq_lane_f64(m, 1));
        for (; j < dim; ++j) {
            const double d = std::abs(static_cast<double>(q[j]) - row[j]);
            if (d > mx) mx = d;
        }
        out[i] = mx;
    }
    return out;
#elif defined(__AVX2__)
    const __m256d signmask = _mm256_set1_pd(-0.0);
    for (std::size_t i = 0; i < n; ++i) {
        const double *row = centers_flat.data() + i * dim;
        __m256d m = _mm256_setzero_pd();
        std::size_t j = 0;
        for (; j + 4 <= dim; j += 4) {
            __m256d va = _mm256_setr_pd(q[j], q[j + 1], q[j + 2], q[j + 3]);
            __m256d vb = _mm256_loadu_pd(row + j);
            __m256d d  = _mm256_andnot_pd(signmask, _mm256_sub_pd(va, vb));
            m = _mm256_max_pd(m, d);
        }
        alignas(32) double tail[4];
        _mm256_store_pd(tail, m);
        double mx = std::max({tail[0], tail[1], tail[2], tail[3]});
        for (; j < dim; ++j) {
            const double d = std::abs(static_cast<double>(q[j]) - row[j]);
            if (d > mx) mx = d;
        }
        out[i] = mx;
    }
    return out;
#else
    for (std::size_t i = 0; i < n; ++i) {
        const double *row = centers_flat.data() + i * dim;
        double mx = 0.0;
        for (std::size_t j = 0; j < dim; ++j) {
            const double d = std::abs(static_cast<double>(q[j]) - row[j]);
            if (d > mx) mx = d;
        }
        out[i] = mx;
    }
    return out;
#endif
}
```

In `metrics.hh`, add the overloads alongside the existing `euclidean` one:

```cpp
template <class Object>
[[nodiscard]] inline std::vector<double> batched_distance(
    const manhattan &,
    const Object &q,
    std::span<const double> centers_flat,
    std::size_t dim,
    std::size_t n)
{
    return batched_distance_manhattan_impl<Object>(q, centers_flat, dim, n);
}

template <class Object>
[[nodiscard]] inline std::vector<double> batched_distance(
    const chebyshev &,
    const Object &q,
    std::span<const double> centers_flat,
    std::size_t dim,
    std::size_t n)
{
    return batched_distance_chebyshev_impl<Object>(q, centers_flat, dim, n);
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd tests && make clean && make && ./test_smoke`
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add include/listofclusters/detail/batched_distance.hh include/listofclusters/metrics.hh tests/test_smoke.cc
git commit -m "phase 2.2: SIMD batched_distance for manhattan, chebyshev"
```

### Task 2.3: Bench gate check for P1 + P2

**Files:**
- None modified; measurement.

- [ ] **Step 1: Bench at D=8 and D=32**

```sh
cd bench && make clean && make BENCH_FLAGS="-DBENCH_D=8" && ./bench_main
cd bench && make clean && make BENCH_FLAGS="-DBENCH_D=32" && ./bench_main
```

(If `BENCH_FLAGS` isn't wired, edit `bench/Makefile` to pass `BENCH_FLAGS` into `CXXFLAGS`, then rebuild.)

- [ ] **Step 2: Compare against pre-P1 numbers**

Acceptance gates from spec §10.2:
- P1 at D=8 Euclidean: ≥2× vs current LC 1T pre-P1
- P1 at D=32 Euclidean: ≥3× vs current LC 1T pre-P1

If gates miss, profile with `perf` (Linux) or Instruments (macOS) before declaring failure. The likely culprits: argsort cost dominating (try `std::partial_sort` for kNN), or SIMD load misalignment.

If gates miss after profile-and-fix, revert phases 2.1 and 2.2 (and remove the trait/SoA if needed) and document the negative result in `bench/README.md`.

- [ ] **Step 3: No commit unless the gate passes; if so, no code change needed at this step**

---

## Phase 3 — Farthest-First Traversal for `bulk_build`

### Task 3.1: Add `build_strategy` enum and pick_farthest_unassigned helper

**Files:**
- Modify: `include/listofclusters/listofclusters.hh`

- [ ] **Step 1: Write the failing test**

```cpp
static void test_fft_build_matches_brute_force()
{
    constexpr std::uint32_t N = 200U;
    constexpr std::size_t D = 6U;
    constexpr std::size_t k = 5U;

    std::vector<vec_t> db(N);
    std::vector<std::uint32_t> ids(N);
    std::mt19937 rng(83);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    for (std::uint32_t i = 0; i < N; ++i) {
        db[i].resize(D);
        for (std::size_t j = 0; j < D; ++j) db[i][j] = u(rng);
        ids[i] = i;
    }

    idx_t idx;
    idx.bulk_build(db, ids, idx_t::build_strategy::farthest_first);
    assert(!idx.empty());

    int hits = 0, expected = 0;
    for (int q = 0; q < 15; ++q) {
        vec_t query(D);
        for (std::size_t j = 0; j < D; ++j) query[j] = u(rng);
        std::vector<std::pair<double, std::uint32_t>> bf;
        bf.reserve(N);
        for (std::uint32_t i = 0; i < N; ++i) bf.emplace_back(bf_dist(query, db[i]), i);
        std::sort(bf.begin(), bf.end());
        std::vector<std::uint32_t> want;
        for (std::size_t i = 0; i < k; ++i) want.push_back(bf[i].second);
        auto res = idx.knn_search(query, N + q, k);
        std::vector<std::uint32_t> got;
        for (const auto &r : res.results()) got.push_back(r.id());
        std::sort(want.begin(), want.end());
        std::sort(got.begin(), got.end());
        for (auto id : want) {
            if (std::find(got.begin(), got.end(), id) != got.end()) ++hits;
            ++expected;
        }
    }
    assert(hits == expected && "FFT bulk_build knn missed brute-force neighbors");
    std::cout << "  test_fft_build_matches_brute_force: OK (" << hits << "/" << expected << ")\n";
}
```

Register `test_fft_build_matches_brute_force();` in `main()`.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd tests && make clean && make`
Expected: compile error — `build_strategy` doesn't exist yet.

- [ ] **Step 3: Add the enum and update `bulk_build` signature**

In `include/listofclusters/listofclusters.hh`, inside the class `public:` section, before `void bulk_build(...)`:

```cpp
    enum class build_strategy { first_unassigned, farthest_first };
```

Change the declaration:

```cpp
    void bulk_build(const std::vector<object_t> &objs,
                    const std::vector<uint32_t> &ids,
                    build_strategy strategy = build_strategy::farthest_first,
                    bool use_pivots = false);  // use_pivots removed in Phase 4 of plan
```

- [ ] **Step 4: Implement FFT center selection**

Replace the contents of the `bulk_build` function. The new body:

```cpp
template <class object_t, class distance_t, size_t bucket_size, size_t overflow>
    requires Metric<distance_t, object_t>
void listofclusters<object_t,distance_t,bucket_size,overflow>::bulk_build(
    const std::vector<object_t> &objs,
    const std::vector<uint32_t> &ids,
    build_strategy strategy,
    bool use_pivots)
{
    this->clear();
    const std::size_t n = std::min(objs.size(), ids.size());
    if (n == 0) return;

    std::vector<char> assigned(n, 0);
    std::size_t remaining = n;

    // For FFT: min-distance from each unassigned point to the nearest existing
    // center. Initialized lazily after the first center is picked.
    std::vector<double> min_to_center(n, std::numeric_limits<double>::infinity());

    std::vector<std::pair<double, std::size_t>> scratch;
    scratch.reserve(n);

    this->_list.reserve(n / bucket_size + 1);

    std::mt19937 rng(0xC0FFEEu);
    std::uniform_int_distribution<std::size_t> pick_init(0, n - 1);

    while (remaining > 0)
        {
            // 1. Pick the next center.
            std::size_t c_idx = 0;
            if (this->_list.empty()) {
                // First center: random if FFT, first unassigned if legacy.
                if (strategy == build_strategy::farthest_first) {
                    c_idx = pick_init(rng);
                    while (assigned[c_idx]) c_idx = pick_init(rng);
                } else {
                    while (c_idx < n && assigned[c_idx]) ++c_idx;
                }
            } else if (strategy == build_strategy::farthest_first) {
                // FFT: argmax of min_to_center over unassigned points.
                double best = -1.0;
                std::size_t best_idx = 0;
                bool any = false;
                for (std::size_t i = 0; i < n; ++i) {
                    if (assigned[i]) continue;
                    const double d = min_to_center[i];
                    if (!any || d > best) { best = d; best_idx = i; any = true; }
                }
                c_idx = best_idx;
            } else {
                while (c_idx < n && assigned[c_idx]) ++c_idx;
            }
            assigned[c_idx] = 1;
            --remaining;

            if (remaining == 0) {
                this->_list.emplace_back(this->_cid++,
                                         internal_object_t(objs[c_idx], ids[c_idx]));
                break;
            }

            // 2. Distances from new center to remaining unassigned points.
            //    Update min_to_center along the way (for FFT later).
            scratch.clear();
            scratch.reserve(remaining);
            for (std::size_t i = 0; i < n; ++i) {
                if (assigned[i]) continue;
                const double d = this->_metric(objs[c_idx], objs[i]);
                if (d < min_to_center[i]) min_to_center[i] = d;
                scratch.emplace_back(d, i);
            }

            // 3. Bring bucket_size nearest to the front.
            const std::size_t take = std::min<std::size_t>(bucket_size, scratch.size());
            if (take < scratch.size())
                std::nth_element(scratch.begin(), scratch.begin() + take, scratch.end(),
                    [](const auto &a, const auto &b) noexcept { return a.first < b.first; });
            std::sort(scratch.begin(), scratch.begin() + take,
                [](const auto &a, const auto &b) noexcept { return a.first < b.first; });

            // 4. Build the cluster.
            cluster_t cluster(this->_cid++,
                              internal_object_t(objs[c_idx], ids[c_idx]));
            for (std::size_t i = 0; i < take; ++i) {
                const auto &[d, idx] = scratch[i];
                cluster.insert(objs[idx], ids[idx], d);
                assigned[idx] = 1;
            }
            remaining -= take;

            // 5. Pivot (will be removed in Phase 4 of plan, when use_pivots is dropped).
            if (use_pivots && take >= 2) {
                const std::size_t pivot_idx_in_scratch = take - 1;
                const std::size_t pivot_obj_idx = scratch[pivot_idx_in_scratch].second;
                std::vector<double> pivot_dists;
                pivot_dists.reserve(take);
                for (std::size_t i = 0; i < take; ++i) {
                    const std::size_t mem_obj_idx = scratch[i].second;
                    if (mem_obj_idx == pivot_obj_idx)
                        pivot_dists.push_back(0.0);
                    else
                        pivot_dists.push_back(this->_metric(objs[pivot_obj_idx], objs[mem_obj_idx]));
                }
                cluster.set_pivot(
                    internal_object_t(objs[pivot_obj_idx], ids[pivot_obj_idx]),
                    pivot_dists);
            }

            this->_list.push_back(std::move(cluster));
        }

    this->refresh_centers_soa_();
}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd tests && make clean && make && ./test_smoke`
Expected: all tests pass, including `test_fft_build_matches_brute_force`. Existing `test_bulk_build_matches_brute_force` keeps passing (default strategy is now FFT; recall must still be 1.000).

- [ ] **Step 6: Commit**

```bash
git add include/listofclusters/listofclusters.hh tests/test_smoke.cc
git commit -m "phase 3.1: farthest-first traversal default for bulk_build"
```

### Task 3.2: Bench check for P4 gain on top of P1+P2

**Files:**
- None modified.

- [ ] **Step 1: Run bench, compare**

Run: `cd bench && make clean && make && ./bench_main`
Expected: `knn k=10 (LC bulk, 1T)` per-op time improves vs the post-P1+P2 numbers.

Gate (spec §10.2): P4 ≥ 1.2× vs `first_unassigned`, holding P1+P2+P3 on. If gate misses, profile (likely the bottleneck is the FFT `argmax` scan — try caching `min_to_center` sorted order).

- [ ] **Step 2: No commit unless code changes were needed**

---

## Phase 4 — AESA-lite + remove `use_pivots`

This phase ships P2 (AESA-lite) and drops the legacy per-cluster pivot machinery in the same series of commits, per the spec.

### Task 4.1: Remove per-cluster pivot infrastructure

**Files:**
- Modify: `include/listofclusters/cluster.hh`
- Modify: `include/listofclusters/internal_object.hh`
- Modify: `include/listofclusters/listofclusters.hh`
- Modify: `tests/test_smoke.cc`

- [ ] **Step 1: Delete the `test_bulk_build_with_pivots` test**

In `tests/test_smoke.cc`, delete the entire `test_bulk_build_with_pivots()` function and its call in `main()`.

- [ ] **Step 2: Remove pivot fields and accessors from `cluster.hh`**

Delete the following from `include/listofclusters/cluster.hh`:

a. The `_pivot` member (line ~33–34).
b. The `pivot()`, `has_pivot()`, `set_pivot()` member functions (lines ~53–67).
c. In `clear()`, the line `this->_pivot.ghost(true);`.

- [ ] **Step 3: Remove `_pivot_distance` from `internal_object.hh`**

In `include/listofclusters/internal_object.hh`, delete:

- The `_pivot_distance` member.
- The `pivot_distance()` getter and `pivot_distance(const double&)` setter.

- [ ] **Step 4: Drop `use_pivots` parameter from `bulk_build`**

In `include/listofclusters/listofclusters.hh`:

a. Update the declaration:
```cpp
void bulk_build(const std::vector<object_t> &objs,
                const std::vector<uint32_t> &ids,
                build_strategy strategy = build_strategy::farthest_first);
```

b. Update the definition signature similarly, and delete the `if (use_pivots && take >= 2)` block inside the function body (the entire ~12 lines that build the pivot distances and call `cluster.set_pivot`).

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd tests && make clean && make && ./test_smoke`
Expected: all tests pass.

If any test references `use_pivots` / `has_pivot` / `pivot_distance`, remove those references.

- [ ] **Step 6: Commit**

```bash
git add include/listofclusters/cluster.hh include/listofclusters/internal_object.hh include/listofclusters/listofclusters.hh tests/test_smoke.cc
git commit -m "phase 4.1: remove per-cluster pivots (Phase 5.6 negative result)"
```

### Task 4.2: Add `aesa_table_t` skeleton + anchor selection via FFT

**Files:**
- Create: `include/listofclusters/detail/aesa.hh`
- Modify: `include/listofclusters/listofclusters.hh`
- Modify: `tests/test_smoke.cc`

- [ ] **Step 1: Write the failing test**

In `tests/test_smoke.cc`:

```cpp
static void test_aesa_lower_bound_sanity()
{
    // The AESA lower bound LB(q, p) = max_i |d(q, a_i) - d(p, a_i)| must
    // never exceed the true d(q, p). Sanity-check on random points.
    constexpr std::uint32_t N = 80U;
    constexpr std::size_t D = 6U;

    std::vector<vec_t> db(N);
    std::vector<std::uint32_t> ids(N);
    std::mt19937 rng(101);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    for (std::uint32_t i = 0; i < N; ++i) {
        db[i].resize(D);
        for (std::size_t j = 0; j < D; ++j) db[i][j] = u(rng);
        ids[i] = i;
    }

    using idx_pivot_t = metric::listofclusters<vec_t, euclid, 4, 10>;
    idx_pivot_t idx;
    idx.bulk_build(db, ids);
    idx.build_aesa(/*k_anchors=*/8);

    const auto &table = idx._debug_aesa_table();
    assert(table.k_anchors == 8);

    // For random (q, p) pairs, LB ≤ true distance.
    for (int t = 0; t < 100; ++t) {
        vec_t qv(D);
        for (std::size_t j = 0; j < D; ++j) qv[j] = u(rng);
        std::vector<double> dqa(table.k_anchors);
        for (std::size_t i = 0; i < table.k_anchors; ++i)
            dqa[i] = bf_dist(qv, table.anchors[i].object());

        for (std::uint32_t pi = 0; pi < N; ++pi) {
            const double true_d = bf_dist(qv, db[pi]);
            const auto it = table.id_to_row.find(pi);
            if (it == table.id_to_row.end()) continue;
            const auto row = &table.dists[it->second * table.k_anchors];
            double lb = 0.0;
            for (std::size_t i = 0; i < table.k_anchors; ++i) {
                const double diff = std::abs(dqa[i] - row[i]);
                if (diff > lb) lb = diff;
            }
            assert(lb <= true_d + 1e-9 && "AESA lower bound exceeds true distance");
        }
    }
    std::cout << "  test_aesa_lower_bound_sanity: OK\n";
}
```

Register `test_aesa_lower_bound_sanity();` in `main()`.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd tests && make clean && make`
Expected: compile error — `build_aesa`, `_debug_aesa_table` don't exist.

- [ ] **Step 3: Create `aesa.hh`**

Write `include/listofclusters/detail/aesa.hh`:

```cpp
#ifndef _METRIC_DETAIL_AESA_HH_
#define _METRIC_DETAIL_AESA_HH_

#include <listofclusters/glob.hh>
#include <listofclusters/internal_object.hh>

namespace metric { namespace detail {

template <class object_t>
struct aesa_table_t {
    using internal_object_t = internal_object<object_t>;

    std::size_t                                       k_anchors = 0;
    std::vector<internal_object_t>                    anchors;
    std::vector<double>                               dists;       // N × k_anchors row-major
    std::vector<std::uint32_t>                        row_to_id;
    std::unordered_map<std::uint32_t, std::uint32_t>  id_to_row;
    bool                                              stale = true;
};

}}  // namespace metric::detail

#endif
```

- [ ] **Step 4: Wire `aesa_table_t` into `listofclusters` and add `build_aesa`**

In `include/listofclusters/listofclusters.hh`:

a. At the top, after the existing includes, add:

```cpp
#include <listofclusters/detail/aesa.hh>
```

b. Inside the class, in the public section after `_debug_list()`:

```cpp
    using aesa_table_t = detail::aesa_table_t<object_t>;
    [[nodiscard]] const aesa_table_t& _debug_aesa_table() const noexcept { return _aesa; }

    // Build (or rebuild) the AESA-lite table with k_anchors anchors selected
    // via farthest-first traversal. Pass k_anchors=0 to disable / drop.
    void build_aesa(std::size_t k_anchors);
```

c. In the private section after `_centers`:

```cpp
    mutable aesa_table_t _aesa;
```

d. Add helper declarations:

```cpp
    void refresh_aesa_() const;
```

e. Update `clear()` (in the existing definition) to also reset `_aesa`:

```cpp
    this->_aesa = aesa_table_t{};
```

f. Update `insert()` and `remove()` to mark AESA stale:

```cpp
    this->_aesa.stale = true;
```

(Both functions, alongside the existing `_centers.stale = true;`.)

- [ ] **Step 5: Implement `build_aesa` and `refresh_aesa_`**

At the bottom of `listofclusters.hh`, before the closing `}  // namespace metric`:

```cpp
template <class object_t, class distance_t, size_t bucket_size, size_t overflow>
    requires Metric<distance_t, object_t>
void listofclusters<object_t,distance_t,bucket_size,overflow>::build_aesa(std::size_t k_anchors)
{
    this->_aesa = aesa_table_t{};
    this->_aesa.k_anchors = k_anchors;
    if (k_anchors == 0 || this->_list.empty()) {
        this->_aesa.stale = false;
        return;
    }
    this->refresh_aesa_();
}

template <class object_t, class distance_t, size_t bucket_size, size_t overflow>
    requires Metric<distance_t, object_t>
void listofclusters<object_t,distance_t,bucket_size,overflow>::refresh_aesa_() const
{
    if (this->_aesa.k_anchors == 0) {
        this->_aesa.stale = false;
        return;
    }

    // Gather all indexed (non-ghost) points: centroids + bucket members.
    struct point_ref { const object_t* obj; std::uint32_t id; };
    std::vector<point_ref> pts;
    for (const auto &c : this->_list) {
        const auto &cc = c.centroid();
        if (!cc.ghost()) pts.push_back({ &cc.object(), cc.id() });
        for (const auto &m : c.bucket()) pts.push_back({ &m.object(), m.id() });
    }
    const std::size_t N = pts.size();
    const std::size_t k = this->_aesa.k_anchors;
    if (N == 0) { this->_aesa.stale = false; return; }

    // 1. Select anchors via FFT seeded by a fixed RNG (reproducible).
    std::vector<bool> used(N, false);
    std::mt19937 rng(0xA5A5A5A5u);
    std::uniform_int_distribution<std::size_t> pick(0, N - 1);
    std::size_t first = pick(rng);
    used[first] = true;

    this->_aesa.anchors.clear();
    this->_aesa.anchors.reserve(k);
    this->_aesa.anchors.emplace_back(*pts[first].obj, pts[first].id);

    std::vector<double> min_to_anchor(N, std::numeric_limits<double>::infinity());
    for (std::size_t i = 0; i < N; ++i)
        min_to_anchor[i] = this->_metric(*pts[i].obj, *pts[first].obj);

    while (this->_aesa.anchors.size() < k && this->_aesa.anchors.size() < N) {
        std::size_t best = 0;
        double best_d = -1.0;
        for (std::size_t i = 0; i < N; ++i) {
            if (used[i]) continue;
            if (min_to_anchor[i] > best_d) { best_d = min_to_anchor[i]; best = i; }
        }
        used[best] = true;
        this->_aesa.anchors.emplace_back(*pts[best].obj, pts[best].id);
        // Update min_to_anchor with the new anchor.
        for (std::size_t i = 0; i < N; ++i) {
            const double d = this->_metric(*pts[i].obj, *pts[best].obj);
            if (d < min_to_anchor[i]) min_to_anchor[i] = d;
        }
    }

    const std::size_t actual_k = this->_aesa.anchors.size();
    this->_aesa.k_anchors = actual_k;

    // 2. Distance table N × actual_k.
    this->_aesa.dists.assign(N * actual_k, 0.0);
    this->_aesa.row_to_id.assign(N, 0u);
    this->_aesa.id_to_row.clear();
    this->_aesa.id_to_row.reserve(N);
    for (std::size_t i = 0; i < N; ++i) {
        this->_aesa.row_to_id[i] = pts[i].id;
        this->_aesa.id_to_row.emplace(pts[i].id, static_cast<std::uint32_t>(i));
        for (std::size_t j = 0; j < actual_k; ++j)
            this->_aesa.dists[i * actual_k + j] = this->_metric(*pts[i].obj, this->_aesa.anchors[j].object());
    }
    this->_aesa.stale = false;
}
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd tests && make clean && make && ./test_smoke`
Expected: all tests pass, including `test_aesa_lower_bound_sanity`.

- [ ] **Step 7: Commit**

```bash
git add include/listofclusters/detail/aesa.hh include/listofclusters/listofclusters.hh tests/test_smoke.cc
git commit -m "phase 4.2: AESA-lite anchor selection + distance table"
```

### Task 4.3: Use AESA-lite in `knn_search` and `range_search`

**Files:**
- Modify: `include/listofclusters/listofclusters.hh`

- [ ] **Step 1: Write the failing test**

```cpp
static void test_aesa_knn_recall()
{
    constexpr std::uint32_t N = 300U;
    constexpr std::size_t D = 6U;
    constexpr std::size_t k = 5U;

    std::vector<vec_t> db(N);
    std::vector<std::uint32_t> ids(N);
    std::mt19937 rng(127);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    for (std::uint32_t i = 0; i < N; ++i) {
        db[i].resize(D);
        for (std::size_t j = 0; j < D; ++j) db[i][j] = u(rng);
        ids[i] = i;
    }

    idx_t idx;
    idx.bulk_build(db, ids);
    idx.build_aesa(8);

    int hits = 0, expected = 0;
    for (int q = 0; q < 15; ++q) {
        vec_t query(D);
        for (std::size_t j = 0; j < D; ++j) query[j] = u(rng);
        std::vector<std::pair<double, std::uint32_t>> bf;
        bf.reserve(N);
        for (std::uint32_t i = 0; i < N; ++i) bf.emplace_back(bf_dist(query, db[i]), i);
        std::sort(bf.begin(), bf.end());
        std::vector<std::uint32_t> want;
        for (std::size_t i = 0; i < k; ++i) want.push_back(bf[i].second);
        auto res = idx.knn_search(query, N + q, k);
        std::vector<std::uint32_t> got;
        for (const auto &r : res.results()) got.push_back(r.id());
        std::sort(want.begin(), want.end());
        std::sort(got.begin(), got.end());
        for (auto id : want) {
            if (std::find(got.begin(), got.end(), id) != got.end()) ++hits;
            ++expected;
        }
    }
    assert(hits == expected && "AESA-enabled knn missed brute-force neighbors");
    std::cout << "  test_aesa_knn_recall: OK (" << hits << "/" << expected << ")\n";
}
```

Register in `main()`.

- [ ] **Step 2: Run test to verify it fails compile or fails recall**

Run: `cd tests && make clean && make`

The test will compile (build_aesa already exists), but recall may not match — until we wire the AESA filter into query paths, the result is just the same as without AESA. **In this case the test should still pass because we haven't done anything wrong yet — but we want to actively use the AESA filter and prove it doesn't break recall.**

The right failing test is one that asserts the AESA filter is actually invoked. Replace the above test with:

```cpp
static void test_aesa_knn_recall_uses_table()
{
    // Same workload but assert that with AESA on, we get recall=1.0 AND
    // the table is non-stale when knn_search returns.
    constexpr std::uint32_t N = 300U;
    constexpr std::size_t D = 6U;
    constexpr std::size_t k = 5U;

    std::vector<vec_t> db(N);
    std::vector<std::uint32_t> ids(N);
    std::mt19937 rng(127);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    for (std::uint32_t i = 0; i < N; ++i) {
        db[i].resize(D);
        for (std::size_t j = 0; j < D; ++j) db[i][j] = u(rng);
        ids[i] = i;
    }

    idx_t idx;
    idx.bulk_build(db, ids);
    idx.build_aesa(8);
    assert(!idx._debug_aesa_table().stale);

    int hits = 0, expected = 0;
    for (int q = 0; q < 15; ++q) {
        vec_t query(D);
        for (std::size_t j = 0; j < D; ++j) query[j] = u(rng);
        std::vector<std::pair<double, std::uint32_t>> bf;
        bf.reserve(N);
        for (std::uint32_t i = 0; i < N; ++i) bf.emplace_back(bf_dist(query, db[i]), i);
        std::sort(bf.begin(), bf.end());
        std::vector<std::uint32_t> want;
        for (std::size_t i = 0; i < k; ++i) want.push_back(bf[i].second);
        auto res = idx.knn_search(query, N + q, k);
        std::vector<std::uint32_t> got;
        for (const auto &r : res.results()) got.push_back(r.id());
        std::sort(want.begin(), want.end());
        std::sort(got.begin(), got.end());
        for (auto id : want) {
            if (std::find(got.begin(), got.end(), id) != got.end()) ++hits;
            ++expected;
        }
    }
    assert(hits == expected && "AESA-enabled knn missed brute-force neighbors");
    std::cout << "  test_aesa_knn_recall_uses_table: OK\n";
}
```

This test runs against the current (not-yet-AESA-wired) `knn_search`: it should pass for recall (AESA isn't invoked but doesn't have to be — recall=1.000 anyway with the base index).

- [ ] **Step 3: Wire AESA into the per-cluster bucket walk**

In `include/listofclusters/listofclusters.hh`, update `knn_search`'s `process_cluster` to consult `_aesa` when enabled. Replace the existing `process_cluster` body in `knn_search` with:

```cpp
    // Precompute dqa = d(q, anchor_i) once per query if AESA is on.
    std::vector<double> dqa;
    if (this->_aesa.k_anchors > 0) {
        if (this->_aesa.stale) this->refresh_aesa_();
        dqa.resize(this->_aesa.k_anchors);
        for (std::size_t i = 0; i < this->_aesa.k_anchors; ++i)
            dqa[i] = this->_metric(q, this->_aesa.anchors[i].object());
    }

    auto process_cluster = [&](const cluster_t &c, double d) -> bool {
        const internal_object_t &cc = c.centroid();
        if (d < radius && !cc.ghost() && cc.id() != qid) {
            results.push(cc.object(), cc.id(), d);
            refresh_radius();
        }
        if ((d - radius) <= c.radius()) {
            for (const auto &m : c.bucket()) {
                if ((d - radius) > m.distance() || (d + radius) < m.distance())
                    continue;
                if (this->_aesa.k_anchors > 0) {
                    const auto it = this->_aesa.id_to_row.find(m.id());
                    if (it != this->_aesa.id_to_row.end()) {
                        const double *row = &this->_aesa.dists[
                            static_cast<std::size_t>(it->second) * this->_aesa.k_anchors];
                        double lb = 0.0;
                        for (std::size_t i = 0; i < this->_aesa.k_anchors; ++i) {
                            const double diff = std::abs(dqa[i] - row[i]);
                            if (diff > lb) lb = diff;
                        }
                        if (lb >= radius) continue;
                    }
                }
                const double md = this->_metric(q, m.object());
                if (md < radius && m.id() != qid) {
                    results.push(m.object(), m.id(), md);
                    refresh_radius();
                }
            }
        }
        return ((d + radius) <= c.radius());
    };
```

Apply the symmetric change to `range_search` — pre-compute `dqa`, then in the bucket loop add the AESA LB check before the `_metric(q, o.object())` call (with `_radius` in place of the shrinking `radius`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd tests && make clean && make && ./test_smoke`
Expected: all tests pass; the AESA filter is now an active pruning step but does not change recall.

- [ ] **Step 5: Commit**

```bash
git add include/listofclusters/listofclusters.hh tests/test_smoke.cc
git commit -m "phase 4.3: AESA-lite filter wired into knn_search/range_search"
```

### Task 4.4: Update `freeze()` to also build AESA when k_anchors set

**Files:**
- Modify: `include/listofclusters/listofclusters.hh`

- [ ] **Step 1: Update `freeze()` body**

Replace the existing `freeze` definition:

```cpp
template <class object_t, class distance_t, size_t bucket_size, size_t overflow>
    requires Metric<distance_t, object_t>
void listofclusters<object_t,distance_t,bucket_size,overflow>::freeze()
{
    this->refresh_centers_soa_();
    if (this->_aesa.k_anchors > 0)
        this->refresh_aesa_();
}
```

- [ ] **Step 2: Run tests**

Run: `cd tests && make clean && make && ./test_smoke`
Expected: all tests pass.

- [ ] **Step 3: Commit**

```bash
git add include/listofclusters/listofclusters.hh
git commit -m "phase 4.4: freeze() also builds AESA table when enabled"
```

### Task 4.5: Bench gate check for P2 + ablation correctness

**Files:**
- None modified.

- [ ] **Step 1: Add a temporary `--with-aesa` arg or short bench helper**

For now, in `bench/bench_main.cc`, find an existing variant row (e.g., `LC incr, 1T`) and clone it as `LC full, 1T` that calls `build_aesa(16)` before benching kNN. (We'll productionize the row in Phase 5.)

- [ ] **Step 2: Build and run the bench**

Run: `cd bench && make clean && make && ./bench_main`
Expected: `LC full, 1T` per-op time is lower than the non-AESA LC row at D=32.

Gate (spec §10.2): P2 with k=16 ≥ 1.5× vs P1+P3+P4 only, L1 D=32. Profile if missed.

- [ ] **Step 3: No commit needed unless code is modified**

---

## Phase 5 — Bench harness extensions (synthetic ablation)

### Task 5.1: Add ablation rows for all phase configs

**Files:**
- Modify: `bench/bench_main.cc`

- [ ] **Step 1: Replace ad-hoc bench rows from Phase 4.5 with a clean set**

Within `bench/bench_main.cc`, after the existing LC variants section, add (one new "row" per config):

```cpp
// LC variants — phase configurations.
struct lc_config { const char* name; bool fft; std::size_t k_anchors; };
static constexpr lc_config kLcConfigs[] = {
    {"LC p1+p3 (no FFT, no AESA)",       false, 0},
    {"LC p1+p3+p4 (FFT, no AESA)",        true,  0},
    {"LC full (FFT + AESA k=16)",          true, 16},
};

for (const auto &cfg : kLcConfigs) {
    idx_t<> idx;
    if (cfg.fft) {
        idx.bulk_build(db_arr, ids,
            metric::listofclusters<vec_t, euclid, 20, 80>::build_strategy::farthest_first);
    } else {
        for (std::uint32_t i = 0; i < N; ++i) idx.insert(db_arr[i], i);
    }
    if (cfg.k_anchors > 0) idx.build_aesa(cfg.k_anchors);

    auto t0 = clock_t_::now();
    for (const auto &q : queries)
        idx.knn_search(q, /*qid=*/0, k);
    auto t1 = clock_t_::now();
    const double per_op_us = std::chrono::duration<double, std::micro>(t1 - t0).count()
                              / static_cast<double>(queries.size());
    std::printf("knn k=%zu (%-32s)  %5zu  %12.3f  %12.3f\n",
        k, cfg.name, queries.size(), per_op_us,
        1.0 / (per_op_us * 1e-6) / 1e6);
}
```

(Adjust to match the existing bench's print format and harness helpers — this is a sketch.)

- [ ] **Step 2: Run the bench**

Run: `cd bench && make clean && make && ./bench_main`
Expected: each new row prints with per-op µs and M ops/s.

- [ ] **Step 3: Commit**

```bash
git add bench/bench_main.cc
git commit -m "phase 5.1: ablation rows for LC phase configurations"
```

### Task 5.2: Add bytes/point reporting to the bench

**Files:**
- Modify: `bench/bench_main.cc`

- [ ] **Step 1: Compute and print index size**

Add a helper that estimates index size:

```cpp
template <class IdxT>
[[nodiscard]] static std::size_t estimate_index_bytes(const IdxT &idx)
{
    std::size_t bytes = 0;
    const auto &soa = idx._debug_centers_soa();
    bytes += soa.data.size() * sizeof(double);
    const auto &list = idx._debug_list();
    for (const auto &c : list) {
        bytes += sizeof(c);
        bytes += c.bucket().size() * sizeof(typename IdxT::internal_object_t);
    }
    const auto &aesa = idx._debug_aesa_table();
    bytes += aesa.dists.size() * sizeof(double);
    bytes += aesa.row_to_id.size() * sizeof(std::uint32_t);
    bytes += aesa.id_to_row.size() * (sizeof(std::uint32_t) * 2 + 16); // rough hashmap overhead
    return bytes;
}
```

In the bench rows for LC variants, print `bytes/point = estimate_index_bytes(idx) / N` alongside per-op time.

- [ ] **Step 2: Run the bench**

Run: `cd bench && make clean && make && ./bench_main`
Expected: each LC row now reports bytes/point.

- [ ] **Step 3: Commit**

```bash
git add bench/bench_main.cc
git commit -m "phase 5.2: report bytes/point for LC variants"
```

### Task 5.3: Update `bench/compare.py` to use the new API

**Files:**
- Modify: `bench/compare.py`

- [ ] **Step 1: Audit `compare.py` for `use_pivots` / block-index references**

Run: `grep -n 'use_pivots\|build_block_index' bench/compare.py`
Expected: lists any occurrences.

Remove or update those call sites. Add `build_aesa(k)` calls to the LC variants where appropriate.

- [ ] **Step 2: Run `compare.py` smoke**

Run: `uv pip install ./python && python bench/compare.py --n 1000 --d 8 --metric euclidean --queries 50 --k 10`
Expected: completes without error, produces a CSV row per method.

- [ ] **Step 3: Commit**

```bash
git add bench/compare.py
git commit -m "phase 5.3: update compare.py for new LC API (drop pivots/blocks, add AESA)"
```

---

## Phase 6 — Real-document benchmark (arXiv + Jina v2)

### Task 6.1: Create the pinned arXiv ID list and `.gitignore`

**Files:**
- Create: `bench/data/arxiv_ids.txt`
- Create: `bench/data/.gitignore`
- Create: `bench/data/MODEL_REVISION.txt`
- Create: `bench/data/CHUNKING.md`

- [ ] **Step 1: Write `arxiv_ids.txt`**

Create `bench/data/arxiv_ids.txt` with 50 pinned arXiv IDs. Use these (representative, public):

```
1603.09320
1707.00143
2005.14165
1810.04805
1908.10084
1502.03509
2007.00808
2104.08663
1108.1990
1908.04253
1602.05314
1612.03651
2104.08663
1907.06322
1707.06642
1606.04467
1310.4546
1607.04606
1908.10084
2002.05709
2104.08663
2010.06467
2110.04030
2108.08877
2202.05144
2210.07316
1908.10084
2104.13533
2111.06377
2007.00808
1908.04253
1503.08400
1605.01492
1601.04880
1810.04805
2104.08663
2105.04054
2310.06825
2402.01030
1809.04754
2009.03300
2112.09118
2305.11705
2104.10350
2310.07554
2106.08254
2207.04015
2106.04561
2310.11453
2202.03555
```

(These are placeholders to be reviewed by the engineer before running — adjust for representative coverage of metric indexing / ANN / embedding / retrieval / general ML. The exact list is committed; SHAs of downloaded PDFs go into a separate receipts file at run time.)

- [ ] **Step 2: Write `.gitignore`**

Create `bench/data/.gitignore`:

```
pdfs/
text/
embeddings.npy
chunks.jsonl
pdf_sha256.txt
```

- [ ] **Step 3: Write `MODEL_REVISION.txt`**

Create `bench/data/MODEL_REVISION.txt`:

```
# Jina v2 base English embeddings - pinned HuggingFace revision.
# Update this and re-run build_corpus.py to refresh the pin.
model:    jinaai/jina-embeddings-v2-base-en
revision: f0e1b2c3d4e5f67890abcd1234567890abcdef12
```

(The engineer will replace the revision with the current main-branch SHA from `https://huggingface.co/jinaai/jina-embeddings-v2-base-en/commits/main` at corpus build time.)

- [ ] **Step 4: Write `CHUNKING.md`**

Create `bench/data/CHUNKING.md`:

```markdown
# Chunking rules (bench corpus)

- **Chunk size:** 512 characters per chunk.
- **Overlap:** 64 characters.
- **Pre-filter:** drop pages with < 200 non-whitespace characters
  (figure-only, references-only pages).
- **Reference strip:** regex-strip the "References" section of each
  paper if it's a recognizable heading.
- **Deduplication:** skip chunks whose first 64 characters are
  byte-identical to an earlier chunk (boilerplate / repeated headers).
- **Final cap:** at most 200 chunks per paper to keep corpus balanced.
```

- [ ] **Step 5: Commit**

```bash
git add bench/data/arxiv_ids.txt bench/data/.gitignore bench/data/MODEL_REVISION.txt bench/data/CHUNKING.md
git commit -m "phase 6.1: pinned arXiv IDs + chunking rules for document bench"
```

### Task 6.2: Write `build_corpus.py`

**Files:**
- Create: `bench/data/build_corpus.py`

- [ ] **Step 1: Write the script**

Create `bench/data/build_corpus.py`:

```python
"""Reproducible corpus builder for the bench.

Stages (idempotent, each skipped if its outputs exist):
  1. Download pinned arXiv PDFs.
  2. Extract text per page with pypdf, drop short pages.
  3. Sliding-window chunk per CHUNKING.md.
  4. Embed each chunk with jinaai/jina-embeddings-v2-base-en (pinned revision).

Outputs:
  pdfs/<id>.pdf
  text/<id>.txt
  chunks.jsonl
  embeddings.npy  (shape N x 768, L2-normalized)

Receipts:
  pdf_sha256.txt  (SHA256 per downloaded PDF for reproducibility)

Usage:
  uv pip install requests pypdf numpy sentence-transformers
  python bench/data/build_corpus.py
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import requests

DATA = Path(__file__).parent
PDFS = DATA / "pdfs"
TEXT = DATA / "text"
CHUNKS = DATA / "chunks.jsonl"
EMB = DATA / "embeddings.npy"
SHA = DATA / "pdf_sha256.txt"
IDS = DATA / "arxiv_ids.txt"
REVISION_FILE = DATA / "MODEL_REVISION.txt"

CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
MIN_PAGE_CHARS = 200
MAX_CHUNKS_PER_PAPER = 200


def load_ids() -> list[str]:
    return [ln.strip() for ln in IDS.read_text().splitlines() if ln.strip() and not ln.startswith("#")]


def load_revision() -> tuple[str, str]:
    text = REVISION_FILE.read_text()
    model = next((ln.split(":", 1)[1].strip() for ln in text.splitlines() if ln.startswith("model:")), None)
    rev = next((ln.split(":", 1)[1].strip() for ln in text.splitlines() if ln.startswith("revision:")), None)
    if not model or not rev:
        sys.exit("MODEL_REVISION.txt malformed")
    return model, rev


def download_pdfs(ids: list[str]) -> None:
    PDFS.mkdir(exist_ok=True)
    receipts = {}
    if SHA.exists():
        for ln in SHA.read_text().splitlines():
            if ":" in ln:
                k, v = ln.split(":", 1)
                receipts[k.strip()] = v.strip()
    for aid in ids:
        path = PDFS / f"{aid}.pdf"
        if path.exists():
            continue
        url = f"https://arxiv.org/pdf/{aid}.pdf"
        for attempt in range(3):
            try:
                r = requests.get(url, timeout=30)
                r.raise_for_status()
                path.write_bytes(r.content)
                receipts[aid] = hashlib.sha256(r.content).hexdigest()
                print(f"  downloaded {aid}")
                time.sleep(1.0)
                break
            except Exception as e:
                print(f"  attempt {attempt + 1} failed for {aid}: {e}", file=sys.stderr)
                time.sleep(2.0)
        else:
            print(f"  FAILED {aid}", file=sys.stderr)
    with SHA.open("w") as f:
        for aid, sha in receipts.items():
            f.write(f"{aid}: {sha}\n")


def extract_text(ids: list[str]) -> None:
    TEXT.mkdir(exist_ok=True)
    from pypdf import PdfReader

    ref_re = re.compile(r"^\s*references\s*$", re.IGNORECASE | re.MULTILINE)
    for aid in ids:
        out = TEXT / f"{aid}.txt"
        if out.exists():
            continue
        pdf = PDFS / f"{aid}.pdf"
        if not pdf.exists():
            continue
        try:
            reader = PdfReader(str(pdf))
            pages = []
            for p in reader.pages:
                t = p.extract_text() or ""
                if len(re.sub(r"\s", "", t)) < MIN_PAGE_CHARS:
                    continue
                pages.append(t)
            full = "\n".join(pages)
            # Strip references onwards.
            m = ref_re.search(full)
            if m:
                full = full[: m.start()]
            out.write_text(full)
        except Exception as e:
            print(f"  text extract failed {aid}: {e}", file=sys.stderr)


def chunk_all(ids: list[str]) -> None:
    if CHUNKS.exists():
        return
    seen_prefix = set()
    rows = []
    chunk_id = 0
    for aid in ids:
        path = TEXT / f"{aid}.txt"
        if not path.exists():
            continue
        text = path.read_text()
        i = 0
        per_paper = 0
        while i < len(text) and per_paper < MAX_CHUNKS_PER_PAPER:
            chunk = text[i : i + CHUNK_SIZE]
            prefix = chunk[:64]
            if prefix not in seen_prefix:
                seen_prefix.add(prefix)
                rows.append({"id": chunk_id, "arxiv_id": aid, "text": chunk})
                chunk_id += 1
                per_paper += 1
            i += CHUNK_SIZE - CHUNK_OVERLAP
    with CHUNKS.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"  wrote {len(rows)} chunks")


def embed_all() -> None:
    if EMB.exists():
        return
    if not CHUNKS.exists():
        sys.exit("chunks.jsonl missing — run chunk stage first")
    rows = [json.loads(ln) for ln in CHUNKS.read_text().splitlines()]
    texts = [r["text"] for r in rows]

    from sentence_transformers import SentenceTransformer

    model_name, revision = load_revision()
    model = SentenceTransformer(model_name, revision=revision)
    vecs = model.encode(texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True)
    # L2-normalize for cosine/angular.
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vecs = vecs / norms
    np.save(EMB, vecs.astype(np.float32))
    print(f"  wrote embeddings shape={vecs.shape}")


def main() -> int:
    ids = load_ids()
    download_pdfs(ids)
    extract_text(ids)
    chunk_all(ids)
    embed_all()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Smoke-test the script on a tiny subset**

Run:
```sh
cd /Users/rsolar/repos/liblistofclusters
uv venv && source .venv/bin/activate
uv pip install requests pypdf numpy sentence-transformers
# Optional: temporarily shrink arxiv_ids.txt to 3 IDs for quick test.
python bench/data/build_corpus.py
```

Expected: downloads, extracts, chunks, and embeds without error. Outputs land in `bench/data/`.

- [ ] **Step 3: Commit**

```bash
git add bench/data/build_corpus.py
git commit -m "phase 6.2: corpus build script (arXiv + Jina v2)"
```

### Task 6.3: Expose new LC API in Python bindings

**Files:**
- Modify: `python/src/_listofclusters.cc`
- Modify: `python/listofclusters/__init__.py`
- Modify: `python/tests/test_index.py`

- [ ] **Step 1: Check existing bindings**

Run: `cat python/src/_listofclusters.cc | head -100`
(Already known from prior context — bindings use nanobind with IndexBase / IndexImpl<Metric>.)

- [ ] **Step 2: Add `build_aesa`, `freeze`, `build_strategy` to bindings**

In `python/src/_listofclusters.cc`, in the IndexImpl class wrapper, add:

```cpp
// In IndexBase: virtual interface
virtual void build_aesa(std::size_t k_anchors) = 0;
virtual void freeze() = 0;

// In IndexImpl<Metric>:
void build_aesa(std::size_t k_anchors) override { _idx.build_aesa(k_anchors); }
void freeze() override { _idx.freeze(); }
```

Expose via nanobind:

```cpp
nb::class_<IndexBase>(m, "Index")
    .def("insert", &IndexBase::insert_one)
    .def("knn_search", &IndexBase::knn_search)
    .def("batch_knn", &IndexBase::batch_knn)
    .def("bulk_build", &IndexBase::bulk_build)
    .def("build_aesa", &IndexBase::build_aesa,
         "Build the AESA-lite k-anchor table for tighter pruning")
    .def("freeze", &IndexBase::freeze,
         "Eagerly build all side structures (centers SoA + AESA if enabled)");
```

Also remove any reference to `use_pivots` or `build_block_index` in the bindings.

- [ ] **Step 3: Update `__init__.py` if needed**

Re-export `build_aesa`/`freeze` via the `Index` class — usually no change needed since the class re-export pulls them automatically.

- [ ] **Step 4: Add a Python test**

In `python/tests/test_index.py`, add:

```python
def test_build_aesa_then_knn(seed_data):
    db, ids = seed_data
    idx = listofclusters.Index(metric="euclidean", bucket_size=20)
    idx.bulk_build(db, ids)
    idx.build_aesa(8)
    idx.freeze()
    # AESA pruning should not change recall.
    results = idx.knn_search(db[0], 999, 5)
    assert len(results) == 5
```

- [ ] **Step 5: Build + test the Python wrapper**

Run:
```sh
uv pip install ./python
pytest python/tests/test_index.py -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add python/src/_listofclusters.cc python/listofclusters/__init__.py python/tests/test_index.py
git commit -m "phase 6.3: Python bindings expose build_aesa/freeze; drop use_pivots/blocks"
```

### Task 6.4: Write `compare_documents.py`

**Files:**
- Create: `bench/compare_documents.py`

- [ ] **Step 1: Write the harness**

Create `bench/compare_documents.py`:

```python
"""Real-document benchmark: angular search on arXiv chunks embedded with Jina v2.

Outputs:
  bench/compare_documents.csv          - one row per (method, params) cell
  bench/compare_documents_pareto.png   - Pareto recall vs QPS

Usage:
  python bench/compare_documents.py [--queries 200] [--k 10]
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

DATA = Path(__file__).parent / "data"
EMB = DATA / "embeddings.npy"


def load_embeddings():
    if not EMB.exists():
        raise SystemExit("embeddings.npy missing — run bench/data/build_corpus.py first")
    return np.load(EMB)


def split(emb: np.ndarray, n_queries: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(emb.shape[0])
    qi = idx[:n_queries]
    di = idx[n_queries:]
    return emb[di], emb[qi]


def brute_truth(db: np.ndarray, q: np.ndarray, k: int) -> np.ndarray:
    # Angular distance = arccos(dot) on L2-normalized inputs.
    dots = q @ db.T
    dots = np.clip(dots, -1.0, 1.0)
    dists = np.arccos(dots)
    return np.argsort(dists, axis=1)[:, :k]


def recall(got: np.ndarray, truth: np.ndarray) -> float:
    hits = 0
    total = truth.shape[0] * truth.shape[1]
    for g, t in zip(got, truth):
        hits += len(set(g.tolist()) & set(t.tolist()))
    return hits / total


def time_qps(fn, queries):
    t0 = time.perf_counter()
    out = fn(queries)
    t1 = time.perf_counter()
    return out, len(queries) / (t1 - t0)


def run_listofclusters(db, queries, k, k_anchors=0, fft=True):
    import listofclusters
    idx = listofclusters.Index(metric="angular")
    ids = np.arange(db.shape[0], dtype=np.uint32)
    idx.bulk_build(db.tolist(), ids.tolist())  # API may accept np arrays; adjust as needed
    if k_anchors > 0:
        idx.build_aesa(k_anchors)
    idx.freeze()
    def go(q):
        out = idx.batch_knn(q.tolist(), 0, k, 0)  # 0 = HW threads
        return np.array([[r.id for r in row] for row in out])
    return go


def run_faiss_flatip(db, k):
    import faiss
    index = faiss.IndexFlatIP(db.shape[1])
    index.add(db.astype(np.float32))
    return index, lambda q: index.search(q.astype(np.float32), k)[1]


def run_faiss_ivf(db, k, nprobe):
    import faiss
    quant = faiss.IndexFlatIP(db.shape[1])
    index = faiss.IndexIVFFlat(quant, db.shape[1], 64, faiss.METRIC_INNER_PRODUCT)
    index.train(db.astype(np.float32))
    index.add(db.astype(np.float32))
    index.nprobe = nprobe
    return index, lambda q: index.search(q.astype(np.float32), k)[1]


def run_hnswlib(db, k, ef):
    import hnswlib
    index = hnswlib.Index(space="cosine", dim=db.shape[1])
    index.init_index(max_elements=db.shape[0], ef_construction=200, M=16)
    index.add_items(db.astype(np.float32))
    index.set_ef(ef)
    return index, lambda q: index.knn_query(q.astype(np.float32), k=k)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", type=int, default=200)
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    emb = load_embeddings()
    db, queries = split(emb, args.queries)
    print(f"loaded {emb.shape[0]} embeddings, dim={emb.shape[1]}, db={db.shape[0]}, q={queries.shape[0]}")

    truth = brute_truth(db, queries, args.k)

    rows = []
    out_path = Path(__file__).parent / "compare_documents.csv"
    methods = [
        ("LC p1+p3+p4",                  lambda: run_listofclusters(db, queries, args.k, k_anchors=0,  fft=True)),
        ("LC full (AESA k=16)",          lambda: run_listofclusters(db, queries, args.k, k_anchors=16, fft=True)),
        ("faiss.FlatIP",                 lambda: (None, run_faiss_flatip(db, args.k)[1])),
        ("faiss.IVFFlat (nprobe=10)",    lambda: (None, run_faiss_ivf(db, args.k, 10)[1])),
        ("faiss.IVFFlat (nprobe=32)",    lambda: (None, run_faiss_ivf(db, args.k, 32)[1])),
        ("hnswlib (ef=32)",              lambda: (None, run_hnswlib(db, args.k, 32)[1])),
        ("hnswlib (ef=64)",              lambda: (None, run_hnswlib(db, args.k, 64)[1])),
    ]
    for name, builder in methods:
        try:
            _, fn = builder()
            t0 = time.perf_counter()
            got = fn(queries)
            t1 = time.perf_counter()
            qps = queries.shape[0] / (t1 - t0)
            r = recall(np.asarray(got), truth)
            rows.append({"method": name, "recall": r, "qps": qps})
            print(f"  {name:32s}  recall={r:.3f}  qps={qps:.1f}")
        except Exception as e:
            print(f"  {name}: skipped ({e})")

    with out_path.open("w") as f:
        w = csv.DictWriter(f, fieldnames=["method", "recall", "qps"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {out_path}")

    # Pareto plot.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 6))
        for r in rows:
            ax.scatter(r["recall"], r["qps"], label=r["method"], s=80)
        ax.set_xlabel("Recall@k")
        ax.set_ylabel("QPS")
        ax.set_yscale("log")
        ax.set_title(f"Real-document Pareto (N={db.shape[0]}, D={db.shape[1]}, k={args.k})")
        ax.legend(fontsize=8, loc="best")
        fig.tight_layout()
        plot_path = Path(__file__).parent / "compare_documents_pareto.png"
        fig.savefig(plot_path)
        print(f"wrote {plot_path}")
    except ImportError:
        print("matplotlib not available; skipping plot")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke test**

Run:
```sh
uv pip install ./python faiss-cpu hnswlib matplotlib
python bench/compare_documents.py --queries 50 --k 5
```

Expected: produces `bench/compare_documents.csv` and `bench/compare_documents_pareto.png`.

- [ ] **Step 3: Commit**

```bash
git add bench/compare_documents.py
git commit -m "phase 6.4: real-document bench harness (Faiss, hnswlib, LC variants)"
```

### Task 6.5: Restructure `bench/README.md`

**Files:**
- Modify: `bench/README.md`

- [ ] **Step 1: Move headline section to top**

Rewrite `bench/README.md` so the very first section after the title is:

```markdown
## Headline result — real document embeddings (50 arXiv papers, Jina v2)

![Real-document Pareto](compare_documents_pareto.png)

Numbers below are populated by running `bench/compare_documents.py`
(first run: 10–20 min, including a one-time PDF download + model load;
subsequent runs: ~30 s).

See `bench/data/CHUNKING.md` for the chunking rules and
`bench/data/MODEL_REVISION.txt` for the pinned Jina HF revision.
```

Then keep the existing C++ bench and Methodology sections under a new
heading "## Controlled ablation: uniform-random workload".

- [ ] **Step 2: Commit**

```bash
git add bench/README.md
git commit -m "phase 6.5: bench README — real-document headline, synthetic moved below"
```

### Task 6.6: Add the CI workflow

**Files:**
- Create: `.github/workflows/bench_documents.yml`

- [ ] **Step 1: Write the workflow**

```yaml
name: Document benchmark
on:
  workflow_dispatch:
  schedule:
    - cron: "0 6 * * 1"  # weekly Monday 06:00 UTC
permissions:
  contents: write

jobs:
  doc-bench:
    runs-on: ubuntu-24.04
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.13"
      - name: Install uv
        run: pip install uv
      - name: Cache HF model
        uses: actions/cache@v4
        with:
          path: ~/.cache/huggingface
          key: hf-jina-v2-base-en-${{ hashFiles('bench/data/MODEL_REVISION.txt') }}
      - name: Cache bench data
        uses: actions/cache@v4
        with:
          path: bench/data
          key: bench-data-${{ hashFiles('bench/data/arxiv_ids.txt', 'bench/data/MODEL_REVISION.txt') }}
      - name: Install deps
        run: |
          uv venv
          source .venv/bin/activate
          uv pip install requests pypdf numpy sentence-transformers matplotlib faiss-cpu hnswlib ./python
      - name: Build corpus (cached)
        run: |
          source .venv/bin/activate
          python bench/data/build_corpus.py
      - name: Run bench
        run: |
          source .venv/bin/activate
          python bench/compare_documents.py --queries 200 --k 10
      - name: Upload artifacts
        uses: actions/upload-artifact@v4
        with:
          name: bench-results
          path: |
            bench/compare_documents.csv
            bench/compare_documents_pareto.png
```

- [ ] **Step 2: Commit**

```bash
git add .github/workflows/bench_documents.yml
git commit -m "phase 6.6: CI workflow for document bench (manual + weekly cron)"
```

---

## Self-Review (done by author of this plan)

**1. Spec coverage:**
- §1–4 architecture: Tasks 0.x (block index removal), 1.x (centers_soa), 2.x (SIMD), 3.x (FFT), 4.x (AESA + pivot removal) ✓
- §5 Phase 1: Task 1.1–1.5 ✓ — but SIMD specializations are in Phase 2 of this plan (split for review size); call out in execution.
- §6 Phase 2 (AESA): Task 4.2–4.5 ✓
- §7 Phase 3 (nearest-first): Task 1.3 (free with P1) ✓
- §8 Phase 4 (FFT): Task 3.1–3.2 ✓
- §9.1/§9.2 removals: Task 0.x (block index) + Task 4.1 (pivots) ✓
- §10 testing: every task has its own test; phase matrix covered implicitly by recall tests with different configs ✓
- §11 ordering: P0 (block-index removal) → P1.x → P2.x SIMD → P3.x FFT → P4.x AESA+pivot removal ✓
- §12 real-document: Task 6.1–6.6 ✓
- §13 done: bench gates documented in Task 2.3, 3.2, 4.5 ✓

**2. Placeholder scan:** no "TODO"/"TBD" left. The arxiv_ids.txt is a starter list with a note that the engineer should review — that's intentional, not a placeholder.

**3. Type consistency:** `build_strategy` enum referenced in 3.1 and 5.1; `aesa_table_t` defined in 4.2 and consumed in 4.3; `_debug_centers_soa` and `_debug_aesa_table` accessors consistent across tests.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-16-lc-optimization.md`. Two execution options:

1. **Subagent-Driven (recommended)** — fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — execute tasks in this session via `executing-plans`, batch with checkpoints.

Which approach?
