# Phase 1 — `centers_soa` + scalar batched distance + nearest-first

**Goal.** Restructure the cluster-list scan so the per-query centroid distances are computed contiguously (one pass over a flat SoA), then walked in nearest-first order. P1 + P3 from the spec land together because P3 is free once P1 has `d_centers[]`. SIMD specializations are [Phase 2](phase-2-simd.md).

**Why this first.** Diagnostic: at D=8 Euclidean, Faiss's brute-force `IndexFlatL2` (no pruning) matches our pre-P1 LC's QPS. The bottleneck is per-distance cost, not pruning, because we compute `d(q, c_i)` one cluster at a time through the templated functor. Batching the call against a contiguous matrix lets the compiler (and later, SIMD) vectorize. See spec §1 for the full diagnostic. [JDJ17] is the comparator; [CN05] is the original LC algorithm we're optimizing inside.

**Files modified / created.**
- Create: `include/listofclusters/detail/batched_distance.hh` — trait, scalar default kernel, and per-metric overloads (all in one place to avoid the header layering bug from the monolithic plan).
- Modify: `include/listofclusters/listofclusters.hh` — add `centers_soa_t`, `_centers` member, `refresh_centers_soa_()`, `freeze()`, friend test class hook, batched-distance call sites in `knn_search` and `range_search`.
- Modify: `tests/test_smoke.cc` — add P1 tests using a friend test fixture so we don't leak `_debug_*` accessors into the public API.

---

### Task 1.1: Scalar batched-distance kernel + trait

**Reference.** [JDJ17] §3 (Faiss FlatL2 inner loop pattern).

- [ ] **Step 1: Create `include/listofclusters/detail/batched_distance.hh`**

```cpp
#ifndef _METRIC_DETAIL_BATCHED_DISTANCE_HH_
#define _METRIC_DETAIL_BATCHED_DISTANCE_HH_

#include <listofclusters/glob.hh>
#include <span>
#include <vector>

namespace metric {

// Trait: true when Metric M (over Object O) has a batched-distance kernel
// (scalar default, optionally SIMD via overloads in this header). Default
// false; specializations below flip it for the metrics we batch.
template <class M, class O>
struct supports_batched_distance : std::false_type {};

template <class M, class O>
inline constexpr bool supports_batched_distance_v =
    supports_batched_distance<M, O>::value;

namespace detail {

// Default scalar fallback: reconstruct each row from centers_flat into a
// scratch Object and call the metric functor. Templated so any metric works.
template <class Metric, class Object>
[[nodiscard]] inline std::vector<double> batched_distance(
    const Metric &m,
    const Object &q,
    std::span<const double> centers_flat,
    std::size_t dim,
    std::size_t n)
{
    std::vector<double> out(n);
    Object row{};
    if constexpr (requires { row.resize(dim); }) {
        row.resize(dim);
    }
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < dim; ++j)
            row[j] = centers_flat[i * dim + j];
        out[i] = m(q, row);
    }
    return out;
}

}  // namespace detail
}  // namespace metric

#endif
```

This header is self-contained. Per-metric overloads (Euclidean, L1, L∞) live in this same header in Phase 2 — no reopening of the `metric` namespace from elsewhere.

- [ ] **Step 2: Enable the trait for the four numeric metrics**

At the bottom of `include/listofclusters/metrics.hh`, just before the closing `}  // namespace metric`, add:

```cpp
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
struct supports_batched_distance<euclidean_simd<D>, std::array<double, D>>
    : std::true_type {};
```

The existing closing `}` and `#endif` of `metrics.hh` stay below this block.

- [ ] **Step 3: Add the unit test**

In `tests/test_smoke.cc`, add the include:

```cpp
#include <listofclusters/detail/batched_distance.hh>
```

and the test:

```cpp
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

    static_assert(metric::supports_batched_distance_v<metric::euclidean, v_t>);
    static_assert(metric::supports_batched_distance_v<metric::manhattan, v_t>);
    static_assert(!metric::supports_batched_distance_v<metric::levenshtein, std::string>);

    std::cout << "  test_scalar_batched_distance: OK\n";
}
```

Register `test_scalar_batched_distance();` in `main()`.

- [ ] **Step 4: Build and run**

```sh
cd tests && make clean && make && ./test_smoke
```

Expected: all tests pass (the static_asserts at compile time prove the trait correctly distinguishes numeric vs Levenshtein).

- [ ] **Step 5: Commit**

```sh
git add include/listofclusters/detail/batched_distance.hh include/listofclusters/metrics.hh tests/test_smoke.cc
git commit -m "phase 1.1: scalar batched_distance kernel + trait"
```

---

### Task 1.2: `centers_soa` member, lazy rebuild, `freeze()`, friend test access

**Note on the test hook.** Instead of exposing `_debug_centers_soa()` / `_debug_list()` as public API (which the monolithic plan did, leaking internals), declare a single `friend class lc_test_access` and put debug accessors there. The test fixture lives in `tests/test_smoke.cc`.

- [ ] **Step 1: Modify `include/listofclusters/listofclusters.hh`**

Inside the class, add a friend declaration before the public section:

```cpp
template <class O, class M, size_t B, size_t V> friend class lc_test_access;
```

(Forward-declared in the header — the template parameter names must match the class's.)

After the existing private members, add:

```cpp
public:
    struct centers_soa_t {
        std::vector<double> data;
        std::size_t         dim   = 0;
        std::size_t         n     = 0;
        bool                stale = true;
    };

    // Eagerly build all side structures. Useful for purely-online callers
    // who don't call bulk_build but want to amortize the first-query cost.
    void freeze();

private:
    mutable centers_soa_t _centers;

    void refresh_centers_soa_() const;
```

The `mutable` matters because query methods are `const` but must rebuild on stale.

- [ ] **Step 2: Implement `refresh_centers_soa_` and `freeze`**

Append before the closing `}  // namespace metric` in `listofclusters.hh`:

```cpp
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

- [ ] **Step 3: Wire invalidation and bulk rebuild**

- In `bulk_build`, immediately before the final closing `}`, append: `this->refresh_centers_soa_();`
- In `insert(const object_t&, const uint32_t&)`, at the top: `this->_centers.stale = true;`
- In `remove(const object_t&, const uint32_t&)`, at the top: `this->_centers.stale = true;`
- In `clear()`, after `this->_cid = 0U;`: `this->_centers = centers_soa_t{};`

- [ ] **Step 4: Add the friend test fixture**

In `tests/test_smoke.cc`, **above** the existing tests (and after the `idx_t` typedef), add:

```cpp
// Test-only accessor — friended into `metric::listofclusters` so tests can
// inspect private state without leaking _debug_* methods into the public API.
template <class O, class M, std::size_t B, std::size_t V>
class lc_test_access
{
public:
    using idx_type = metric::listofclusters<O, M, B, V>;
    static const auto& centers_soa(const idx_type &i) { return i._centers; }
    static const auto& list(const idx_type &i) { return i._list; }
};

using lc_access = lc_test_access<vec_t, euclid, 4, 10>;
```

Now add the consistency test:

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

    const auto &soa = lc_access::centers_soa(idx);
    const auto &list = lc_access::list(idx);
    assert(!soa.stale);
    assert(soa.n == list.size());
    assert(soa.dim == D);
    for (std::size_t i = 0; i < list.size(); ++i)
        for (std::size_t j = 0; j < D; ++j)
            assert(soa.data[i * D + j] == list[i].centroid().object()[j]);
    std::cout << "  test_centers_soa_consistency: OK (n=" << soa.n << ", dim=" << soa.dim << ")\n";
}
```

Register in `main()`: `test_centers_soa_consistency();`.

- [ ] **Step 5: Build and run**

```sh
cd tests && make clean && make && ./test_smoke
```

Expected: all tests pass, sanitizers clean.

- [ ] **Step 6: Commit**

```sh
git add include/listofclusters/listofclusters.hh tests/test_smoke.cc
git commit -m "phase 1.2: centers_soa member + lazy rebuild + freeze() + friend test access"
```

---

### Task 1.3: `knn_search` uses batched distance + nearest-first

**Reference.** [CN05] §5 (LC kNN with shrinking radius); the nearest-first traversal is a standard optimization for shrinking-radius search noted across the metric-space literature [CNBM01].

- [ ] **Step 1: Replace `knn_search` body in `include/listofclusters/listofclusters.hh`**

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

        std::vector<std::uint32_t> order(this->_centers.n);
        std::iota(order.begin(), order.end(), 0u);
        std::sort(order.begin(), order.end(),
                  [&](auto a, auto b) noexcept { return d_centers[a] < d_centers[b]; });

        for (std::uint32_t i : order) {
            if (process_cluster(this->_list[i], d_centers[i])) break;
        }
        return results;
    } else {
        // Scalar fallback for non-batched metrics (e.g. Levenshtein on strings):
        // pre-compute d(q, centroid) once and walk nearest-first.
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

Make sure `<numeric>` is in `glob.hh` or add it (`std::iota`).

- [ ] **Step 2: Verify the existing recall tests still pass**

```sh
cd tests && make clean && make && ./test_smoke
```

Expected: `test_knn_matches_brute_force`, `test_batch_knn_matches_serial`, `test_build_and_knn`, `test_large_n_no_crash` all pass.

- [ ] **Step 3: Add a test that asserts the SoA was actually used**

```cpp
static void test_knn_uses_centers_soa()
{
    constexpr std::uint32_t N = 80U;
    constexpr std::size_t D = 4U;
    std::vector<vec_t> db(N);
    std::vector<std::uint32_t> ids(N);
    std::mt19937 rng(33);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    for (std::uint32_t i = 0; i < N; ++i) {
        db[i].resize(D);
        for (std::size_t j = 0; j < D; ++j) db[i][j] = u(rng);
        ids[i] = i;
    }

    idx_t idx;
    idx.bulk_build(db, ids);
    assert(!lc_access::centers_soa(idx).stale);

    // Force online edit, the SoA must mark stale.
    idx.insert(db[0], 9999U);
    assert(lc_access::centers_soa(idx).stale);

    // After a query, the SoA must be refreshed (no longer stale).
    auto r = idx.knn_search(db[0], 12345U, 5);
    (void)r;
    assert(!lc_access::centers_soa(idx).stale);

    std::cout << "  test_knn_uses_centers_soa: OK\n";
}
```

Register `test_knn_uses_centers_soa();` in `main()`.

- [ ] **Step 4: Run tests**

```sh
cd tests && make clean && make && ./test_smoke
```

Expected: all pass.

- [ ] **Step 5: Commit**

```sh
git add include/listofclusters/listofclusters.hh tests/test_smoke.cc
git commit -m "phase 1.3: knn_search uses batched centroid distance + nearest-first walk"
```

---

### Task 1.4: Same restructure for `range_search`

- [ ] **Step 1: Replace the internal `range_search(resultslist_t&, const double&)` body**

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

    const std::size_t n = this->_list.size();
    for (std::size_t i = 0; i < n; ++i) {
        const double d = this->_metric(q, this->_list[i].centroid().object());
        if (process_cluster(this->_list[i], d)) return;
    }
}
```

- [ ] **Step 2: Optionally remove the now-redundant `explore()` declaration**

`explore()` was the old per-cluster recursive helper. If still declared but unused after Phase 0 cleanup, remove the declaration and definition.

- [ ] **Step 3: Run tests**

```sh
cd tests && make clean && make && ./test_smoke
```

Expected: `test_range_search` and all others pass.

- [ ] **Step 4: Commit**

```sh
git add include/listofclusters/listofclusters.hh
git commit -m "phase 1.4: range_search uses batched centroid distance + nearest-first walk"
```

---

### Task 1.5: Bench gate (P1 scalar, no SIMD yet)

- [ ] **Step 1: Build and run the bench**

```sh
cd bench && make clean && make && ./bench_main
```

- [ ] **Step 2: Compare against pre-P1 baseline**

Gate (spec §10.2): `knn k=10 (LC incr, 1T)` and `knn k=10 (LC bulk, 1T)` must improve by ≥1.5× vs pre-P1 numbers (the SoA + nearest-first benefit without SIMD).

If gate misses: profile with `Instruments` (macOS) or `perf` (Linux). Likely culprits in order:
1. argsort dominates → use `std::partial_sort` and only sort the top-`k_clusters_to_visit` indices.
2. Scratch object allocation per row → pre-allocate `row` outside the loop in `detail::batched_distance`.
3. `std::vector<double>` per-call allocation in `batched_distance` → cache via a thread-local or pass an output buffer.

If still misses after profile + fix, revert Phase 1 and document the negative result in `bench/README.md`. Per [memory rule](../../../.claude/projects/-Users-rsolar-repos-liblistofclusters/memory/feedback_remove_negative_results.md), no commit-and-leave-as-opt-in.

- [ ] **Step 3: No commit needed unless code was changed during profiling**

---

## Phase exit check

```sh
cd tests && make clean && make && ./test_smoke    # all green incl. P1 tests
cd ../bench && make clean && make && ./bench_main # gate hit or documented
git log --oneline -5                              # 4 P1 commits
```
