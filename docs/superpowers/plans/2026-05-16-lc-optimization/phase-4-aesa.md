# Phase 4 — AESA-lite global anchor table

**Goal.** Add a metric-agnostic algorithmic filter that prunes individual candidates using k global anchor distances. The biggest wins are at high D and on non-numeric metrics where Phase 2's SIMD doesn't apply (Levenshtein, Canberra at high D, custom metrics).

**References.** [V86] (original AESA, O(N²) memory); [MOV94] (LAESA, the linear-memory variant that motivates our k ≪ N "AESA-lite"); [BNC03] (anchor / pivot selection strategies); [CNBM01] §3 (theoretical foundation of pivot-based filtering).

**The bound.** For any metric `d`, any anchor `a`, any point `p`, and query `q`:

```
d(q, p) ≥ |d(q, a) − d(p, a)|     (triangle inequality)
```

With k anchors the lower bound is `LB(q, p) = max_i |d(q, a_i) − d(p, a_i)|`. If `LB > current_radius`, skip the candidate. The bound is monotone in k.

**Files modified / created.**
- Create: `include/listofclusters/detail/aesa.hh` — `aesa_table_t` struct.
- Modify: `include/listofclusters/listofclusters.hh` — wire in the table, build / refresh / freeze hooks, query-path filter, friend test access.
- Modify: `tests/test_smoke.cc` — sanity check on the lower bound, recall test with AESA on.
- Modify: `python/src/_listofclusters.cc` — expose `build_aesa(k)` and `freeze()` to Python.
- Modify: `python/tests/test_index.py` — exercise the new Python API.

---

### Task 4.1: `aesa_table_t` struct + private member

- [ ] **Step 1: Create `include/listofclusters/detail/aesa.hh`**

```cpp
#ifndef _METRIC_DETAIL_AESA_HH_
#define _METRIC_DETAIL_AESA_HH_

#include <listofclusters/glob.hh>
#include <listofclusters/internal_object.hh>

namespace metric { namespace detail {

// AESA-lite global pivot table: k anchors, N×k stored distances, used to
// derive a triangle-inequality lower bound on d(q, p) per candidate during
// queries. See LAESA [MOV94] for the algorithmic ancestor.
template <class object_t>
struct aesa_table_t {
    using internal_object_t = internal_object<object_t>;

    std::size_t                                       k_anchors = 0;
    std::vector<internal_object_t>                    anchors;
    // Row-major N × k_anchors.
    std::vector<double>                               dists;
    std::vector<std::uint32_t>                        row_to_id;
    std::unordered_map<std::uint32_t, std::uint32_t>  id_to_row;
    bool                                              stale = true;
};

}}  // namespace metric::detail

#endif
```

- [ ] **Step 2: Wire into `include/listofclusters/listofclusters.hh`**

a. Add the include near the top:

```cpp
#include <listofclusters/detail/aesa.hh>
```

b. Inside the class public section (after the existing `centers_soa_t` block):

```cpp
    using aesa_table_t = detail::aesa_table_t<object_t>;

    // Build (or rebuild) the AESA-lite table with k_anchors anchors selected
    // via farthest-first traversal [G85, BNC03]. Pass k_anchors=0 to disable.
    void build_aesa(std::size_t k_anchors);
```

c. Inside the private section (after `_centers`):

```cpp
    mutable aesa_table_t _aesa;

    void refresh_aesa_() const;
```

d. Wire invalidation into the same call sites as `_centers`:

- `insert()`: `this->_aesa.stale = true;` alongside `this->_centers.stale = true;`
- `remove()`: same.
- `clear()`: `this->_aesa = aesa_table_t{};`

e. Extend `freeze()`:

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

f. Extend `lc_test_access` (in `tests/test_smoke.cc`) with an AESA accessor:

```cpp
template <class O, class M, std::size_t B, std::size_t V>
class lc_test_access {
public:
    using idx_type = metric::listofclusters<O, M, B, V>;
    static const auto& centers_soa(const idx_type &i) { return i._centers; }
    static const auto& list(const idx_type &i) { return i._list; }
    static const auto& aesa(const idx_type &i) { return i._aesa; }
};
```

- [ ] **Step 3: Build & confirm the existing tests still pass**

```sh
cd tests && make clean && make && ./test_smoke
```

Expected: all green (no behavior change yet — the AESA table is allocated but unused).

- [ ] **Step 4: Commit**

```sh
git add include/listofclusters/detail/aesa.hh include/listofclusters/listofclusters.hh tests/test_smoke.cc
git commit -m "phase 4.1: AESA-lite table struct + invalidation hooks (no behavior yet)"
```

---

### Task 4.2: Anchor selection (FFT) + table build

**Reference.** [G85] for FFT; [MOV94] §III on how LAESA picks anchors. [BNC03] compares random, FFT, and "outlier" anchor strategies; FFT is the simplest and provably 2-approximate for the max-min objective. We pin the RNG seed for reproducibility.

- [ ] **Step 1: Implement `build_aesa` and `refresh_aesa_`**

Append before the closing `}  // namespace metric` in `listofclusters.hh`:

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

    // Flatten all indexed (non-ghost) points: centroids + bucket members.
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

    // 1. FFT anchor selection [G85, BNC03].
    std::mt19937 rng(0xA5A5A5A5u);
    std::uniform_int_distribution<std::size_t> pick(0, N - 1);
    const std::size_t first = pick(rng);

    this->_aesa.anchors.clear();
    this->_aesa.anchors.reserve(k);
    this->_aesa.anchors.emplace_back(*pts[first].obj, pts[first].id);

    std::vector<double> min_to_anchor(N, std::numeric_limits<double>::infinity());
    for (std::size_t i = 0; i < N; ++i)
        min_to_anchor[i] = this->_metric(*pts[i].obj, *pts[first].obj);

    while (this->_aesa.anchors.size() < k && this->_aesa.anchors.size() < N) {
        std::size_t best = 0;
        double best_d = -1.0;
        for (std::size_t i = 0; i < N; ++i)
            if (min_to_anchor[i] > best_d) { best_d = min_to_anchor[i]; best = i; }
        this->_aesa.anchors.emplace_back(*pts[best].obj, pts[best].id);
        const auto &new_anchor = *pts[best].obj;
        for (std::size_t i = 0; i < N; ++i) {
            const double d = this->_metric(*pts[i].obj, new_anchor);
            if (d < min_to_anchor[i]) min_to_anchor[i] = d;
        }
    }
    const std::size_t actual_k = this->_aesa.anchors.size();
    this->_aesa.k_anchors = actual_k;

    // 2. N × k distance table.
    this->_aesa.dists.assign(N * actual_k, 0.0);
    this->_aesa.row_to_id.assign(N, 0u);
    this->_aesa.id_to_row.clear();
    this->_aesa.id_to_row.reserve(N);
    for (std::size_t i = 0; i < N; ++i) {
        this->_aesa.row_to_id[i] = pts[i].id;
        this->_aesa.id_to_row.emplace(pts[i].id, static_cast<std::uint32_t>(i));
        for (std::size_t j = 0; j < actual_k; ++j)
            this->_aesa.dists[i * actual_k + j] =
                this->_metric(*pts[i].obj, this->_aesa.anchors[j].object());
    }
    this->_aesa.stale = false;
}
```

- [ ] **Step 2: Lower-bound sanity test**

```cpp
static void test_aesa_lower_bound_holds()
{
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

    idx_t idx;
    idx.bulk_build(db, ids);
    idx.build_aesa(8);

    const auto &table = lc_access::aesa(idx);
    assert(table.k_anchors > 0);
    assert(!table.stale);

    // For random queries, LB(q, p) <= true d(q, p) must hold for every p.
    for (int t = 0; t < 50; ++t) {
        vec_t qv(D);
        for (std::size_t j = 0; j < D; ++j) qv[j] = u(rng);
        std::vector<double> dqa(table.k_anchors);
        for (std::size_t i = 0; i < table.k_anchors; ++i)
            dqa[i] = bf_dist(qv, table.anchors[i].object());

        for (std::uint32_t pi = 0; pi < N; ++pi) {
            const double true_d = bf_dist(qv, db[pi]);
            const auto it = table.id_to_row.find(pi);
            if (it == table.id_to_row.end()) continue;
            const double *row = &table.dists[static_cast<std::size_t>(it->second) * table.k_anchors];
            double lb = 0.0;
            for (std::size_t i = 0; i < table.k_anchors; ++i) {
                const double diff = std::abs(dqa[i] - row[i]);
                if (diff > lb) lb = diff;
            }
            assert(lb <= true_d + 1e-9 && "AESA lower bound exceeds true distance");
        }
    }
    std::cout << "  test_aesa_lower_bound_holds: OK\n";
}
```

Register in `main()`.

- [ ] **Step 3: Build & run**

```sh
cd tests && make clean && make && ./test_smoke
```

Expected: all green; the LB invariant holds.

- [ ] **Step 4: Commit**

```sh
git add include/listofclusters/listofclusters.hh tests/test_smoke.cc
git commit -m "phase 4.2: AESA-lite anchor selection (FFT) + distance table [MOV94, G85]"
```

---

### Task 4.3: Wire the AESA filter into `knn_search` and `range_search`

- [ ] **Step 1: Update `knn_search`**

In `include/listofclusters/listofclusters.hh`, replace the body of `knn_search` again (incrementally over Phase 1's version):

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

    // Pre-compute d(q, anchor_i) once per query when AESA is enabled.
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
        for (std::uint32_t i : order)
            if (process_cluster(this->_list[i], d_centers[i])) break;
        return results;
    } else {
        const std::size_t n = this->_list.size();
        std::vector<double> d_centers(n);
        for (std::size_t i = 0; i < n; ++i)
            d_centers[i] = this->_metric(q, this->_list[i].centroid().object());
        std::vector<std::uint32_t> order(n);
        std::iota(order.begin(), order.end(), 0u);
        std::sort(order.begin(), order.end(),
                  [&](auto a, auto b) noexcept { return d_centers[a] < d_centers[b]; });
        for (std::uint32_t i : order)
            if (process_cluster(this->_list[i], d_centers[i])) break;
        return results;
    }
}
```

- [ ] **Step 2: Symmetric update to `range_search`**

In `range_search(resultslist_t&, const double&)`, mirror the AESA pre-compute and the per-bucket-member LB check (with `_radius` instead of the shrinking `radius`). Code structure is identical to `knn_search` above; the differences are the unbounded radius and the `<=` vs `<` predicate.

- [ ] **Step 3: Recall test with AESA on**

```cpp
static void test_aesa_knn_recall_preserved()
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
    assert(!lc_access::aesa(idx).stale);

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
    assert(hits == expected && "AESA filter regressed recall");
    std::cout << "  test_aesa_knn_recall_preserved: OK (" << hits << "/" << expected << ")\n";
}
```

Register in `main()`.

- [ ] **Step 4: Run tests**

```sh
cd tests && make clean && make && ./test_smoke
```

Expected: all green; AESA filter active, recall preserved.

- [ ] **Step 5: Commit**

```sh
git add include/listofclusters/listofclusters.hh tests/test_smoke.cc
git commit -m "phase 4.3: wire AESA-lite filter into knn_search / range_search"
```

---

### Task 4.4: Python bindings expose `build_aesa(k)` and `freeze()`

- [ ] **Step 1: Update `python/src/_listofclusters.cc`**

In `IndexBase` (virtual interface):

```cpp
virtual void build_aesa(std::size_t k_anchors) = 0;
virtual void freeze() = 0;
```

In `IndexImpl<Metric>` (override):

```cpp
void build_aesa(std::size_t k_anchors) override { _idx.build_aesa(k_anchors); }
void freeze() override { _idx.freeze(); }
```

In the `nb::class_<IndexBase>` block, after the existing `.def("bulk_build", …)`:

```cpp
.def("build_aesa",
    &IndexBase::build_aesa,
    nb::arg("k_anchors"),
    "Build the AESA-lite k-anchor pivot table for tighter candidate "
    "filtering. k_anchors=0 disables and frees the table.")
.def("freeze",
    &IndexBase::freeze,
    "Eagerly build / refresh all side structures "
    "(centers SoA, AESA-lite if enabled). Amortizes first-query cost.")
```

- [ ] **Step 2: Add a Python test**

In `python/tests/test_index.py`:

```python
def test_build_aesa_preserves_recall():
    rng = np.random.default_rng(11)
    db = rng.uniform(-1.0, 1.0, size=(200, 4))
    ids = np.arange(len(db), dtype=np.uint32)

    idx = Index(metric="euclidean")
    idx.bulk_build(db, ids)
    idx.build_aesa(8)
    idx.freeze()

    for _ in range(5):
        q = rng.uniform(-1.0, 1.0, size=4)
        nbrs, _ = idx.knn(q, k=5)

        bf = np.linalg.norm(db - q, axis=1)
        expected = np.argsort(bf)[:5]
        for e in expected:
            assert e in set(nbrs), f"AESA-on knn missed {e}"


def test_build_aesa_zero_disables():
    idx = Index(metric="euclidean")
    idx.insert(np.array([0.0, 1.0]), np.uint32(0))
    idx.insert(np.array([1.0, 0.0]), np.uint32(1))
    idx.build_aesa(0)
    # No throw, no recall change.
    nbrs, _ = idx.knn(np.array([0.5, 0.5]), k=2)
    assert set(nbrs) == {0, 1}
```

- [ ] **Step 3: Build the wheel + run the tests**

```sh
cd python && uv pip install -e . && pytest tests/ -v
```

Expected: all green.

- [ ] **Step 4: Commit**

```sh
git add python/src/_listofclusters.cc python/tests/test_index.py
git commit -m "phase 4.4: Python bindings expose build_aesa + freeze"
```

---

### Task 4.5: Bench gate (P2 with AESA on top of P1+P2+P3)

- [ ] **Step 1: Add a temporary AESA-on bench row**

In `bench/bench_main.cc`, clone `bench_knn` into `bench_knn_aesa`:

```cpp
[[nodiscard]] static Stats
bench_knn_aesa(const std::vector<vec_t> &db, const std::vector<vec_t> &queries,
               std::size_t k, std::size_t k_anchors, int repeats)
{
    std::vector<std::uint32_t> ids(db.size());
    for (std::uint32_t i = 0; i < db.size(); ++i) ids[i] = i;
    idx_t<> idx;
    idx.bulk_build(db, ids);
    idx.build_aesa(k_anchors);
    idx.freeze();

    std::vector<double> samples_ns;
    samples_ns.reserve(repeats);
    volatile std::size_t sink = 0;
    for (int r = 0; r < repeats; ++r) {
        const double ns = time_ns([&] {
            for (std::uint32_t q = 0; q < queries.size(); ++q) {
                auto res = idx.knn_search(queries[q],
                    static_cast<std::uint32_t>(db.size() + q), k);
                sink += res.results().size();
            }
        });
        samples_ns.push_back(ns);
    }
    (void)sink;
    return summarize(std::move(samples_ns), queries.size());
}
```

Register a `print_row("knn k=10 (LC AESA k=16, 1T)", queries.size(), …);` row in `main()`.

- [ ] **Step 2: Run at D=32 with metric L1**

You'll need to swap the `euclid` functor and the metric used by `idx_t<>` to L1 for the spec's L1 D=32 gate. Easiest: add a parallel typedef:

```cpp
struct manhattan_d { /* ... */ };  // L1 on vec_t
template <std::size_t bucket = 20, std::size_t overflow = 80>
using idx_l1_t = metric::listofclusters<vec_t, manhattan_d, bucket, overflow>;
```

and re-run.

- [ ] **Step 3: Verify gate**

Spec §10.2: P2 with k=16 ≥ 1.5× vs P1+P3+P4 only, on L1 at D=32. If miss, profile (likely `id_to_row.find` is hot — move the row index to a field on `internal_object_t` per the spec's note on this in §6.4).

If miss after fix, revert P2 and document.

- [ ] **Step 4: No commit unless code changes**

---

## Phase exit check

```sh
cd tests && make clean && make && ./test_smoke
cd ../python && uv pip install -e . && pytest tests/ -v
cd ../bench && make clean && make && ./bench_main
git log --oneline -6
```
