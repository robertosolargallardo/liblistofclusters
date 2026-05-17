# Phase 3 — Farthest-First Traversal in `bulk_build`

**Goal.** Replace the "first unassigned" center selection in `bulk_build` with Gonzalez's Farthest-First Traversal [G85]. FFT produces well-separated centers and tighter cluster radii, which compounds with Phase 1's nearest-first walk (the right cluster shows up first more often) and Phase 2's batched centroid distances (the tighter `cluster.radius` means more early-termination triggers).

**References.** [G85] (FFT algorithm); [HS85] (2-approximation proof); [BNC03] (pivot selection study — FFT competitive among simple choices); [CN05] §3 (the original LC paper's static-build pseudocode, which uses "first unassigned").

**Files modified.**
- `include/listofclusters/listofclusters.hh` — add `build_strategy` enum; rewrite `bulk_build` center pick to FFT incrementally maintaining `min_to_center[]`.

**API impact.** New default. `bulk_build` gains a strategy enum parameter. The old `first_unassigned` path stays available for A/B comparison and paper-faithful repro.

---

### Task 3.1: Add `build_strategy` enum + FFT center selection

**Algorithm reference.** [G85] §3. Maintain `min_to_center[i]` = distance from point i to the nearest already-chosen center. To pick the next center, take `argmax_i min_to_center[i]` over the unassigned set. When a new center is added, update each unassigned point's `min_to_center[i] = min(min_to_center[i], d(point_i, new_center))`. The update is one distance per unassigned point per center added — total O(N · |centers|) = O((N/m) · N) for an LC build, matching the original "first unassigned" cost asymptotically with a ~2× constant factor (one extra distance pass per cluster to update `min_to_center`).

[HS85] proves Gonzalez's algorithm is a 2-approximation to the k-center problem and that no polynomial-time (2 − ε)-approximation exists unless P = NP.

- [ ] **Step 1: Add the enum to `include/listofclusters/listofclusters.hh`**

Inside the class, in the public section before the `bulk_build` declaration:

```cpp
    enum class build_strategy { first_unassigned, farthest_first };
```

- [ ] **Step 2: Update the `bulk_build` declaration**

```cpp
    // Canonical static LC build (Chavez & Navarro PRL 2005, §3) with two
    // center-selection strategies:
    //   - first_unassigned: paper-faithful; pick the lowest-index unassigned
    //     point as the next center.
    //   - farthest_first  : Gonzalez FFT [G85]; pick the unassigned point
    //     with the maximum min-distance to already-chosen centers. Produces
    //     spatially separated centers, tighter cluster radii, better pruning.
    // Default is farthest_first.
    void bulk_build(const std::vector<object_t> &objs,
                    const std::vector<uint32_t> &ids,
                    build_strategy strategy = build_strategy::farthest_first);
```

- [ ] **Step 3: Replace `bulk_build`'s function body**

```cpp
template <class object_t, class distance_t, size_t bucket_size, size_t overflow>
    requires Metric<distance_t, object_t>
void listofclusters<object_t,distance_t,bucket_size,overflow>::bulk_build(
    const std::vector<object_t> &objs,
    const std::vector<uint32_t> &ids,
    build_strategy strategy)
{
    this->clear();
    const std::size_t n = std::min(objs.size(), ids.size());
    if (n == 0) return;

    std::vector<char> assigned(n, 0);
    std::size_t remaining = n;

    // For FFT: minimum distance from each (still-unassigned) point to any
    // already-chosen center. Updated incrementally as centers are added.
    std::vector<double> min_to_center(n, std::numeric_limits<double>::infinity());

    std::vector<std::pair<double, std::size_t>> scratch;
    scratch.reserve(n);

    this->_list.reserve(n / bucket_size + 1);

    // Fixed seed for reproducible FFT seeding. Future work could thread a
    // user-provided seed if non-determinism becomes important.
    std::mt19937 rng(0xC0FFEEu);

    while (remaining > 0) {
        // 1. Pick the next center.
        std::size_t c_idx = 0;
        if (this->_list.empty()) {
            // First center: random unassigned for FFT, first unassigned for legacy.
            if (strategy == build_strategy::farthest_first) {
                std::uniform_int_distribution<std::size_t> pick(0, n - 1);
                do { c_idx = pick(rng); } while (assigned[c_idx]);
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

        // 2. Distances from new center to remaining unassigned points; update
        //    min_to_center as we go (so the next FFT pick is O(N) without a
        //    per-pair recompute).
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

        this->_list.push_back(std::move(cluster));
    }

    this->refresh_centers_soa_();
}
```

Note: the old `use_pivots` block and `pivot_dists` scratch from the previous bulk_build are absent here — Phase 0 already removed them.

- [ ] **Step 4: Add recall test for the new strategy**

In `tests/test_smoke.cc`:

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

Register in `main()`.

The existing `test_bulk_build_matches_brute_force` continues to pass against the new default (FFT). To explicitly exercise the legacy path:

```cpp
static void test_first_unassigned_build_matches_brute_force()
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
    idx.bulk_build(db, ids, idx_t::build_strategy::first_unassigned);

    int hits = 0, expected = 0;
    for (int q = 0; q < 10; ++q) {
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
    assert(hits == expected);
    std::cout << "  test_first_unassigned_build_matches_brute_force: OK\n";
}
```

Register both in `main()`.

- [ ] **Step 5: Run tests**

```sh
cd tests && make clean && make && ./test_smoke
```

Expected: all green.

- [ ] **Step 6: Commit**

```sh
git add include/listofclusters/listofclusters.hh tests/test_smoke.cc
git commit -m "phase 3.1: farthest-first traversal default for bulk_build [G85]"
```

---

### Task 3.2: Bench gate (P3 on top of P1+P2)

- [ ] **Step 1: Bench compare FFT vs first_unassigned**

The existing bench harness already builds via `bulk_build(db, ids)` — with the new default this calls FFT. To compare to first_unassigned, add a temporary parallel row that calls `idx.bulk_build(db, ids, idx_t<>::build_strategy::first_unassigned)`.

```sh
cd bench && make clean && make BENCH_FLAGS="-DBENCH_D=32" && ./bench_main
```

- [ ] **Step 2: Verify gate**

Spec §10.2: P3 ≥ 1.2× vs `first_unassigned`, holding P1+P2 on, at any D.

If gate misses, profile. Common diagnoses:
- FFT center selection is dominated by the O(N) argmax scan rather than the per-pair distance cost. Mitigation: cache `min_to_center` in a sorted structure (e.g., a max-heap keyed by `min_to_center[i]`) — but ASLR + cache invalidation usually means a flat scan is fine at the N we benchmark.
- Tighter cluster radii produce worse pruning when bucket members are far. Mitigation: sanity-check `cluster.radius` distribution before vs after. If FFT actually inflates the average radius at high D (curse of dimensionality), it's a known signal — see [CNBM01] §6 on intrinsic dimensionality.

If still missing, revert and document.

- [ ] **Step 3: No commit unless profiling required code changes**

---

## Phase exit check

```sh
cd tests && make clean && make && ./test_smoke
cd ../bench && make clean && make && ./bench_main
git log --oneline -3
```
