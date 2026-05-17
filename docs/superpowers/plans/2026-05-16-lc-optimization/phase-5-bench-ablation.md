# Phase 5 — Bench harness extensions for ablation

**Goal.** Productionize the LC-variant rows added ad-hoc during Phase 1–4 bench gates into a clean ablation matrix. Add bytes/point reporting so the AESA / centers_soa memory cost is visible alongside QPS. Update `bench/compare.py` to drop references to removed flags and to optionally turn AESA on.

**Files modified.**
- `bench/bench_main.cc` — replace ad-hoc rows from Phases 1–4 with a small config-driven loop.
- `bench/compare.py` — drop `use_pivots` / `build_block_index` refs; add `--k-anchors` flag.
- `bench/README.md` — the synthetic-ablation section gets reworked here (the headline real-document section comes in Phase 6).

---

### Task 5.1: Add `bench_knn_lc_variant` driver

The existing bench uses a function-per-row pattern (`bench_insert`, `bench_knn`, `bench_knn_bulk`, etc.) feeding into `summarize()` / `print_row()`. Phase 5.1 follows the same pattern instead of inventing a parallel one.

- [ ] **Step 1: Add the variant driver in `bench/bench_main.cc`**

After the existing `bench_knn_bulk`:

```cpp
// LC kNN with selectable build strategy and AESA k. Returns the same Stats
// format as the other bench_knn_* drivers.
[[nodiscard]] static Stats
bench_knn_lc_variant(const std::vector<vec_t> &db,
                     const std::vector<vec_t> &queries,
                     std::size_t k,
                     int repeats,
                     idx_t<>::build_strategy strategy,
                     std::size_t k_anchors)
{
    std::vector<std::uint32_t> ids(db.size());
    for (std::uint32_t i = 0; i < db.size(); ++i) ids[i] = i;
    idx_t<> idx;
    idx.bulk_build(db, ids, strategy);
    if (k_anchors > 0) idx.build_aesa(k_anchors);
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

- [ ] **Step 2: Register the ablation rows in `main()`**

Replace the temporary AESA-only row added in Phase 4.5 with:

```cpp
struct lc_variant { const char* label; idx_t<>::build_strategy strategy; std::size_t k_anchors; };
constexpr lc_variant kVariants[] = {
    {"LC (legacy first_unassigned, no AESA)",
        idx_t<>::build_strategy::first_unassigned, 0},
    {"LC (FFT, no AESA)",
        idx_t<>::build_strategy::farthest_first,   0},
    {"LC (FFT + AESA k=8)",
        idx_t<>::build_strategy::farthest_first,   8},
    {"LC (FFT + AESA k=16)",
        idx_t<>::build_strategy::farthest_first,  16},
    {"LC (FFT + AESA k=32)",
        idx_t<>::build_strategy::farthest_first,  32},
};
for (const auto &v : kVariants) {
    const auto s = bench_knn_lc_variant(db, queries, k, repeats, v.strategy, v.k_anchors);
    print_row(v.label, queries.size(), s);
}
```

- [ ] **Step 3: Build and run**

```sh
cd bench && make clean && make && ./bench_main
```

Expected: 5 new ablation rows appear, sorted by config.

- [ ] **Step 4: Commit**

```sh
git add bench/bench_main.cc
git commit -m "phase 5.1: LC ablation rows (build_strategy × k_anchors)"
```

---

### Task 5.2: Bytes/point reporting

- [ ] **Step 1: Add an estimator**

In `bench/bench_main.cc`, near the timing helpers:

```cpp
template <class IdxT>
[[nodiscard]] static std::size_t
estimate_index_bytes(const IdxT &idx, std::size_t N_points)
{
    // We need a way to introspect; use the same friend-class hook the tests use.
    using A = lc_test_access<vec_t, euclid, 20, 80>;
    const auto &soa  = A::centers_soa(idx);
    const auto &list = A::list(idx);
    const auto &aesa = A::aesa(idx);

    std::size_t bytes = 0;
    bytes += soa.data.size() * sizeof(double);
    for (const auto &c : list) {
        bytes += sizeof(c);
        bytes += c.bucket().size() * sizeof(typename IdxT::internal_object_t);
    }
    bytes += aesa.dists.size() * sizeof(double);
    bytes += aesa.row_to_id.size() * sizeof(std::uint32_t);
    // unordered_map overhead estimate: ~32 bytes per bucket; refine if hot.
    bytes += aesa.id_to_row.size() * 32;
    (void)N_points;
    return bytes;
}
```

`lc_test_access` already lives in `tests/test_smoke.cc`. For the bench, copy the small accessor declaration into `bench/bench_main.cc` (it's templated on the same parameters; declared on top in an anonymous namespace).

- [ ] **Step 2: Extend `print_row` to optionally show bytes/point**

```cpp
static void print_row_with_size(const std::string &name,
                                std::size_t ops_per_sample,
                                const Stats &s,
                                std::size_t bytes_per_point)
{
    std::printf("%-40s %10zu %12.3f %14.3f %14zu\n",
                name.c_str(), ops_per_sample,
                s.per_op_ns / 1000.0, s.ops_per_s / 1e6,
                bytes_per_point);
}
```

Update `print_header` to add the column:

```cpp
std::printf("%-40s %10s %12s %14s %14s\n",
            "benchmark", "ops/sample", "per-op (us)",
            "throughput (M/s)", "bytes/point");
```

For LC variants, build the index once (separately from the timing loop), call `estimate_index_bytes`, then run the timing loop. Or factor: have `bench_knn_lc_variant` return both `Stats` and the byte count via a small struct.

- [ ] **Step 3: Rebuild + run**

```sh
cd bench && make clean && make && ./bench_main
```

Expected: each LC row now reports bytes/point alongside QPS.

- [ ] **Step 4: Commit**

```sh
git add bench/bench_main.cc
git commit -m "phase 5.2: bench reports bytes/point for LC variants"
```

---

### Task 5.3: Update `bench/compare.py`

- [ ] **Step 1: Audit for references to removed flags**

```sh
grep -n 'use_pivots\|build_block_index' bench/compare.py
```

Delete or update every hit. The `bulk_build` call site changes from:

```python
idx.bulk_build(V, ids, use_pivots=False)
```

to:

```python
idx.bulk_build(V, ids)
```

- [ ] **Step 2: Add a `--k-anchors` CLI flag**

In `bench/compare.py`'s argparse section, add:

```python
parser.add_argument("--k-anchors", type=int, default=0,
                    help="If >0, call idx.build_aesa(k) before queries.")
```

In the LC method runner:

```python
idx.bulk_build(V, ids)
if args.k_anchors > 0:
    idx.build_aesa(args.k_anchors)
idx.freeze()
```

- [ ] **Step 3: Smoke**

```sh
uv pip install -e ./python
python bench/compare.py --n 1000 --d 8 --metric euclidean --queries 50 --k 10 --k-anchors 8
```

Expected: completes, writes CSV / per-metric plot.

- [ ] **Step 4: Commit**

```sh
git add bench/compare.py
git commit -m "phase 5.3: compare.py uses new API (drops use_pivots, adds --k-anchors)"
```

---

## Phase exit check

```sh
cd bench && make clean && make && ./bench_main
python bench/compare.py --n 5000 --d 8,32 --metric euclidean,manhattan --queries 100 --k 10 --k-anchors 16
git log --oneline -4
```
