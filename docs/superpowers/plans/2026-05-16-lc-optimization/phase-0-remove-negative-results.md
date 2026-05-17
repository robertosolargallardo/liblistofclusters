# Phase 0 — Remove negative-result infrastructure

**Goal.** Drop the two documented negative results (block index from Phase 5.8, per-cluster pivots from Phase 5.6) before any new API surface is added. Cleanest base for the rest of the work.

**References.** [`references.md`](references.md) §"Diversity-promoting alternatives explored and rejected". Spec §9.1 / §9.2.

**Files modified.**
- `include/listofclusters/listofclusters.hh` — drop `_blocks`, `block_t`, `build_block_index()`; simplify `range_search` / `knn_search`; drop `use_pivots` parameter and pivot-build block from `bulk_build`.
- `include/listofclusters/cluster.hh` — drop `_pivot`, `has_pivot()`, `pivot()`, `set_pivot()`.
- `include/listofclusters/internal_object.hh` — drop `_pivot_distance` field and accessors.
- `tests/test_smoke.cc` — drop `test_bulk_build_with_pivots`.
- `bench/bench_main.cc` — drop `bench_knn_incr_blocked` and its row.
- `bench/README.md` — drop the `LC incr+blocks` row from the cached table.
- `python/src/_listofclusters.cc` — drop the `use_pivots` argument from the `bulk_build` binding.

---

### Task 0.1: Remove block index from the main header

- [ ] **Step 1: Run the existing test suite to baseline pass state**

```sh
cd tests && make clean && make && ./test_smoke
```

Expected: all 12 tests pass under ASan+UBSan.

- [ ] **Step 2: Edit `include/listofclusters/listofclusters.hh`**

a. Delete the `block_t` struct (the `struct block_t { … };` block inside the class public section).

b. Delete the `_blocks` private member (the `std::vector<block_t> _blocks;` line).

c. Delete the `build_block_index` declaration:

```cpp
    void build_block_index(std::size_t block_size = 0);
```

d. Delete the entire `build_block_index` definition (the `void listofclusters<…>::build_block_index(…) { … }` function).

e. In `clear()`, delete `this->_blocks.clear();`.

f. In `range_search(resultslist_t&, const double&)`, delete the `const bool tiered = !this->_blocks.empty();` line and the `if (tiered) { … return; }` block. The function's body is the existing flat-scan loop using `process_cluster`.

g. In `knn_search`, delete the `if (!this->_blocks.empty()) { … return results; }` block. The function's body is the existing flat-scan loop using `process_cluster`.

- [ ] **Step 3: Sweep for stragglers**

```sh
grep -rn '_blocks\|block_t\|build_block_index' include/ tests/ bench/ python/
```

Expected output: hits only in `bench/bench_main.cc` and `bench/README.md` (cleaned in Tasks 0.4 / 0.5).

- [ ] **Step 4: Run the test suite**

```sh
cd tests && make clean && make && ./test_smoke
```

Expected: all 12 tests pass.

- [ ] **Step 5: Commit**

```sh
git add include/listofclusters/listofclusters.hh
git commit -m "phase 0: remove two-tier block index machinery (Phase 5.8 negative result)"
```

---

### Task 0.2: Remove per-cluster pivots

- [ ] **Step 1: Delete the existing pivot test**

In `tests/test_smoke.cc`, delete the entire function `test_bulk_build_with_pivots()` and its call site inside `main()`.

- [ ] **Step 2: Remove pivot machinery from `include/listofclusters/cluster.hh`**

- Delete `internal_object_t _pivot{};` (private member, around line 33).
- Delete the doc comment immediately above it (lines 26–32).
- Delete `pivot()`, `has_pivot()`, and `set_pivot()` member functions.
- In `clear()`, delete `this->_pivot.ghost(true);`.

- [ ] **Step 3: Remove pivot_distance from `include/listofclusters/internal_object.hh`**

- Delete the `_pivot_distance` member field and its inline comment.
- Delete both `pivot_distance()` accessors (getter + setter).

- [ ] **Step 4: Update `bulk_build` in `include/listofclusters/listofclusters.hh`**

a. Change the declaration (drop the `use_pivots` parameter):

```cpp
void bulk_build(const std::vector<object_t> &objs,
                const std::vector<uint32_t> &ids);
```

b. Change the definition signature similarly.

c. Inside the definition, delete the `if (use_pivots && take >= 2) { … cluster.set_pivot(…, pivot_dists); }` block (~14 lines).

- [ ] **Step 5: Sweep range_search and knn_search for pivot branches**

In `listofclusters.hh`, inside `range_search(resultslist_t&, double)`, `explore(...)`, and `knn_search(...)`, delete every branch gated on `c.has_pivot()` / `use_pivot`:

- `const bool use_pivot = c.has_pivot();`
- `bool dp_set = false; double dp = 0.0;`
- The inner `if (use_pivot) { … }` block that computes / consults `pivot_distance`.

- [ ] **Step 6: Remove `use_pivots` from the Python bindings**

In `python/src/_listofclusters.cc`:

- In the `IndexBase` virtual: change `virtual void bulk_build(...)`'s signature to drop the `bool use_pivots` argument.
- In `IndexImpl::bulk_build` (override): drop the `bool use_pivots` argument; the body becomes `_idx.bulk_build(objs, ids);`.
- In the `.def("bulk_build", ...)` lambda: drop the `bool use_pivots` argument and the matching `nb::arg("use_pivots") = false`. The lambda becomes:

```cpp
.def("bulk_build",
    [](IndexBase& self,
       nb::ndarray<const double, nb::ndim<2>, nb::c_contig> V,
       nb::ndarray<const std::uint32_t, nb::ndim<1>, nb::c_contig> ids) {
        if (V.shape(0) != ids.shape(0))
            throw std::invalid_argument("V.shape[0] must equal ids.shape[0]");
        self.bulk_build(to_vec2d(V), to_ids(ids));
    },
    nb::arg("V"), nb::arg("ids"),
    "Canonical LC construction. Replaces any existing index state.")
```

- [ ] **Step 7: Sweep for any remaining references**

```sh
grep -rn 'use_pivots\|has_pivot\|pivot_distance\|set_pivot' include/ tests/ bench/ python/
```

Expected: empty output.

- [ ] **Step 8: Run all tests (C++ and Python)**

```sh
cd tests && make clean && make && ./test_smoke
cd ../python && uv pip install -e . && pytest tests/ -v
```

Expected: all green. Python's `test_*` suite must pass (it doesn't reference `use_pivots`, so this is a pure subtraction).

- [ ] **Step 9: Commit**

```sh
git add include/listofclusters/listofclusters.hh include/listofclusters/cluster.hh include/listofclusters/internal_object.hh tests/test_smoke.cc python/src/_listofclusters.cc
git commit -m "phase 0: remove per-cluster pivot infrastructure (Phase 5.6 negative result)"
```

---

### Task 0.3: Drop the blocks variant from the bench

- [ ] **Step 1: Edit `bench/bench_main.cc`**

- Delete the `bench_knn_incr_blocked` function (around line 282).
- Delete the corresponding `print_row("knn k=10 (LC incr+blocks, 1T)", …);` call in `main()`.

- [ ] **Step 2: Rebuild the bench**

```sh
cd bench && make clean && make
```

Expected: compiles cleanly. No references to `build_block_index` or `bench_knn_incr_blocked`.

- [ ] **Step 3: Smoke-run the bench**

```sh
./bench_main
```

Expected: prints the bench table without the `LC incr+blocks` row.

- [ ] **Step 4: Commit**

```sh
git add bench/bench_main.cc
git commit -m "phase 0: drop block-index variant from bench"
```

---

### Task 0.4: Drop the blocks row from the cached bench README table

- [ ] **Step 1: Edit `bench/README.md`**

Inside the `Latest run on this machine` code block (the cached table starting `benchmark                        ops/sample  per-op (us) …`), delete the line:

```
knn k=10 (LC incr+blocks, 1T)           200       10.893          0.092
```

- [ ] **Step 2: Commit**

```sh
git add bench/README.md
git commit -m "phase 0: drop block-index row from bench README cache"
```

---

## Phase exit check

```sh
cd tests && make clean && make && ./test_smoke    # ASan+UBSan green
cd ../bench && make clean && make && ./bench_main # bench builds and runs
cd ../python && pytest tests/ -v                  # Python wrapper green
git log --oneline -5                              # 4 commits: 0.1 through 0.4
```

No QPS gate at Phase 0 — pure removal, no optimization yet. Subsequent phases add the speed.
