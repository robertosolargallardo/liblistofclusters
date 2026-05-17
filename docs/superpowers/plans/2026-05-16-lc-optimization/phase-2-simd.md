# Phase 2 — SIMD specializations for batched distance

**Goal.** Push the per-distance cost of the centroid pass down to Faiss `FlatL2` territory by adding NEON (arm64) / AVX2 (x86_64) inner kernels for the three numeric metrics that batch cleanly: `euclidean`, `manhattan`, `chebyshev`.

**References.** [JDJ17] §3 (FlatL2 SIMD inner loop pattern); existing `metric::euclidean_simd<D>` from Phase 5.7 (NEON/AVX2 templates with FMA). Phase 5.7 SIMD'd one distance at a time; Phase 2 lifts the outer loop into the SIMD region. ARM NEON and Intel AVX2 intrinsics manuals are the source of truth for the operations used (`vfmaq_f64`, `_mm256_fmadd_pd`, `vmaxq_f64`, `_mm256_max_pd`, `_mm256_andnot_pd` for `fabs`).

**Files modified.**
- `include/listofclusters/detail/batched_distance.hh` — add three per-metric impls (Euclidean, Manhattan, Chebyshev) under arm64/x86_64/scalar branches. Add per-metric overloads of `detail::batched_distance` that route through the impls.
- `bench/Makefile` — wire `BENCH_FLAGS` to `CXXFLAGS` so D can be varied without editing source.

---

### Task 2.1: Wire `BENCH_FLAGS` so we can sweep D from the command line

Required up front so the gate-check steps don't paper over a missing env var.

- [ ] **Step 1: Edit `bench/Makefile`**

Change:

```make
CXXFLAGS?=-Wall -Wextra -Wpedantic -Wno-unused-parameter -O3 -std=c++23 -DNDEBUG -pthread
```

to:

```make
BENCH_FLAGS?=
CXXFLAGS?=-Wall -Wextra -Wpedantic -Wno-unused-parameter -O3 -std=c++23 -DNDEBUG -pthread $(BENCH_FLAGS)
```

- [ ] **Step 2: Smoke-test the override**

```sh
cd bench && make clean && make BENCH_FLAGS="-DBENCH_D=32" && ./bench_main 10000 32 200 uniform
```

Expected: `kD = 32` at runtime; the bench reports rows for D=32.

- [ ] **Step 3: Commit**

```sh
git add bench/Makefile
git commit -m "phase 2.1: bench Makefile honors BENCH_FLAGS for sweeping D"
```

---

### Task 2.2: NEON / AVX2 batched Euclidean

- [ ] **Step 1: Add the impl in `include/listofclusters/detail/batched_distance.hh`**

Below the `detail::batched_distance` template, inside the same `metric::detail` namespace, append:

```cpp
#if defined(__ARM_NEON)
#  include <arm_neon.h>
#endif
#if defined(__AVX2__)
#  include <immintrin.h>
#endif

namespace metric { namespace detail {

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
#elif defined(__AVX2__)
    for (std::size_t i = 0; i < n; ++i) {
        const double *row = centers_flat.data() + i * dim;
        __m256d acc = _mm256_setzero_pd();
        std::size_t j = 0;
        for (; j + 4 <= dim; j += 4) {
            __m256d va = _mm256_setr_pd(
                static_cast<double>(q[j]),     static_cast<double>(q[j + 1]),
                static_cast<double>(q[j + 2]), static_cast<double>(q[j + 3]));
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
#else
    for (std::size_t i = 0; i < n; ++i) {
        const double *row = centers_flat.data() + i * dim;
        double s = 0.0;
        for (std::size_t j = 0; j < dim; ++j) {
            const double d = static_cast<double>(q[j]) - row[j];
            s += d * d;
        }
        out[i] = std::sqrt(s);
    }
#endif
    return out;
}

}}  // namespace metric::detail
```

- [ ] **Step 2: Wire the overload from `metrics.hh`**

In `include/listofclusters/metrics.hh`, in the trait-specialization block at the bottom, append (still inside `namespace metric { ... }`):

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

This overload is preferred over the generic template when the metric is `euclidean`.

- [ ] **Step 3: Add a numerical cross-check test**

```cpp
static void test_simd_batched_euclidean_matches_scalar()
{
    using v_t = std::vector<double>;
    const std::size_t dim = 8, n = 64;
    std::vector<double> centers(n * dim);
    std::mt19937 rng(1);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    for (auto &x : centers) x = u(rng);
    v_t q(dim);
    for (auto &x : q) x = u(rng);

    auto simd = metric::detail::batched_distance(
        metric::euclidean{}, q,
        std::span<const double>(centers.data(), centers.size()), dim, n);

    metric::euclidean m{};
    v_t row(dim);
    auto approx = [](double x, double y) { return std::abs(x - y) < 1e-9; };
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < dim; ++j) row[j] = centers[i * dim + j];
        assert(approx(simd[i], m(q, row)));
    }
    std::cout << "  test_simd_batched_euclidean_matches_scalar: OK\n";
}
```

Register in `main()`.

- [ ] **Step 4: Build and run**

```sh
cd tests && make clean && make && ./test_smoke
```

Expected: all tests pass under ASan+UBSan.

- [ ] **Step 5: Commit**

```sh
git add include/listofclusters/detail/batched_distance.hh include/listofclusters/metrics.hh tests/test_smoke.cc
git commit -m "phase 2.2: NEON/AVX2 batched euclidean kernel"
```

---

### Task 2.3: Manhattan + Chebyshev SIMD impls

L1 wants vector `abs` + horizontal `add`. L∞ wants vector `abs` + horizontal `max`. NEON has `vabsq_f64`; AVX2 emulates `fabs(double)` with `_mm256_andnot_pd(signmask, x)` where `signmask = _mm256_set1_pd(-0.0)`.

- [ ] **Step 1: Add impls in `include/listofclusters/detail/batched_distance.hh`**

Inside `metric::detail`:

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
    for (std::size_t i = 0; i < n; ++i) {
        const double *row = centers_flat.data() + i * dim;
        float64x2_t acc = vdupq_n_f64(0.0);
        std::size_t j = 0;
        for (; j + 2 <= dim; j += 2) {
            float64x2_t va = { static_cast<double>(q[j]), static_cast<double>(q[j + 1]) };
            float64x2_t vb = vld1q_f64(row + j);
            acc = vaddq_f64(acc, vabsq_f64(vsubq_f64(va, vb)));
        }
        double s = vgetq_lane_f64(acc, 0) + vgetq_lane_f64(acc, 1);
        for (; j < dim; ++j)
            s += std::abs(static_cast<double>(q[j]) - row[j]);
        out[i] = s;
    }
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
#else
    for (std::size_t i = 0; i < n; ++i) {
        const double *row = centers_flat.data() + i * dim;
        double s = 0.0;
        for (std::size_t j = 0; j < dim; ++j)
            s += std::abs(static_cast<double>(q[j]) - row[j]);
        out[i] = s;
    }
#endif
    return out;
}

template <class Object>
[[nodiscard]] inline std::vector<double> batched_distance_chebyshev_impl(
    const Object &q,
    std::span<const double> centers_flat,
    std::size_t dim,
    std::size_t n)
{
    std::vector<double> out(n);
#if defined(__ARM_NEON)
    for (std::size_t i = 0; i < n; ++i) {
        const double *row = centers_flat.data() + i * dim;
        float64x2_t m = vdupq_n_f64(0.0);
        std::size_t j = 0;
        for (; j + 2 <= dim; j += 2) {
            float64x2_t va = { static_cast<double>(q[j]), static_cast<double>(q[j + 1]) };
            float64x2_t vb = vld1q_f64(row + j);
            m = vmaxq_f64(m, vabsq_f64(vsubq_f64(va, vb)));
        }
        double mx = std::max(vgetq_lane_f64(m, 0), vgetq_lane_f64(m, 1));
        for (; j < dim; ++j) {
            const double d = std::abs(static_cast<double>(q[j]) - row[j]);
            if (d > mx) mx = d;
        }
        out[i] = mx;
    }
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
#endif
    return out;
}
```

- [ ] **Step 2: Add the overloads in `metrics.hh`**

In the same trait-specialization block:

```cpp
namespace detail {
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
}  // namespace detail
```

- [ ] **Step 3: Cross-check tests**

```cpp
static void test_simd_batched_l1_linf_match_scalar()
{
    const std::size_t dim = 8, n = 32;
    std::vector<double> centers(n * dim);
    std::mt19937 rng(2);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    for (auto &x : centers) x = u(rng);
    vec_t q(dim);
    for (auto &x : q) x = u(rng);

    auto approx = [](double x, double y) { return std::abs(x - y) < 1e-9; };
    vec_t row(dim);

    auto sl1 = metric::detail::batched_distance(
        metric::manhattan{}, q,
        std::span<const double>(centers.data(), centers.size()), dim, n);
    metric::manhattan m1{};
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < dim; ++j) row[j] = centers[i * dim + j];
        assert(approx(sl1[i], m1(q, row)));
    }

    auto sli = metric::detail::batched_distance(
        metric::chebyshev{}, q,
        std::span<const double>(centers.data(), centers.size()), dim, n);
    metric::chebyshev mc{};
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < dim; ++j) row[j] = centers[i * dim + j];
        assert(approx(sli[i], mc(q, row)));
    }
    std::cout << "  test_simd_batched_l1_linf_match_scalar: OK\n";
}
```

Register in `main()`.

- [ ] **Step 4: Build and run**

```sh
cd tests && make clean && make && ./test_smoke
```

Expected: all pass.

- [ ] **Step 5: Commit**

```sh
git add include/listofclusters/detail/batched_distance.hh include/listofclusters/metrics.hh tests/test_smoke.cc
git commit -m "phase 2.3: SIMD batched manhattan + chebyshev kernels"
```

---

### Task 2.4: Bench gate (P1 + P2 combined)

- [ ] **Step 1: Bench at D=8 and D=32 Euclidean**

```sh
cd bench
make clean && make BENCH_FLAGS="-DBENCH_D=8"  && ./bench_main 10000 8 200 uniform | tee bench_d8.txt
make clean && make BENCH_FLAGS="-DBENCH_D=32" && ./bench_main 10000 32 200 uniform | tee bench_d32.txt
```

- [ ] **Step 2: Verify gates**

From spec §10.2:
- P2 at D=8: ≥ 2× vs current LC 1T
- P2 at D=32: ≥ 3× vs current LC 1T

Compute ratios from the saved bench output and from the pre-P1 baseline cached in `bench/README.md`.

If gates miss: profile, look first at whether `batched_distance` is being inlined and the SIMD intrinsics are actually firing (check disassembly: `objdump -d bench_main | grep -A 5 batched_distance_euclidean`). On Apple Silicon you should see `fmla` instructions; on x86_64 with AVX2 you should see `vfmadd231pd`.

Common fixes:
- `-mfma -mavx2` not enabled on x86_64 → add to CXXFLAGS for x86 builds (NEON is unconditional on arm64 GCC/Clang).
- Compiler isn't inlining the impl across the function-template boundary → mark impls `__attribute__((always_inline))` or move to the cpp-translation-unit-friendly form.

If still missing after profile + fix, revert Phase 2 and document.

- [ ] **Step 3: No commit unless code was changed during profiling**

---

## Phase exit check

```sh
cd tests && make clean && make && ./test_smoke
cd ../bench && make clean && make BENCH_FLAGS="-DBENCH_D=8"  && ./bench_main
cd ../bench && make clean && make BENCH_FLAGS="-DBENCH_D=32" && ./bench_main
git log --oneline -5
```
