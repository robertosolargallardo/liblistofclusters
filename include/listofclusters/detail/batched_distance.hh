#ifndef _METRIC_DETAIL_BATCHED_DISTANCE_HH_
#define _METRIC_DETAIL_BATCHED_DISTANCE_HH_

// Batched distance kernels: compute d(q, centers[i]) for i in [0, n) where
// centers are stored in a contiguous row-major matrix (centers_flat,
// dim columns). Default is a scalar fallback that reconstructs each row
// into a scratch Object and calls the metric functor. Per-metric SIMD
// overloads live in this header alongside the scalar fallback and are
// resolved by ADL on the metric type.
//
// References:
//   [JDJ17] Johnson, Douze, Jegou — Billion-scale similarity search with
//           GPUs (Faiss). The FlatL2 SIMD inner loop pattern.
//   [CN05]  Chavez & Navarro — A compact space decomposition for effective
//           metric indexing. The original LC algorithm this kernel
//           accelerates.

#include <listofclusters/glob.hh>
#include <cmath>
#include <span>
#include <vector>

#if defined(__ARM_NEON)
#  include <arm_neon.h>
#endif
#if defined(__AVX2__)
#  include <immintrin.h>
#endif

namespace metric {

struct euclidean;   // forward decls so the per-metric overloads below can
struct manhattan;   // refer to the type without including metrics.hh
struct chebyshev;   // (which itself includes this header at the bottom).

// Trait: true when the metric M (over object O) has a batched-distance
// kernel. Default false; specializations below flip it for euclidean,
// manhattan, chebyshev, minkowski<p>, and euclidean_simd<D>.
template <class M, class O>
struct supports_batched_distance : std::false_type {};

template <class M, class O>
inline constexpr bool supports_batched_distance_v =
    supports_batched_distance<M, O>::value;

namespace detail {

// Default scalar fallback. Reconstruct each row from centers_flat into a
// scratch Object and call the metric. Works for any Metric/Object that
// satisfies the Metric concept; SIMD overloads below preempt this for
// the known numeric metrics.
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

// -----------------------------------------------------------------------------
// SIMD impls — load q lane-wise, walk centers row-by-row, FMA the diffs.
// Pattern follows the standard SIMD reduction template (see [JDJ17] §3,
// existing metric::euclidean_simd<D> from Phase 5.7 of this repo).
// -----------------------------------------------------------------------------

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
        float64x2_t mx = vdupq_n_f64(0.0);
        std::size_t j = 0;
        for (; j + 2 <= dim; j += 2) {
            float64x2_t va = { static_cast<double>(q[j]), static_cast<double>(q[j + 1]) };
            float64x2_t vb = vld1q_f64(row + j);
            mx = vmaxq_f64(mx, vabsq_f64(vsubq_f64(va, vb)));
        }
        double m = std::max(vgetq_lane_f64(mx, 0), vgetq_lane_f64(mx, 1));
        for (; j < dim; ++j) {
            const double d = std::abs(static_cast<double>(q[j]) - row[j]);
            if (d > m) m = d;
        }
        out[i] = m;
    }
#elif defined(__AVX2__)
    const __m256d signmask = _mm256_set1_pd(-0.0);
    for (std::size_t i = 0; i < n; ++i) {
        const double *row = centers_flat.data() + i * dim;
        __m256d mx = _mm256_setzero_pd();
        std::size_t j = 0;
        for (; j + 4 <= dim; j += 4) {
            __m256d va = _mm256_setr_pd(q[j], q[j + 1], q[j + 2], q[j + 3]);
            __m256d vb = _mm256_loadu_pd(row + j);
            __m256d d  = _mm256_andnot_pd(signmask, _mm256_sub_pd(va, vb));
            mx = _mm256_max_pd(mx, d);
        }
        alignas(32) double tail[4];
        _mm256_store_pd(tail, mx);
        double m = std::max({tail[0], tail[1], tail[2], tail[3]});
        for (; j < dim; ++j) {
            const double d = std::abs(static_cast<double>(q[j]) - row[j]);
            if (d > m) m = d;
        }
        out[i] = m;
    }
#else
    for (std::size_t i = 0; i < n; ++i) {
        const double *row = centers_flat.data() + i * dim;
        double m = 0.0;
        for (std::size_t j = 0; j < dim; ++j) {
            const double d = std::abs(static_cast<double>(q[j]) - row[j]);
            if (d > m) m = d;
        }
        out[i] = m;
    }
#endif
    return out;
}

// ADL-routed overloads. metric::{euclidean, manhattan, chebyshev} live in
// metrics.hh; these overloads preempt the generic template.
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
}  // namespace metric

#endif
