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
#include <cstdio>
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

// -----------------------------------------------------------------------------
// batched_pairwise_distance — compute the (Q × N) distance matrix between
// `n_queries` queries and `n_points` indexed points in ONE pass, with the
// point row in the OUTER loop and queries in the INNER loop. This makes
// each point row stay hot in cache while it is reused across all Q queries
// — amortizing the corpus memory bandwidth (N × D bytes) over Q queries
// instead of paying it Q times.
//
// Without this restructure, calling batched_distance Q times costs
// Q × (N × D bytes) of memory reads. Per-query batch was memory-bound at
// high D (e.g. D=768) where the corpus far exceeds L2 cache. This kernel
// is what Faiss's FlatL2 essentially does via cache-blocked BLAS GEMM.
// -----------------------------------------------------------------------------

template <class Metric, class Object>
[[nodiscard]] std::vector<double> batched_pairwise_distance(
    const Metric &m,
    std::span<const double> queries_flat,
    std::span<const double> points_flat,
    std::size_t dim,
    std::size_t n_queries,
    std::size_t n_points)
{
    // Generic fallback: build scratch Objects per row and call the metric.
    std::vector<double> out(n_queries * n_points);
    Object q_row{}, p_row{};
    if constexpr (requires { q_row.resize(dim); }) {
        q_row.resize(dim);
        p_row.resize(dim);
    }
    for (std::size_t p = 0; p < n_points; ++p) {
        for (std::size_t j = 0; j < dim; ++j) p_row[j] = points_flat[p * dim + j];
        for (std::size_t q = 0; q < n_queries; ++q) {
            for (std::size_t j = 0; j < dim; ++j) q_row[j] = queries_flat[q * dim + j];
            out[q * n_points + p] = m(q_row, p_row);
        }
    }
    return out;
}

// SIMD overload for Euclidean. Point row in outer loop = cache-resident
// while inner Q queries iterate; query rows streamed sequentially from a
// small Q×D matrix that ideally fits in L1.
[[nodiscard]] inline std::vector<double> batched_pairwise_distance_euclidean(
    std::span<const double> queries_flat,
    std::span<const double> points_flat,
    std::size_t dim,
    std::size_t n_queries,
    std::size_t n_points)
{
    std::vector<double> out(n_queries * n_points);
#if defined(__ARM_NEON)
    // 4 queries × 1 point per inner iteration: load p_row once, use against
    // 4 query rows. This amortizes the p_row loads and packs 4 independent
    // accumulator chains, hiding FMA latency and improving NEON FMA pipe
    // utilization (closer to Apple M1's 4-pipe issue width).
    for (std::size_t p = 0; p < n_points; ++p) {
        const double *p_row = points_flat.data() + p * dim;
        std::size_t q = 0;
        for (; q + 4 <= n_queries; q += 4) {
            const double *q0r = queries_flat.data() + (q + 0) * dim;
            const double *q1r = queries_flat.data() + (q + 1) * dim;
            const double *q2r = queries_flat.data() + (q + 2) * dim;
            const double *q3r = queries_flat.data() + (q + 3) * dim;
            float64x2_t a0 = vdupq_n_f64(0.0), b0 = vdupq_n_f64(0.0);
            float64x2_t a1 = vdupq_n_f64(0.0), b1 = vdupq_n_f64(0.0);
            float64x2_t a2 = vdupq_n_f64(0.0), b2 = vdupq_n_f64(0.0);
            float64x2_t a3 = vdupq_n_f64(0.0), b3 = vdupq_n_f64(0.0);
            std::size_t j = 0;
            for (; j + 4 <= dim; j += 4) {
                float64x2_t pb0 = vld1q_f64(p_row + j);
                float64x2_t pb1 = vld1q_f64(p_row + j + 2);
                float64x2_t d0 = vsubq_f64(vld1q_f64(q0r + j),     pb0);
                float64x2_t d1 = vsubq_f64(vld1q_f64(q0r + j + 2), pb1);
                a0 = vfmaq_f64(a0, d0, d0);
                b0 = vfmaq_f64(b0, d1, d1);
                d0 = vsubq_f64(vld1q_f64(q1r + j),     pb0);
                d1 = vsubq_f64(vld1q_f64(q1r + j + 2), pb1);
                a1 = vfmaq_f64(a1, d0, d0);
                b1 = vfmaq_f64(b1, d1, d1);
                d0 = vsubq_f64(vld1q_f64(q2r + j),     pb0);
                d1 = vsubq_f64(vld1q_f64(q2r + j + 2), pb1);
                a2 = vfmaq_f64(a2, d0, d0);
                b2 = vfmaq_f64(b2, d1, d1);
                d0 = vsubq_f64(vld1q_f64(q3r + j),     pb0);
                d1 = vsubq_f64(vld1q_f64(q3r + j + 2), pb1);
                a3 = vfmaq_f64(a3, d0, d0);
                b3 = vfmaq_f64(b3, d1, d1);
            }
            float64x2_t s0 = vaddq_f64(a0, b0);
            float64x2_t s1 = vaddq_f64(a1, b1);
            float64x2_t s2 = vaddq_f64(a2, b2);
            float64x2_t s3 = vaddq_f64(a3, b3);
            double r0 = vgetq_lane_f64(s0, 0) + vgetq_lane_f64(s0, 1);
            double r1 = vgetq_lane_f64(s1, 0) + vgetq_lane_f64(s1, 1);
            double r2 = vgetq_lane_f64(s2, 0) + vgetq_lane_f64(s2, 1);
            double r3 = vgetq_lane_f64(s3, 0) + vgetq_lane_f64(s3, 1);
            for (std::size_t jj = j; jj < dim; ++jj) {
                double d;
                d = q0r[jj] - p_row[jj]; r0 += d * d;
                d = q1r[jj] - p_row[jj]; r1 += d * d;
                d = q2r[jj] - p_row[jj]; r2 += d * d;
                d = q3r[jj] - p_row[jj]; r3 += d * d;
            }
            out[(q + 0) * n_points + p] = std::sqrt(r0);
            out[(q + 1) * n_points + p] = std::sqrt(r1);
            out[(q + 2) * n_points + p] = std::sqrt(r2);
            out[(q + 3) * n_points + p] = std::sqrt(r3);
        }
        // Tail: queries not aligned to 4. Fall back to the single-q kernel.
        for (; q < n_queries; ++q) {
            const double *q_row = queries_flat.data() + q * dim;
            float64x2_t acc0 = vdupq_n_f64(0.0);
            float64x2_t acc1 = vdupq_n_f64(0.0);
            std::size_t j = 0;
            for (; j + 4 <= dim; j += 4) {
                float64x2_t qa0 = vld1q_f64(q_row + j);
                float64x2_t qa1 = vld1q_f64(q_row + j + 2);
                float64x2_t pb0 = vld1q_f64(p_row + j);
                float64x2_t pb1 = vld1q_f64(p_row + j + 2);
                float64x2_t d0 = vsubq_f64(qa0, pb0);
                float64x2_t d1 = vsubq_f64(qa1, pb1);
                acc0 = vfmaq_f64(acc0, d0, d0);
                acc1 = vfmaq_f64(acc1, d1, d1);
            }
            float64x2_t acc = vaddq_f64(acc0, acc1);
            double s = vgetq_lane_f64(acc, 0) + vgetq_lane_f64(acc, 1);
            for (; j < dim; ++j) {
                const double d = q_row[j] - p_row[j];
                s += d * d;
            }
            out[q * n_points + p] = std::sqrt(s);
        }
    }
#elif defined(__AVX2__)
    for (std::size_t p = 0; p < n_points; ++p) {
        const double *p_row = points_flat.data() + p * dim;
        for (std::size_t q = 0; q < n_queries; ++q) {
            const double *q_row = queries_flat.data() + q * dim;
            __m256d acc = _mm256_setzero_pd();
            std::size_t j = 0;
            for (; j + 4 <= dim; j += 4) {
                __m256d qa = _mm256_loadu_pd(q_row + j);
                __m256d pb = _mm256_loadu_pd(p_row + j);
                __m256d d  = _mm256_sub_pd(qa, pb);
                acc        = _mm256_fmadd_pd(d, d, acc);
            }
            alignas(32) double tail[4];
            _mm256_store_pd(tail, acc);
            double s = tail[0] + tail[1] + tail[2] + tail[3];
            for (; j < dim; ++j) {
                const double d = q_row[j] - p_row[j];
                s += d * d;
            }
            out[q * n_points + p] = std::sqrt(s);
        }
    }
#else
    for (std::size_t p = 0; p < n_points; ++p) {
        const double *p_row = points_flat.data() + p * dim;
        for (std::size_t q = 0; q < n_queries; ++q) {
            const double *q_row = queries_flat.data() + q * dim;
            double s = 0.0;
            for (std::size_t j = 0; j < dim; ++j) {
                const double d = q_row[j] - p_row[j];
                s += d * d;
            }
            out[q * n_points + p] = std::sqrt(s);
        }
    }
#endif
    return out;
}

// -----------------------------------------------------------------------------
// distance_with_threshold — adaptive early-abandoning distance compute.
//
// For metrics where partial accumulation has a monotone lower-bound property,
// abandon the per-dim loop as soon as the partial sum exceeds the threshold.
// Used by the LC bucket walk: each candidate's distance computation adapts to
// the current shrinking-radius / fixed query radius, terminating early when
// the candidate provably cannot be in the result set.
//
// Stays EXACT: returns the true metric value when ≤ threshold, +inf otherwise.
// The caller only inspects the return when it intends to insert into results,
// so the +inf sentinel is fine — it compares as "not in radius".
//
// Reference: this is the classic 'partial distance / early abandoning' trick
// used in time-series 1-NN (e.g. UCR Suite [Rakthanmanon et al. 2012]) and in
// metric-space indexes when the inner metric admits a monotone partial bound
// (Chavez et al. 2001 §3).
// -----------------------------------------------------------------------------

template <class Metric, class Object>
[[nodiscard]] inline double distance_with_threshold(
    const Metric &m,
    const Object &a,
    const Object &b,
    double threshold) noexcept(noexcept(m(a, b)))
{
    if constexpr (std::is_same_v<Metric, euclidean>) {
        const double threshold_sq = threshold * threshold;
        double s = 0.0;
        const std::size_t n = std::size(a);
        for (std::size_t i = 0; i < n; ++i) {
            const double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
            s += d * d;
            if (s > threshold_sq) return std::numeric_limits<double>::infinity();
        }
        return std::sqrt(s);
    } else if constexpr (std::is_same_v<Metric, manhattan>) {
        double s = 0.0;
        const std::size_t n = std::size(a);
        for (std::size_t i = 0; i < n; ++i) {
            s += std::abs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
            if (s > threshold) return std::numeric_limits<double>::infinity();
        }
        return s;
    } else if constexpr (std::is_same_v<Metric, chebyshev>) {
        double mx = 0.0;
        const std::size_t n = std::size(a);
        for (std::size_t i = 0; i < n; ++i) {
            const double d = std::abs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
            if (d > mx) mx = d;
            if (mx > threshold) return std::numeric_limits<double>::infinity();
        }
        return mx;
    } else {
        // Generic metric: no abandon path available. Compute and let the
        // caller compare. Equivalent in cost to a direct m(a, b) call.
        return m(a, b);
    }
}

}  // namespace detail
}  // namespace metric

#endif
