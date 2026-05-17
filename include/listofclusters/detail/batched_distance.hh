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
#include <span>
#include <vector>

namespace metric {

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

}  // namespace detail
}  // namespace metric

#endif
