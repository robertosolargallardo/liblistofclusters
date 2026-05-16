#ifndef _METRIC_METRICS_HH_
#define _METRIC_METRICS_HH_

// Built-in metric functors for liblistofclusters.
//
// Every metric here is a stateless functor (zero-size with
// [[no_unique_address]] in the index, so it costs no storage) and provably
// satisfies the four metric axioms documented on `metric::Metric` in
// <listofclusters/glob.hh>:
//   non-negativity, identity, symmetry, triangle inequality.
//
// Each functor is templated on a `container` type modelling a fixed-size,
// indexable sequence of doubles - std::vector<double>, std::array<double,N>,
// std::span<const double>, an Armadillo rowvec, etc. all work. Both args
// MUST have the same size or behavior is undefined.
//
// Quick reference:
//   euclidean      L2,   sqrt(sum (a_i - b_i)^2)
//   manhattan      L1,   sum |a_i - b_i|
//   chebyshev      Linf, max |a_i - b_i|
//   minkowski<p>   Lp,   (sum |a_i - b_i|^p)^(1/p)   p as int template arg
//   hamming        count of positions where a_i != b_i  (works on any
//                   container of comparable elements)
//   angular        arccos(dot(a,b) / (|a| |b|)) - valid metric on the
//                   unit sphere; NOT cosine similarity, NOT cosine
//                   "distance" 1 - cos which is not a metric.
//
// What's deliberately NOT here because they're not metrics:
//   - squared Euclidean (drops the sqrt, breaks triangle inequality)
//   - cosine distance defined as 1 - cosine_sim
//   - KL divergence, Bregman divergences in general (asymmetric)
//   - dot product, anything where larger = closer

#include <listofclusters/glob.hh>
#include <cmath>
#include <cstddef>
#include <iterator>

namespace metric
{

// L2 (Euclidean) distance.
struct euclidean {
    template <class container>
    [[nodiscard]] double operator()(const container &a, const container &b) const noexcept
    {
        double s = 0.0;
        const std::size_t n = std::size(a);
        for (std::size_t i = 0; i < n; ++i) {
            const double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
            s += d * d;
        }
        return std::sqrt(s);
    }
};

// L1 (Manhattan / taxicab) distance.
struct manhattan {
    template <class container>
    [[nodiscard]] double operator()(const container &a, const container &b) const noexcept
    {
        double s = 0.0;
        const std::size_t n = std::size(a);
        for (std::size_t i = 0; i < n; ++i)
            s += std::abs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
        return s;
    }
};

// L-infinity (Chebyshev / chessboard) distance.
struct chebyshev {
    template <class container>
    [[nodiscard]] double operator()(const container &a, const container &b) const noexcept
    {
        double m = 0.0;
        const std::size_t n = std::size(a);
        for (std::size_t i = 0; i < n; ++i) {
            const double d = std::abs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
            if (d > m) m = d;
        }
        return m;
    }
};

// Minkowski Lp distance for compile-time p >= 1. Valid metric for p >= 1.
// p < 1 violates the triangle inequality and is rejected at compile time.
template <int p>
struct minkowski {
    static_assert(p >= 1, "Minkowski distance is a metric only for p >= 1");

    template <class container>
    [[nodiscard]] double operator()(const container &a, const container &b) const noexcept
    {
        double s = 0.0;
        const std::size_t n = std::size(a);
        for (std::size_t i = 0; i < n; ++i) {
            const double d = std::abs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
            s += std::pow(d, static_cast<double>(p));
        }
        return std::pow(s, 1.0 / static_cast<double>(p));
    }
};

// Hamming distance: number of positions where a[i] != b[i].
// Valid metric on any set with equality - works for strings, integer
// vectors, bit vectors, etc.
struct hamming {
    template <class container>
    [[nodiscard]] double operator()(const container &a, const container &b) const noexcept
    {
        std::size_t s = 0;
        const std::size_t n = std::size(a);
        for (std::size_t i = 0; i < n; ++i)
            if (!(a[i] == b[i])) ++s;
        return static_cast<double>(s);
    }
};

// Angular distance on the unit sphere: arccos(dot(a, b) / (|a| |b|)).
// NOTE: this is NOT cosine similarity, NOR cosine "distance" defined as
// 1 - cos. Those are not metrics; angular IS a metric (bounded in [0, pi]).
// Pre-normalize inputs for speed (or use this functor as-is if your inputs
// are not already unit-length).
struct angular {
    template <class container>
    [[nodiscard]] double operator()(const container &a, const container &b) const noexcept
    {
        double dot = 0.0, na = 0.0, nb = 0.0;
        const std::size_t n = std::size(a);
        for (std::size_t i = 0; i < n; ++i) {
            const double ai = static_cast<double>(a[i]);
            const double bi = static_cast<double>(b[i]);
            dot += ai * bi;
            na  += ai * ai;
            nb  += bi * bi;
        }
        const double denom = std::sqrt(na) * std::sqrt(nb);
        if (denom == 0.0) return 0.0;  // convention: zero-length vectors are colinear
        double cos = dot / denom;
        // Numerical safety: clamp to [-1, 1] before acos.
        if (cos > 1.0) cos = 1.0;
        else if (cos < -1.0) cos = -1.0;
        return std::acos(cos);
    }
};

}  // namespace metric
#endif
