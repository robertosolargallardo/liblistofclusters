#ifndef _METRIC_GLOB_HH_
#define _METRIC_GLOB_HH_
#include <map>
#include <set>
#include <list>
#include <queue>
#include <tuple>
#include <vector>
#include <unordered_map>
#include <limits>
#include <random>
#include <thread>
#include <chrono>
#include <utility>
#include <concepts>
#include <algorithm>
#include <iostream>
#include <type_traits>
#include <cstddef>
#include <cstdint>

namespace metric
{
// Concept Metric<M, O>: callable d(a, b) -> double, taking two `object_t`
// values by const-reference.
//
// SEMANTIC REQUIREMENT (not enforced by the type system):
//
// liblistofclusters is an exact metric-space index. It uses the triangle
// inequality to prune the search, so the distance function MUST be a
// metric in the mathematical sense. For any x, y, z in the object space:
//
//   1. d(x, y) >= 0                          (non-negativity)
//   2. d(x, y) == 0  iff  x == y             (identity of indiscernibles)
//   3. d(x, y) == d(y, x)                    (symmetry)
//   4. d(x, z) <= d(x, y) + d(y, z)          (triangle inequality)
//
// Violating any of these (in particular the triangle inequality) will not
// cause crashes or assertions - the algorithms will quietly return wrong
// results. Quasi-metric distances (e.g. KL divergence, squared Euclidean,
// raw cosine similarity) are NOT valid here. See <listofclusters/metrics.hh>
// for ready-made functors that satisfy these axioms.
template <class M, class O>
concept Metric =
    std::invocable<const M&, const O&, const O&> &&
    std::convertible_to<std::invoke_result_t<const M&, const O&, const O&>, double>;
}  // namespace metric

#define RADIUS_INC_PERC  0.1
#define MAX_RADIUS       std::numeric_limits<double>::max()
#endif
