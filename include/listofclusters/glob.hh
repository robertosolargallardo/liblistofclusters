#ifndef _METRIC_GLOB_HH_
#define _METRIC_GLOB_HH_
#include <map>
#include <set>
#include <list>
#include <queue>
#include <tuple>
#include <vector>
#include <limits>
#include <random>
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
// A Metric is any callable (functor, lambda, or function pointer) taking two
// `object_t` values by const-reference and returning something convertible to
// double. This lets the index inline the distance call and avoids the
// per-call value copies of the old free-function template-parameter form.
template <class M, class O>
concept Metric =
    std::invocable<const M&, const O&, const O&> &&
    std::convertible_to<std::invoke_result_t<const M&, const O&, const O&>, double>;
}  // namespace metric

#define RADIUS_INC_PERC  0.1
#define MAX_RADIUS       std::numeric_limits<double>::max()
#endif
