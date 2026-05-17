#ifndef _METRIC_DETAIL_AESA_HH_
#define _METRIC_DETAIL_AESA_HH_

// AESA-lite global anchor table for triangle-inequality pruning of
// candidates during knn_search / range_search. Inspired by the LAESA
// variant of AESA: store d(point, anchor_i) for k anchors per indexed
// point; at query time compute d(q, anchor_i) once for each anchor and
// derive the per-candidate lower bound
//
//   LB(q, p) = max_i |d(q, a_i) - d(p, a_i)|
//
// Any candidate p with LB > current_radius can be pruned without
// computing the real distance.
//
// References:
//   [V86]   Vidal, E. (1986) — An algorithm for finding nearest neighbours
//           in (approximately) constant average time. (Original AESA.)
//   [MOV94] Mico, Oncina, Vidal (1994) — A new version of AESA with linear
//           preprocessing-time and memory requirements. (LAESA — the k≪N
//           variant we follow.)
//   [BNC03] Bustos, Navarro, Chavez (2003) — Pivot selection techniques
//           for proximity searching in metric spaces. (Justifies FFT
//           anchor selection over random / outlier choices.)

#include <listofclusters/glob.hh>
#include <listofclusters/internal_object.hh>

namespace metric { namespace detail {

template <class object_t>
struct aesa_table_t {
    using internal_object_t = internal_object<object_t>;

    std::size_t                                       k_anchors = 0;
    std::vector<internal_object_t>                    anchors;
    // Row-major N × k_anchors. dists[row * k + i] = d(point_at_row, anchor_i).
    // Row order matches the canonical query-time traversal:
    //   cluster i centroid → cluster i bucket[0..m-1] → cluster i+1 centroid → ...
    // cluster_centroid_row[i] is the dists row of cluster i's centroid; bucket
    // member j of cluster i is at cluster_centroid_row[i] + 1 + j. This avoids
    // a per-candidate hashmap lookup that was the AESA hot-path bottleneck.
    std::vector<double>                               dists;
    std::vector<std::uint32_t>                        cluster_centroid_row;
    bool                                              stale = true;
};

}}  // namespace metric::detail

#endif
