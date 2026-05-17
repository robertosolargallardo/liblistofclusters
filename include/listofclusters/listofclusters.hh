#ifndef _METRIC_LISTOFCLUSTERS_HH_
#define _METRIC_LISTOFCLUSTERS_HH_
#include <listofclusters/glob.hh>
#include <listofclusters/cluster.hh>
#include <listofclusters/resultslist.hh>
#include <listofclusters/internal_object.hh>
#include <listofclusters/detail/batched_distance.hh>
#include <listofclusters/detail/aesa.hh>
#include <numeric>
#include <span>

namespace metric
{

// Forward declared so the LC class can friend it (used by tests + bench to
// inspect private side structures without leaking _debug_* accessors).
template <class O, class M, std::size_t B, std::size_t V> class lc_test_access;

// Fixed-bucket-size List of Clusters metric index
// (Chavez & Navarro, "A compact space decomposition for effective metric
// indexing", Pattern Recognition Letters 26, 2005). The structure is a flat
// list of (center, radius, bucket) triples. `bucket_size` is m* in the
// paper. `overflow` is retained for legacy callers but is unused under the
// canonical algorithm and asserted >= bucket_size.
//
// `distance_t` is any callable satisfying Metric<distance_t, object_t> -
// typically a stateless functor. Stateless metrics cost no storage thanks
// to [[no_unique_address]].
template <class object_t,
          class distance_t,
          size_t bucket_size,
          size_t overflow>
    requires Metric<distance_t, object_t>
class listofclusters
{
public:
    typedef internal_object<object_t> internal_object_t;
    typedef resultslist<object_t>     resultslist_t;
    typedef cluster<object_t>         cluster_t;
    typedef std::vector<cluster_t>    list_t;

    // Side structure: contiguous row-major matrix of cluster centroids.
    // Kept in sync with _list and rebuilt lazily on the first query that
    // needs it (or eagerly via freeze()). Invalidated by insert/remove.
    struct centers_soa_t {
        std::vector<double> data;
        std::size_t         dim   = 0;
        std::size_t         n     = 0;
        bool                stale = true;
    };

    using aesa_table_t = detail::aesa_table_t<object_t>;

private:
    template <class O, class M, std::size_t B, std::size_t V> friend class lc_test_access;

    list_t                 _list;
    uint32_t               _cid{0U};
    [[no_unique_address]] distance_t _metric{};
    mutable centers_soa_t  _centers;
    mutable aesa_table_t   _aesa;

public:
    listofclusters(void) = default;
    listofclusters(const listofclusters&) = default;
    listofclusters(listofclusters&&) noexcept = default;
    listofclusters& operator=(const listofclusters&) = default;
    listofclusters& operator=(listofclusters&&) noexcept = default;
    ~listofclusters(void) = default;

    explicit listofclusters(distance_t _m) : _metric(std::move(_m)) {}

    void insert(const object_t&, const uint32_t&);
    void remove(const object_t&, const uint32_t&);
    void clear(void) noexcept;

    // Sequential batch insert. Convenience wrapper for inserting many
    // (object, id) pairs from a single call. Equivalent to a manual loop
    // of insert(); kept for API symmetry with batch_knn. NOT internally
    // parallelized - concurrent inserts into the same _list would race.
    void insert(const std::vector<object_t> &objs,
                const std::vector<uint32_t> &ids);

    // Sequential batch delete. Convenience wrapper for removing many
    // (object, id) pairs in one call. Same not-parallel constraint as
    // batch insert.
    void remove(const std::vector<object_t> &objs,
                const std::vector<uint32_t> &ids);

    // Canonical static LC build (Chavez & Navarro PRL 2005, §3) with two
    // center-selection strategies:
    //   - first_unassigned: paper-faithful; pick the lowest-index unassigned
    //     point as the next center.
    //   - farthest_first  : Gonzalez FFT [Gonzalez 1985]; pick the unassigned
    //     point with the maximum min-distance to already-chosen centers.
    //     Tighter, more separated clusters. Default.
    enum class build_strategy { first_unassigned, farthest_first };

    void bulk_build(const std::vector<object_t> &objs,
                    const std::vector<uint32_t> &ids,
                    build_strategy strategy = build_strategy::farthest_first);

    // Queries are read-only with respect to the index: they only touch _list
    // and the (stateless) metric. Multiple knn_search/range_search/batch_knn
    // calls on the same index are safe to run concurrently. Concurrent
    // insert/remove with queries IS NOT SAFE - the index is not lock-free.
    [[nodiscard]] resultslist_t knn_search(const object_t&, const uint32_t&, const size_t&) const;
    [[nodiscard]] resultslist_t range_search(const object_t&, const uint32_t&, const double&) const;

    // Parallel batch kNN. Each query in `queries[i]` is searched with id
    // `start_qid + i`. Queries are partitioned across `nthreads` worker
    // threads (default: hardware_concurrency). Returns one resultslist per
    // input query, in input order. Thread setup happens inside the call;
    // for fine-grained batches the caller should reuse threads externally.
    [[nodiscard]] std::vector<resultslist_t>
    batch_knn(const std::vector<object_t> &queries,
              std::uint32_t start_qid,
              std::size_t k,
              unsigned nthreads = 0) const;

    [[nodiscard]] size_t size(void) const noexcept { return _list.size(); }
    [[nodiscard]] bool empty(void) const noexcept { return _list.empty(); }

    // Eagerly build (or refresh) all side structures. Useful for purely-online
    // callers that don't call bulk_build but want to amortize the first-query
    // rebuild cost.
    void freeze();

    // Build (or rebuild) an AESA-lite global anchor table with k_anchors
    // anchors selected via Farthest-First Traversal [G85, BNC03]. Pass 0
    // to disable / free. See detail/aesa.hh for the lower-bound derivation.
    void build_aesa(std::size_t k_anchors);

    void centroids(std::ostream &os = std::cout) const
    {
        for(const auto& c : this->_list)
            os << c.centroid().id() << '\n';
    }

private:
    void range_search(resultslist_t&, const double&) const;
    void explore(resultslist_t&, const cluster_t&, const double&) const;
    void refresh_centers_soa_() const;
    void refresh_aesa_() const;

    static_assert(overflow >= bucket_size,
                  "overflow must be >= bucket_size (legacy invariant)");
};

// ---------------------------------------------------------------------------
// insert
//
// Canonical fixed-bucket-size dynamic insert, Chavez & Navarro 2005 §5.3.
// Walk the list left-to-right. The first cluster whose ball contains the
// element absorbs it; if that pushes the bucket over `bucket_size`, eject
// the bucket member farthest from the centroid (shrinking r_i) and continue
// inserting *that* element in the tail of the list. Reaching the end of the
// list without a fit appends the element as a new cluster.
// ---------------------------------------------------------------------------
template <class object_t, class distance_t, size_t bucket_size, size_t overflow>
    requires Metric<distance_t, object_t>
void listofclusters<object_t,distance_t,bucket_size,overflow>::insert(const object_t &_object, const uint32_t &_id)
{
    this->_centers.stale = true;
    this->_aesa.stale = true;
    object_t obj = _object;
    uint32_t id  = _id;

    for(auto it = this->_list.begin(); it != this->_list.end(); ++it)
        {
            const double d = this->_metric(obj, it->centroid().object());
            if(d <= it->radius() || it->bucket_count() < bucket_size)
                {
                    // Element fits this ball, or the ball still has room
                    // (absorbing the element will simply grow r_i).
                    it->insert(obj, id, d);
                    if(it->bucket_count() > bucket_size)
                        {
                            internal_object_t ejected = it->pop_farthest();
                            obj = ejected.object();
                            id  = ejected.id();
                            continue;
                        }
                    return;
                }
        }

    // No existing ball absorbed the element - it becomes the next center.
    this->_list.emplace_back(this->_cid++, internal_object_t(obj, id));
}

// ---------------------------------------------------------------------------
// remove
// ---------------------------------------------------------------------------
template <class object_t, class distance_t, size_t bucket_size, size_t overflow>
    requires Metric<distance_t, object_t>
void listofclusters<object_t,distance_t,bucket_size,overflow>::remove(const object_t &_object, const uint32_t &_id)
{
    this->_centers.stale = true;
    this->_aesa.stale = true;
    for(auto it = this->_list.begin(); it != this->_list.end(); ++it)
        {
            const double d = this->_metric(_object, it->centroid().object());
            if(d <= it->radius() || it->centroid().id() == _id)
                {
                    it->remove(_id);
                    if(it->empty())
                        this->_list.erase(it);
                    return;
                }
        }
}

template <class object_t, class distance_t, size_t bucket_size, size_t overflow>
    requires Metric<distance_t, object_t>
void listofclusters<object_t,distance_t,bucket_size,overflow>::clear(void) noexcept
{
    this->_list.clear();
    this->_cid = 0U;
    this->_centers = centers_soa_t{};
    this->_aesa = aesa_table_t{};
}

// ---------------------------------------------------------------------------
// insert (batch) - sequential ergonomic wrapper around insert().
// ---------------------------------------------------------------------------
template <class object_t, class distance_t, size_t bucket_size, size_t overflow>
    requires Metric<distance_t, object_t>
void listofclusters<object_t,distance_t,bucket_size,overflow>::insert(
    const std::vector<object_t> &objs,
    const std::vector<uint32_t> &ids)
{
    const std::size_t n = std::min(objs.size(), ids.size());
    this->_list.reserve(this->_list.size() + n / bucket_size + 1);
    for (std::size_t i = 0; i < n; ++i)
        this->insert(objs[i], ids[i]);
}

// ---------------------------------------------------------------------------
// remove (batch) - sequential ergonomic wrapper around remove().
// ---------------------------------------------------------------------------
template <class object_t, class distance_t, size_t bucket_size, size_t overflow>
    requires Metric<distance_t, object_t>
void listofclusters<object_t,distance_t,bucket_size,overflow>::remove(
    const std::vector<object_t> &objs,
    const std::vector<uint32_t> &ids)
{
    const std::size_t n = std::min(objs.size(), ids.size());
    for (std::size_t i = 0; i < n; ++i)
        this->remove(objs[i], ids[i]);
}

// ---------------------------------------------------------------------------
// bulk_build - canonical static LC construction (Chavez & Navarro 2005 §3).
//
// Replaces existing index state. For each cluster:
//   1. Pick a center c from the still-unassigned points (first one, as in
//      the paper's pseudocode).
//   2. Compute d(c, u) for every other unassigned u.
//   3. Take the bucket_size nearest unassigned points as the bucket; the
//      radius is the largest of those distances.
//   4. Mark all selected points (center + bucket) as assigned.
//   5. Repeat with what's left.
//
// Complexity: O((n / bucket_size) * n) distance computations. Slower than
// incremental insert (which is O(n * |list|) but with a smaller list), but
// produces tighter clusters - better pruning at query time.
// ---------------------------------------------------------------------------
template <class object_t, class distance_t, size_t bucket_size, size_t overflow>
    requires Metric<distance_t, object_t>
void listofclusters<object_t,distance_t,bucket_size,overflow>::bulk_build(
    const std::vector<object_t> &objs,
    const std::vector<uint32_t> &ids,
    build_strategy strategy)
{
    this->clear();
    const std::size_t n = std::min(objs.size(), ids.size());
    if (n == 0) return;

    std::vector<char> assigned(n, 0);
    std::size_t remaining = n;
    std::vector<std::pair<double, std::size_t>> scratch;
    scratch.reserve(n);

    // For FFT: min-distance from each (still-unassigned) point to any
    // already-chosen center. Updated incrementally to keep total distance
    // count at O((n/m)*n), matching first_unassigned modulo constants.
    // Reference: Gonzalez 1985, Theoretical Computer Science 38.
    std::vector<double> min_to_center(n, std::numeric_limits<double>::infinity());

    this->_list.reserve(n / bucket_size + 1);

    // Fixed seed for reproducible FFT seeding (test suite + bench A/B).
    std::mt19937 rng(0xC0FFEEu);

    while (remaining > 0)
        {
            // 1. Pick the next center.
            std::size_t c_idx = 0;
            if (this->_list.empty()) {
                // First center: random unassigned (FFT) or first unassigned.
                if (strategy == build_strategy::farthest_first) {
                    std::uniform_int_distribution<std::size_t> pick(0, n - 1);
                    do { c_idx = pick(rng); } while (assigned[c_idx]);
                } else {
                    while (c_idx < n && assigned[c_idx]) ++c_idx;
                }
            } else if (strategy == build_strategy::farthest_first) {
                // FFT: argmax of min_to_center over unassigned points.
                double best = -1.0;
                std::size_t best_idx = 0;
                bool any = false;
                for (std::size_t i = 0; i < n; ++i) {
                    if (assigned[i]) continue;
                    const double d = min_to_center[i];
                    if (!any || d > best) { best = d; best_idx = i; any = true; }
                }
                c_idx = best_idx;
            } else {
                while (c_idx < n && assigned[c_idx]) ++c_idx;
            }
            assigned[c_idx] = 1;
            --remaining;

            if (remaining == 0)
                {
                    // Lone leftover: a cluster with just a centroid.
                    this->_list.emplace_back(this->_cid++,
                                             internal_object_t(objs[c_idx], ids[c_idx]));
                    break;
                }

            // 2. Distances from center to all other unassigned points,
            //    updating min_to_center incrementally for FFT's next pick.
            scratch.clear();
            scratch.reserve(remaining);
            for (std::size_t i = 0; i < n; ++i)
                {
                    if (assigned[i]) continue;
                    const double d = this->_metric(objs[c_idx], objs[i]);
                    if (d < min_to_center[i]) min_to_center[i] = d;
                    scratch.emplace_back(d, i);
                }

            // 3. Bring the bucket_size nearest to the front (don't sort the tail).
            const std::size_t take = std::min<std::size_t>(bucket_size, scratch.size());
            if (take < scratch.size())
                std::nth_element(scratch.begin(), scratch.begin() + take, scratch.end(),
                    [](const auto &a, const auto &b) noexcept { return a.first < b.first; });
            std::sort(scratch.begin(), scratch.begin() + take,
                [](const auto &a, const auto &b) noexcept { return a.first < b.first; });

            // 4. Build the cluster.
            cluster_t cluster(this->_cid++,
                              internal_object_t(objs[c_idx], ids[c_idx]));
            for (std::size_t i = 0; i < take; ++i)
                {
                    const auto &[d, idx] = scratch[i];
                    cluster.insert(objs[idx], ids[idx], d);
                    assigned[idx] = 1;
                }
            remaining -= take;

            this->_list.push_back(std::move(cluster));
        }

    this->refresh_centers_soa_();
}

// ---------------------------------------------------------------------------
// range_search (public) and recursive search of the flat list
// ---------------------------------------------------------------------------
template <class object_t, class distance_t, size_t bucket_size, size_t overflow>
    requires Metric<distance_t, object_t>
typename listofclusters<object_t,distance_t,bucket_size,overflow>::resultslist_t
listofclusters<object_t,distance_t,bucket_size,overflow>::range_search(const object_t &_object, const uint32_t &_id, const double &_radius) const
{
    resultslist_t results(internal_object_t(_object, _id));
    this->range_search(results, _radius);
    return results;
}

template <class object_t, class distance_t, size_t bucket_size, size_t overflow>
    requires Metric<distance_t, object_t>
void listofclusters<object_t,distance_t,bucket_size,overflow>::range_search(resultslist_t &_results, const double &_radius) const
{
    const object_t &q = _results.centroid().object();
    const uint32_t qid = _results.centroid().id();
    if (this->_list.empty()) return;

    std::vector<double> dqa;
    if (this->_aesa.k_anchors > 0) {
        if (this->_aesa.stale) this->refresh_aesa_();
        dqa.resize(this->_aesa.k_anchors);
        for (std::size_t i = 0; i < this->_aesa.k_anchors; ++i)
            dqa[i] = this->_metric(q, this->_aesa.anchors[i].object());
    }

    const std::size_t k_anchors = this->_aesa.k_anchors;
    const double *aesa_dists = k_anchors ? this->_aesa.dists.data() : nullptr;

    auto process_cluster = [&](std::size_t ci, const cluster_t &c, double d) -> bool {
        const internal_object_t &cc = c.centroid();
        if((d - _radius) <= c.radius()) {
            if(d <= _radius && !cc.ghost() && cc.id() != qid)
                _results.push(cc.object(), cc.id(), d);
            const std::size_t base = k_anchors
                ? static_cast<std::size_t>(this->_aesa.cluster_centroid_row[ci])
                : 0;
            std::size_t bi = 0;
            for(const auto &o : c.bucket()) {
                if((d - _radius) > o.distance() || (d + _radius) < o.distance()) {
                    ++bi; continue;
                }
                if (k_anchors) {
                    const double *row = aesa_dists + (base + 1 + bi) * k_anchors;
                    double lb = 0.0;
                    for (std::size_t i = 0; i < k_anchors; ++i) {
                        const double diff = std::abs(dqa[i] - row[i]);
                        if (diff > lb) lb = diff;
                    }
                    if (lb > _radius) { ++bi; continue; }
                }
                const double md = this->_metric(q, o.object());
                if(md <= _radius && o.id() != qid)
                    _results.push(o.object(), o.id(), md);
                ++bi;
            }
        }
        // LC build-order early-termination invariant (Chavez & Navarro 2005 §4).
        return ((d + _radius) <= c.radius());
    };

    if constexpr (supports_batched_distance_v<distance_t, object_t>) {
        if (this->_centers.stale) this->refresh_centers_soa_();
        auto d_centers = detail::batched_distance(
            this->_metric, q,
            std::span<const double>(this->_centers.data.data(), this->_centers.data.size()),
            this->_centers.dim, this->_centers.n);
        for (std::size_t i = 0; i < this->_centers.n; ++i) {
            if (process_cluster(i, this->_list[i], d_centers[i])) return;
        }
        return;
    }

    const std::size_t n = this->_list.size();
    for (std::size_t i = 0; i < n; ++i) {
        const auto &c = this->_list[i];
        const double d = this->_metric(q, c.centroid().object());
        if (process_cluster(i, c, d)) return;
    }
}

// explore() retained for ABI compatibility; thin wrapper around the
// inlined logic in range_search above.
template <class object_t, class distance_t, size_t bucket_size, size_t overflow>
    requires Metric<distance_t, object_t>
void listofclusters<object_t,distance_t,bucket_size,overflow>::explore(resultslist_t &_results, const cluster_t &_cluster, const double &_radius) const
{
    const object_t &q = _results.centroid().object();
    const uint32_t qid = _results.centroid().id();
    const internal_object_t &cc = _cluster.centroid();
    const double dqc = this->_metric(q, cc.object());

    if(dqc <= _radius && !cc.ghost() && cc.id() != qid)
        _results.push(cc.object(), cc.id(), dqc);

    for(const auto &o : _cluster.bucket())
        {
            if((dqc - _radius) > o.distance() || (dqc + _radius) < o.distance())
                continue;
            const double d = this->_metric(q, o.object());
            if(d <= _radius && o.id() != qid)
                _results.push(o.object(), o.id(), d);
        }
}

// ---------------------------------------------------------------------------
// knn_search - single-pass shrinking-radius algorithm
//
// The previous implementation called range_search in an expanding-radius
// retry loop until k results accumulated. Each retry was a full re-traversal
// of the cluster list, so a worst-case query revisited every cluster 2-3
// times. The single-pass version below walks the list once: it maintains
// top-k results, uses the kth-best distance as a shrinking radius, and
// prunes clusters and bucket members via triangle inequality as it goes.
// ---------------------------------------------------------------------------
template <class object_t, class distance_t, size_t bucket_size, size_t overflow>
    requires Metric<distance_t, object_t>
typename listofclusters<object_t,distance_t,bucket_size,overflow>::resultslist_t
listofclusters<object_t,distance_t,bucket_size,overflow>::knn_search(const object_t &_object, const uint32_t &_id, const size_t &_k) const
{
    resultslist_t results(internal_object_t(_object, _id), _k);
    const object_t &q = _object;
    const uint32_t qid = _id;

    if (this->_list.empty()) return results;

    // Current pruning radius: the kth-best distance once we've accumulated
    // k results, otherwise infinity.
    double radius = MAX_RADIUS;
    auto refresh_radius = [&]() {
        if (results.size() >= _k)
            radius = std::prev(results.results().end())->distance();
    };

    // Pre-compute d(q, anchor_i) once per query if AESA is enabled.
    std::vector<double> dqa;
    if (this->_aesa.k_anchors > 0) {
        if (this->_aesa.stale) this->refresh_aesa_();
        dqa.resize(this->_aesa.k_anchors);
        for (std::size_t i = 0; i < this->_aesa.k_anchors; ++i)
            dqa[i] = this->_metric(q, this->_aesa.anchors[i].object());
    }

    const std::size_t k_anchors = this->_aesa.k_anchors;
    const double *aesa_dists = k_anchors ? this->_aesa.dists.data() : nullptr;

    // Per-cluster walk, sharing the precomputed d(q, centroid_i) passed in.
    // ci is the cluster index, used to address into the AESA dists table
    // via cluster_centroid_row[ci] — no hashmap lookup on the hot path.
    auto process_cluster = [&](std::size_t ci, const cluster_t &c, double d) -> bool {
        const internal_object_t &cc = c.centroid();
        if (d < radius && !cc.ghost() && cc.id() != qid) {
            results.push(cc.object(), cc.id(), d);
            refresh_radius();
        }
        if ((d - radius) <= c.radius()) {
            const std::size_t base = k_anchors
                ? static_cast<std::size_t>(this->_aesa.cluster_centroid_row[ci])
                : 0;
            std::size_t bi = 0;
            for (const auto &m : c.bucket()) {
                if ((d - radius) > m.distance() || (d + radius) < m.distance()) {
                    ++bi; continue;
                }
                if (k_anchors) {
                    const double *row = aesa_dists + (base + 1 + bi) * k_anchors;
                    double lb = 0.0;
                    for (std::size_t i = 0; i < k_anchors; ++i) {
                        const double diff = std::abs(dqa[i] - row[i]);
                        if (diff > lb) lb = diff;
                    }
                    if (lb >= radius) { ++bi; continue; }
                }
                const double md = this->_metric(q, m.object());
                if (md < radius && m.id() != qid) {
                    results.push(m.object(), m.id(), md);
                    refresh_radius();
                }
                ++bi;
            }
        }
        return ((d + radius) <= c.radius());
    };

    // Phase 1 (P1): compute d(q, all_centers) in one batched pass and walk
    // in insertion order. The batched call wins through contiguous-memory
    // access (and SIMD overloads in Phase 2). Walk order is INSERTION-ORDER,
    // not nearest-first, because LC's early-termination invariant
    // ((d + radius) <= c.radius() ⇒ no later cluster contains qualifying
    // points) is build-order-dependent (Chavez & Navarro 2005 §4) and breaks
    // under arbitrary cluster reordering.
    if constexpr (supports_batched_distance_v<distance_t, object_t>) {
        if (this->_centers.stale) this->refresh_centers_soa_();
        auto d_centers = detail::batched_distance(
            this->_metric, q,
            std::span<const double>(this->_centers.data.data(), this->_centers.data.size()),
            this->_centers.dim, this->_centers.n);

        for (std::size_t i = 0; i < this->_centers.n; ++i) {
            if (process_cluster(i, this->_list[i], d_centers[i])) break;
        }
        return results;
    } else {
        // Scalar fallback for non-batched metrics (Levenshtein, custom):
        // same shape, the metric functor is called per-cluster.
        const std::size_t n = this->_list.size();
        for (std::size_t i = 0; i < n; ++i) {
            const auto &c = this->_list[i];
            const double d = this->_metric(q, c.centroid().object());
            if (process_cluster(i, c, d)) break;
        }
        return results;
    }
}

// ---------------------------------------------------------------------------
// batch_knn - parallel kNN across multiple queries
//
// The index is read-only during queries (no member state mutates), so we
// simply partition the query batch across worker threads. Each worker
// calls knn_search for its assigned queries; results are written to a
// pre-sized output vector at the input index. No synchronization needed
// between workers because each thread writes to a disjoint slice.
// ---------------------------------------------------------------------------
template <class object_t, class distance_t, size_t bucket_size, size_t overflow>
    requires Metric<distance_t, object_t>
std::vector<typename listofclusters<object_t,distance_t,bucket_size,overflow>::resultslist_t>
listofclusters<object_t,distance_t,bucket_size,overflow>::batch_knn(
    const std::vector<object_t> &queries,
    std::uint32_t start_qid,
    std::size_t k,
    unsigned nthreads) const
{
    std::vector<resultslist_t> results(queries.size());
    if (queries.empty()) return results;

    if (nthreads == 0)
        nthreads = std::max(1u, std::thread::hardware_concurrency());
    if (nthreads > queries.size())
        nthreads = static_cast<unsigned>(queries.size());

    // Fast path: a single-threaded batch avoids std::thread overhead and
    // is equivalent to a manual loop of knn_search calls.
    if (nthreads == 1) {
        for (std::size_t i = 0; i < queries.size(); ++i)
            results[i] = this->knn_search(queries[i], start_qid + static_cast<std::uint32_t>(i), k);
        return results;
    }

    const std::size_t Q = queries.size();
    const std::size_t chunk = (Q + nthreads - 1) / nthreads;
    std::vector<std::thread> ts;
    ts.reserve(nthreads);
    for (unsigned t = 0; t < nthreads; ++t) {
        const std::size_t q0 = t * chunk;
        const std::size_t q1 = std::min(q0 + chunk, Q);
        if (q0 >= q1) break;
        ts.emplace_back([this, &queries, &results, start_qid, k, q0, q1]() {
            for (std::size_t i = q0; i < q1; ++i)
                results[i] = this->knn_search(queries[i], start_qid + static_cast<std::uint32_t>(i), k);
        });
    }
    for (auto &t : ts) t.join();
    return results;
}

// ---------------------------------------------------------------------------
// refresh_centers_soa_ — rebuild flat SoA of cluster centroids from _list.
// Called by bulk_build, and lazily by query paths when _centers.stale is true.
// ---------------------------------------------------------------------------
template <class object_t, class distance_t, size_t bucket_size, size_t overflow>
    requires Metric<distance_t, object_t>
void listofclusters<object_t,distance_t,bucket_size,overflow>::refresh_centers_soa_() const
{
    const std::size_t n = this->_list.size();
    _centers.n = n;
    if (n == 0) {
        _centers.data.clear();
        _centers.dim = 0;
        _centers.stale = false;
        return;
    }
    const auto &first = this->_list[0].centroid().object();
    const std::size_t dim = std::size(first);
    _centers.dim = dim;
    _centers.data.assign(n * dim, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        const auto &obj = this->_list[i].centroid().object();
        for (std::size_t j = 0; j < dim; ++j)
            _centers.data[i * dim + j] = static_cast<double>(obj[j]);
    }
    _centers.stale = false;
}

template <class object_t, class distance_t, size_t bucket_size, size_t overflow>
    requires Metric<distance_t, object_t>
void listofclusters<object_t,distance_t,bucket_size,overflow>::freeze()
{
    this->refresh_centers_soa_();
    if (this->_aesa.k_anchors > 0)
        this->refresh_aesa_();
}

// ---------------------------------------------------------------------------
// build_aesa / refresh_aesa_ — global anchor table (Mico-Oncina-Vidal 1994
// LAESA, with FFT anchor selection per Gonzalez 1985 + Bustos-Navarro-Chavez
// 2003).
// ---------------------------------------------------------------------------
template <class object_t, class distance_t, size_t bucket_size, size_t overflow>
    requires Metric<distance_t, object_t>
void listofclusters<object_t,distance_t,bucket_size,overflow>::build_aesa(std::size_t k_anchors)
{
    this->_aesa = aesa_table_t{};
    this->_aesa.k_anchors = k_anchors;
    if (k_anchors == 0 || this->_list.empty()) {
        this->_aesa.stale = false;
        return;
    }
    this->refresh_aesa_();
}

template <class object_t, class distance_t, size_t bucket_size, size_t overflow>
    requires Metric<distance_t, object_t>
void listofclusters<object_t,distance_t,bucket_size,overflow>::refresh_aesa_() const
{
    if (this->_aesa.k_anchors == 0) {
        this->_aesa.stale = false;
        return;
    }

    // Flatten all indexed (non-ghost) points in canonical query traversal order:
    //   cluster 0 centroid, cluster 0 bucket[0..], cluster 1 centroid, ...
    // cluster_centroid_row[i] is the row offset of cluster i's centroid.
    struct point_ref { const object_t* obj; };
    std::vector<point_ref> pts;
    this->_aesa.cluster_centroid_row.assign(this->_list.size(), 0u);
    for (std::size_t ci = 0; ci < this->_list.size(); ++ci) {
        const auto &c = this->_list[ci];
        const auto &cc = c.centroid();
        this->_aesa.cluster_centroid_row[ci] = static_cast<std::uint32_t>(pts.size());
        // Centroid row is always present (even for ghost — keeps row indexing
        // aligned; ghost centroids just consume one row's worth of distances).
        pts.push_back({ &cc.object() });
        for (const auto &m : c.bucket()) pts.push_back({ &m.object() });
    }
    const std::size_t N = pts.size();
    const std::size_t k = this->_aesa.k_anchors;
    if (N == 0) { this->_aesa.stale = false; return; }

    // 1. Anchor selection via Farthest-First Traversal (Gonzalez 1985).
    std::mt19937 rng(0xA5A5A5A5u);
    std::uniform_int_distribution<std::size_t> pick(0, N - 1);
    const std::size_t first = pick(rng);

    this->_aesa.anchors.clear();
    this->_aesa.anchors.reserve(k);
    this->_aesa.anchors.emplace_back(*pts[first].obj, 0u);

    std::vector<double> min_to_anchor(N, std::numeric_limits<double>::infinity());
    for (std::size_t i = 0; i < N; ++i)
        min_to_anchor[i] = this->_metric(*pts[i].obj, *pts[first].obj);

    while (this->_aesa.anchors.size() < k && this->_aesa.anchors.size() < N) {
        std::size_t best = 0;
        double best_d = -1.0;
        for (std::size_t i = 0; i < N; ++i)
            if (min_to_anchor[i] > best_d) { best_d = min_to_anchor[i]; best = i; }
        const auto &new_anchor = *pts[best].obj;
        this->_aesa.anchors.emplace_back(new_anchor, 0u);
        for (std::size_t i = 0; i < N; ++i) {
            const double d = this->_metric(*pts[i].obj, new_anchor);
            if (d < min_to_anchor[i]) min_to_anchor[i] = d;
        }
    }
    const std::size_t actual_k = this->_aesa.anchors.size();
    this->_aesa.k_anchors = actual_k;

    // 2. N × k distance table.
    this->_aesa.dists.assign(N * actual_k, 0.0);
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < actual_k; ++j)
            this->_aesa.dists[i * actual_k + j] =
                this->_metric(*pts[i].obj, this->_aesa.anchors[j].object());
    }
    this->_aesa.stale = false;
}

}  // namespace metric
#endif
