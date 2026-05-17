#ifndef _METRIC_LISTOFCLUSTERS_HH_
#define _METRIC_LISTOFCLUSTERS_HH_
#include <listofclusters/glob.hh>
#include <listofclusters/cluster.hh>
#include <listofclusters/resultslist.hh>
#include <listofclusters/internal_object.hh>
#include <listofclusters/detail/batched_distance.hh>
#include <atomic>
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

    // Side structure: contiguous row-major matrix of ALL indexed points
    // (centroids + bucket members) in canonical query traversal order.
    // cluster_centroid_row[ci] is the row offset of cluster ci's centroid;
    // bucket member j of cluster ci is at cluster_centroid_row[ci] + 1 + j.
    //
    // Lets the entire per-query distance work happen in one batched SIMD
    // pass (Faiss-style FlatL2 inner loop) instead of N_buckets scalar
    // metric calls. The LC build-order traversal still drives correctness
    // (insertion-order walk, triangle-inequality filter, early termination
    // per Chavez & Navarro 2005 §4), only the distance compute is batched.
    //
    // Kept in sync with _list and rebuilt lazily; invalidated by insert/remove.
    struct centers_soa_t {
        std::vector<double>        data;
        // Float32 mirror of `data` used by the BLAS (cblas_sgemm) fast
        // path in batch_knn. Built alongside `data` to amortize the cast.
        // Also stores ||p||² per point for the d²= ||q||²-2q·p+||p||²
        // decomposition.
        std::vector<float>         data_f32;
        std::vector<float>         sq_norms_f32;
        std::vector<std::uint32_t> cluster_centroid_row;

        std::size_t                dim   = 0;
        std::size_t                n     = 0;        // total point count (rows)
        std::size_t                n_clusters = 0;
        bool                       stale = true;
    };

private:
    template <class O, class M, std::size_t B, std::size_t V> friend class lc_test_access;

    list_t                 _list;
    uint32_t               _cid{0U};
    [[no_unique_address]] distance_t _metric{};
    mutable centers_soa_t  _centers;

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

    void centroids(std::ostream &os = std::cout) const
    {
        for(const auto& c : this->_list)
            os << c.centroid().id() << '\n';
    }

private:
    void range_search(resultslist_t&, const double&) const;
    void explore(resultslist_t&, const cluster_t&, const double&) const;
    void refresh_centers_soa_() const;

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

    if constexpr (supports_batched_distance_v<distance_t, object_t>) {
        if (this->_centers.stale) this->refresh_centers_soa_();
        auto d_all = detail::batched_distance(
            this->_metric, q,
            std::span<const double>(this->_centers.data.data(), this->_centers.data.size()),
            this->_centers.dim, this->_centers.n);

        const std::size_t n_clusters = this->_list.size();
        for (std::size_t i = 0; i < n_clusters; ++i) {
            const auto &c = this->_list[i];
            const std::size_t base = this->_centers.cluster_centroid_row[i];
            const double d = d_all[base];
            const internal_object_t &cc = c.centroid();

            if ((d - _radius) <= c.radius()) {
                if (d <= _radius && !cc.ghost() && cc.id() != qid)
                    _results.push(cc.object(), cc.id(), d);
                std::size_t bi = 0;
                for (const auto &o : c.bucket()) {
                    if ((d - _radius) > o.distance() || (d + _radius) < o.distance()) {
                        ++bi; continue;
                    }
                    const double md = d_all[base + 1 + bi];
                    if (md <= _radius && o.id() != qid)
                        _results.push(o.object(), o.id(), md);
                    ++bi;
                }
            }
            // LC build-order early-termination invariant (Chavez & Navarro 2005 §4).
            if ((d + _radius) <= c.radius()) return;
        }
        return;
    }

    // Scalar fallback.
    for (const auto &c : this->_list) {
        const double d = this->_metric(q, c.centroid().object());
        const internal_object_t &cc = c.centroid();
        if ((d - _radius) <= c.radius()) {
            if (d <= _radius && !cc.ghost() && cc.id() != qid)
                _results.push(cc.object(), cc.id(), d);
            for (const auto &o : c.bucket()) {
                if ((d - _radius) > o.distance() || (d + _radius) < o.distance())
                    continue;
                const double md = detail::distance_with_threshold(
                    this->_metric, q, o.object(), _radius);
                if (md <= _radius && o.id() != qid)
                    _results.push(o.object(), o.id(), md);
            }
        }
        if ((d + _radius) <= c.radius()) return;
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

    double radius = MAX_RADIUS;
    auto refresh_radius = [&]() {
        if (results.size() >= _k)
            radius = std::prev(results.results().end())->distance();
    };

    // Batched-distance fast path: one SIMD pass computes d(q, p) for every
    // indexed point in the corpus. The LC machinery (insertion-order walk,
    // triangle-inequality bucket filter, build-order early termination)
    // then runs over the precomputed distance vector — no scalar metric
    // calls in the bucket walk. Stays EXACT; this is exactly what Faiss's
    // FlatL2 does but preserves LC's pruning structure to bypass clusters
    // whose ball doesn't intersect the query.
    if constexpr (supports_batched_distance_v<distance_t, object_t>) {
        if (this->_centers.stale) this->refresh_centers_soa_();
        std::vector<double> d_all;
#if defined(LISTOFCLUSTERS_USE_BLAS)
        // BLAS single-query path: cblas_sgemv hits Apple Accelerate's
        // AMX/NEON at near-peak. For online (one-at-a-time) queries this
        // closes the gap to Faiss FlatIP, which uses the same kernel.
        if constexpr (std::is_same_v<distance_t, metric::euclidean>) {
            const std::size_t N = this->_centers.n;
            const std::size_t D = this->_centers.dim;
            std::vector<float> q_f32(D);
            for (std::size_t j = 0; j < D; ++j) q_f32[j] = static_cast<float>(q[j]);
            float qsq = 0.0f;
            for (std::size_t j = 0; j < D; ++j) qsq += q_f32[j] * q_f32[j];
            std::vector<float> ip(N);
            cblas_sgemv(CblasRowMajor, CblasNoTrans,
                        static_cast<int>(N), static_cast<int>(D),
                        1.0f, this->_centers.data_f32.data(), static_cast<int>(D),
                        q_f32.data(), 1,
                        0.0f, ip.data(), 1);
            d_all.resize(N);
            const float *psq = this->_centers.sq_norms_f32.data();
            for (std::size_t r = 0; r < N; ++r) {
                float d2 = qsq - 2.0f * ip[r] + psq[r];
                if (d2 < 0.0f) d2 = 0.0f;
                d_all[r] = std::sqrt(static_cast<double>(d2));
            }
        } else
#endif
        {
            d_all = detail::batched_distance(
                this->_metric, q,
                std::span<const double>(this->_centers.data.data(), this->_centers.data.size()),
                this->_centers.dim, this->_centers.n);
        }

        const std::size_t n_clusters = this->_list.size();
        for (std::size_t i = 0; i < n_clusters; ++i) {
            const auto &c = this->_list[i];
            const std::size_t base = this->_centers.cluster_centroid_row[i];
            const double d = d_all[base];
            const internal_object_t &cc = c.centroid();

            if (d < radius && !cc.ghost() && cc.id() != qid) {
                results.push(cc.object(), cc.id(), d);
                refresh_radius();
            }
            if ((d - radius) <= c.radius()) {
                std::size_t bi = 0;
                for (const auto &m : c.bucket()) {
                    if ((d - radius) > m.distance() || (d + radius) < m.distance()) {
                        ++bi; continue;
                    }
                    const double md = d_all[base + 1 + bi];
                    if (md < radius && m.id() != qid) {
                        results.push(m.object(), m.id(), md);
                        refresh_radius();
                    }
                    ++bi;
                }
            }
            if ((d + radius) <= c.radius()) break;
        }
        return results;
    } else {
        // Scalar fallback (Levenshtein, custom): adaptive early-abandon
        // bucket walk.
        for (const auto &c : this->_list) {
            const double d = this->_metric(q, c.centroid().object());
            const internal_object_t &cc = c.centroid();
            if (d < radius && !cc.ghost() && cc.id() != qid) {
                results.push(cc.object(), cc.id(), d);
                refresh_radius();
            }
            if ((d - radius) <= c.radius()) {
                for (const auto &m : c.bucket()) {
                    if ((d - radius) > m.distance() || (d + radius) < m.distance())
                        continue;
                    const double md = detail::distance_with_threshold(
                        this->_metric, q, m.object(), radius);
                    if (md < radius && m.id() != qid) {
                        results.push(m.object(), m.id(), md);
                        refresh_radius();
                    }
                }
            }
            if ((d + radius) <= c.radius()) break;
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
    if (queries.empty() || this->_list.empty()) return results;

    if (nthreads == 0)
        nthreads = std::max(1u, std::thread::hardware_concurrency());
    if (nthreads > queries.size())
        nthreads = static_cast<unsigned>(queries.size());


    // Cache-blocked batched pairwise path: when the metric supports batching,
    // compute the (Q × N) distance matrix in a SINGLE point-outer/query-inner
    // pass so the N×D corpus is read once and reused across all Q queries.
    // This is the standard Faiss-FlatL2 trick. Each query then runs an LC
    // walk over its row of precomputed distances — no scalar metric calls.
    if constexpr (supports_batched_distance_v<distance_t, object_t>) {
        if (this->_centers.stale) this->refresh_centers_soa_();
        const std::size_t Q = queries.size();
        const std::size_t N = this->_centers.n;
        const std::size_t D = this->_centers.dim;

        // Pack queries into a contiguous Q×D matrix.
        std::vector<double> queries_flat(Q * D);
        for (std::size_t i = 0; i < Q; ++i) {
            const auto &qv = queries[i];
            for (std::size_t j = 0; j < D; ++j)
                queries_flat[i * D + j] = static_cast<double>(qv[j]);
        }

        // ONE big SIMD pass: Q×N distance matrix. Cost ~ Q*N*D ops, but the
        // N*D-byte corpus is streamed only once instead of Q times.
        std::vector<double> d_qn;

#if defined(LISTOFCLUSTERS_USE_BLAS)
        // BLAS fast path for Euclidean: cblas_sgemm in float32. Apple
        // Accelerate / OpenBLAS / MKL is hand-tuned to peak; ~10× faster
        // than our NEON kernel on M1 (uses the AMX coprocessor).
        // BLAS does its own internal threading, so we skip the manual
        // nthreads partition for this path.
        //
        // At high D the topk path below recomputes via IP-only sgemm
        // (skipping the per-element sqrt + norm-decomp). Skip this full
        // distance build to avoid double-computing the same sgemm.
        if constexpr (std::is_same_v<distance_t, metric::euclidean>) {
            if (D < 128) {
                std::vector<float> q_f32(Q * D);
                for (std::size_t i = 0; i < Q; ++i) {
                    const auto &qv = queries[i];
                    for (std::size_t j = 0; j < D; ++j)
                        q_f32[i * D + j] = static_cast<float>(qv[j]);
                }
                d_qn = detail::batched_pairwise_distance_euclidean_blas(
                    std::span<const float>(q_f32.data(), q_f32.size()),
                    std::span<const float>(this->_centers.data_f32.data(), this->_centers.data_f32.size()),
                    std::span<const float>(this->_centers.sq_norms_f32.data(), this->_centers.sq_norms_f32.size()),
                    D, Q, N);
            }
        } else
#endif
        {
        // Fallback: NEON/AVX2 multi-query kernel parallelized over POINT
        // chunks (each thread handles a slice of rows × all Q queries).
        d_qn.assign(Q * N, 0.0);
        auto run_pairwise = [&](std::size_t p0, std::size_t p1) {
            if constexpr (std::is_same_v<distance_t, metric::euclidean>) {
                auto chunk = detail::batched_pairwise_distance_euclidean(
                    std::span<const double>(queries_flat.data(), queries_flat.size()),
                    std::span<const double>(this->_centers.data.data() + p0 * D, (p1 - p0) * D),
                    D, Q, p1 - p0);
                // chunk is Q × (p1-p0) row-major. Splat back into d_qn.
                for (std::size_t qi = 0; qi < Q; ++qi)
                    std::copy(chunk.data() + qi * (p1 - p0),
                              chunk.data() + (qi + 1) * (p1 - p0),
                              d_qn.data() + qi * N + p0);
            } else {
                auto chunk = detail::batched_pairwise_distance<distance_t, object_t>(
                    this->_metric,
                    std::span<const double>(queries_flat.data(), queries_flat.size()),
                    std::span<const double>(this->_centers.data.data() + p0 * D, (p1 - p0) * D),
                    D, Q, p1 - p0);
                for (std::size_t qi = 0; qi < Q; ++qi)
                    std::copy(chunk.data() + qi * (p1 - p0),
                              chunk.data() + (qi + 1) * (p1 - p0),
                              d_qn.data() + qi * N + p0);
            }
        };
        if (nthreads <= 1) {
            run_pairwise(0, N);
        } else {
            const std::size_t pchunk = (N + nthreads - 1) / nthreads;
            std::vector<std::thread> ts;
            ts.reserve(nthreads);
            for (unsigned t = 0; t < nthreads; ++t) {
                const std::size_t p0 = t * pchunk;
                const std::size_t p1 = std::min(p0 + pchunk, N);
                if (p0 >= p1) break;
                ts.emplace_back([&, p0, p1]() { run_pairwise(p0, p1); });
            }
            for (auto &t : ts) t.join();
        }
        }  // close BLAS / non-BLAS branch

        const std::size_t n_clusters = this->_list.size();

        // At high D, the LC walk's TI filter prunes nothing and the L2
        // norm-decomposition + per-element sqrt over the full Q×N matrix
        // is wasted work. Instead:
        //   1. Compute the Q×N INNER PRODUCT matrix via sgemm (half the
        //      memory of d_qn, no decomposition or sqrt).
        //   2. Rank top-k by IP descending (= L2 distance ascending for
        //      vectors with bounded norms — exact iff ||q||² and ||p||²
        //      are constant; for general vectors we need both terms).
        //   3. For the top-k WINNERS (≤k per query), compute the actual
        //      L2 distance via NEON. Cheap: k=10 elements per query.
        //
        // For general (non-normalized) vectors we use d²=||q||²-2·IP+||p||²
        // for ranking too, since d² ranking = d ranking — but skip sqrt.
        // Saves the ~1 ms per-element sqrt + double-precision write that
        // dominated the post-sgemm step.
        if (D >= 128) {
#if defined(LISTOFCLUSTERS_USE_BLAS)
            // Recompute via IP-only sgemm (much cheaper output materialize).
            std::vector<float> q_f32_topk(Q * D);
            for (std::size_t i = 0; i < Q; ++i) {
                const auto &qv = queries[i];
                for (std::size_t j = 0; j < D; ++j)
                    q_f32_topk[i * D + j] = static_cast<float>(qv[j]);
            }
            std::vector<float> q_sq(Q, 0.0f);
            for (std::size_t i = 0; i < Q; ++i) {
                float s = 0.0f;
                for (std::size_t j = 0; j < D; ++j) {
                    const float v = q_f32_topk[i * D + j]; s += v * v;
                }
                q_sq[i] = s;
            }
            std::vector<float> ip_qn = detail::batched_pairwise_ip_blas(
                std::span<const float>(q_f32_topk.data(), q_f32_topk.size()),
                std::span<const float>(this->_centers.data_f32.data(), this->_centers.data_f32.size()),
                D, Q, N);

            d_qn.clear();
            d_qn.shrink_to_fit();   // release the now-unused L2 matrix

            const float *p_sq = this->_centers.sq_norms_f32.data();
            auto topk_one = [&](std::size_t i) {
                const std::uint32_t qid = start_qid + static_cast<std::uint32_t>(i);
                auto &res = results[i];
                res = resultslist_t(internal_object_t(queries[i], qid), k);
                const float *ip_row = ip_qn.data() + i * N;
                const float qsq_i = q_sq[i];
                // Top-k by SQUARED distance ascending. d²[r] = q_sq - 2*ip + p_sq[r].
                std::vector<std::pair<float, std::uint32_t>> topk;
                topk.reserve(k + 1);
                for (std::uint32_t r = 0; r < N; ++r) {
                    float d2 = qsq_i - 2.0f * ip_row[r] + p_sq[r];
                    if (d2 < 0.0f) d2 = 0.0f;
                    if (topk.size() < k) {
                        topk.emplace_back(d2, r);
                        std::push_heap(topk.begin(), topk.end());
                    } else if (d2 < topk.front().first) {
                        std::pop_heap(topk.begin(), topk.end());
                        topk.back() = {d2, r};
                        std::push_heap(topk.begin(), topk.end());
                    }
                }
                std::sort_heap(topk.begin(), topk.end());
                for (const auto &[d2, r] : topk) {
                    const double d = std::sqrt(static_cast<double>(d2));
                    const auto &rows = this->_centers.cluster_centroid_row;
                    auto it = std::upper_bound(rows.begin(), rows.end(), r);
                    const std::size_t ci = static_cast<std::size_t>(it - rows.begin() - 1);
                    const std::uint32_t base = rows[ci];
                    if (r == base) {
                        const auto &cc = this->_list[ci].centroid();
                        if (!cc.ghost() && cc.id() != qid)
                            res.push(cc.object(), cc.id(), d);
                    } else {
                        const std::size_t bi = r - base - 1;
                        const auto &m = this->_list[ci].bucket()[bi];
                        if (m.id() != qid)
                            res.push(m.object(), m.id(), d);
                    }
                }
            };
#else
            // Non-BLAS fallback: d_qn was filled by the NEON/AVX2 path.
            // Heap top-k over the precomputed distance matrix.
            auto topk_one = [&](std::size_t i) {
                const std::uint32_t qid = start_qid + static_cast<std::uint32_t>(i);
                auto &res = results[i];
                res = resultslist_t(internal_object_t(queries[i], qid), k);
                const double *d_row = d_qn.data() + i * N;
                std::vector<std::pair<double, std::uint32_t>> topk;
                topk.reserve(k + 1);
                for (std::uint32_t r = 0; r < N; ++r) {
                    const double d = d_row[r];
                    if (topk.size() < k) {
                        topk.emplace_back(d, r);
                        std::push_heap(topk.begin(), topk.end());
                    } else if (d < topk.front().first) {
                        std::pop_heap(topk.begin(), topk.end());
                        topk.back() = {d, r};
                        std::push_heap(topk.begin(), topk.end());
                    }
                }
                std::sort_heap(topk.begin(), topk.end());
                for (const auto &[d, r] : topk) {
                    const auto &rows = this->_centers.cluster_centroid_row;
                    auto it = std::upper_bound(rows.begin(), rows.end(), r);
                    const std::size_t ci = static_cast<std::size_t>(it - rows.begin() - 1);
                    const std::uint32_t base = rows[ci];
                    if (r == base) {
                        const auto &cc = this->_list[ci].centroid();
                        if (!cc.ghost() && cc.id() != qid)
                            res.push(cc.object(), cc.id(), d);
                    } else {
                        const std::size_t bi = r - base - 1;
                        const auto &m = this->_list[ci].bucket()[bi];
                        if (m.id() != qid)
                            res.push(m.object(), m.id(), d);
                    }
                }
            };
#endif
            if (nthreads <= 1) {
                for (std::size_t i = 0; i < Q; ++i) topk_one(i);
            } else {
                const std::size_t chunk = (Q + nthreads - 1) / nthreads;
                std::vector<std::thread> ts;
                ts.reserve(nthreads);
                for (unsigned t = 0; t < nthreads; ++t) {
                    const std::size_t q0 = t * chunk;
                    const std::size_t q1 = std::min(q0 + chunk, Q);
                    if (q0 >= q1) break;
                    ts.emplace_back([&, q0, q1]() {
                        for (std::size_t i = q0; i < q1; ++i) topk_one(i);
                    });
                }
                for (auto &t : ts) t.join();
            }
            return results;
        }

        auto walk_one = [&](std::size_t i) {
            const std::uint32_t qid = start_qid + static_cast<std::uint32_t>(i);
            auto &res = results[i];
            res = resultslist_t(internal_object_t(queries[i], qid), k);
            const double *d_row = d_qn.data() + i * N;

            double radius = MAX_RADIUS;
            auto refresh = [&]() {
                if (res.size() >= k)
                    radius = std::prev(res.results().end())->distance();
            };

            for (std::size_t ci = 0; ci < n_clusters; ++ci) {
                const auto &c = this->_list[ci];
                const std::size_t base = this->_centers.cluster_centroid_row[ci];
                const double d = d_row[base];
                const internal_object_t &cc = c.centroid();

                if (d < radius && !cc.ghost() && cc.id() != qid) {
                    res.push(cc.object(), cc.id(), d);
                    refresh();
                }
                if ((d - radius) <= c.radius()) {
                    std::size_t bi = 0;
                    for (const auto &m : c.bucket()) {
                        if ((d - radius) > m.distance() || (d + radius) < m.distance()) {
                            ++bi; continue;
                        }
                        const double md = d_row[base + 1 + bi];
                        if (md < radius && m.id() != qid) {
                            res.push(m.object(), m.id(), md);
                            refresh();
                        }
                        ++bi;
                    }
                }
                if ((d + radius) <= c.radius()) break;
            }
        };

        if (nthreads <= 1) {
            for (std::size_t i = 0; i < Q; ++i) walk_one(i);
        } else {
            const std::size_t chunk = (Q + nthreads - 1) / nthreads;
            std::vector<std::thread> ts;
            ts.reserve(nthreads);
            for (unsigned t = 0; t < nthreads; ++t) {
                const std::size_t q0 = t * chunk;
                const std::size_t q1 = std::min(q0 + chunk, Q);
                if (q0 >= q1) break;
                ts.emplace_back([&, q0, q1]() {
                    for (std::size_t i = q0; i < q1; ++i) walk_one(i);
                });
            }
            for (auto &t : ts) t.join();
        }
        return results;
    }

    // Scalar fallback: original parallel per-query knn_search loop.
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
    const std::size_t n_clusters = this->_list.size();
    _centers.n_clusters = n_clusters;
    _centers.cluster_centroid_row.clear();
    _centers.cluster_centroid_row.reserve(n_clusters);
    if (n_clusters == 0) {
        _centers.data.clear();
        _centers.dim = 0;
        _centers.n = 0;
        _centers.stale = false;
        return;
    }
    const auto &first = this->_list[0].centroid().object();
    const std::size_t dim = std::size(first);
    _centers.dim = dim;

    // First pass: count total rows and record cluster_centroid_row offsets.
    std::size_t row = 0;
    for (const auto &c : this->_list) {
        _centers.cluster_centroid_row.push_back(static_cast<std::uint32_t>(row));
        row += 1 + c.bucket().size();  // centroid + bucket members
    }
    _centers.n = row;
    _centers.data.assign(_centers.n * dim, 0.0);

    // Second pass: copy points in canonical traversal order.
    row = 0;
    for (const auto &c : this->_list) {
        const auto &cobj = c.centroid().object();
        for (std::size_t j = 0; j < dim; ++j)
            _centers.data[row * dim + j] = static_cast<double>(cobj[j]);
        ++row;
        for (const auto &m : c.bucket()) {
            const auto &mobj = m.object();
            for (std::size_t j = 0; j < dim; ++j)
                _centers.data[row * dim + j] = static_cast<double>(mobj[j]);
            ++row;
        }
    }

    // Build the float32 mirror + per-point squared norm for the BLAS path.
    _centers.data_f32.assign(_centers.n * dim, 0.0f);
    _centers.sq_norms_f32.assign(_centers.n, 0.0f);
    for (std::size_t r = 0; r < _centers.n; ++r) {
        float s = 0.0f;
        for (std::size_t j = 0; j < dim; ++j) {
            const float v = static_cast<float>(_centers.data[r * dim + j]);
            _centers.data_f32[r * dim + j] = v;
            s += v * v;
        }
        _centers.sq_norms_f32[r] = s;
    }


    _centers.stale = false;
}

template <class object_t, class distance_t, size_t bucket_size, size_t overflow>
    requires Metric<distance_t, object_t>
void listofclusters<object_t,distance_t,bucket_size,overflow>::freeze()
{
    this->refresh_centers_soa_();
}

}  // namespace metric
#endif
