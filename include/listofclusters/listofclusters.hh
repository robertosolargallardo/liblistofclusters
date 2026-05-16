#ifndef _METRIC_LISTOFCLUSTERS_HH_
#define _METRIC_LISTOFCLUSTERS_HH_
#include <listofclusters/glob.hh>
#include <listofclusters/cluster.hh>
#include <listofclusters/resultslist.hh>
#include <listofclusters/internal_object.hh>

namespace metric
{

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

private:
    list_t     _list;
    uint32_t   _cid{0U};
    [[no_unique_address]] distance_t _metric{};

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

    void centroids(std::ostream &os = std::cout) const
    {
        for(const auto& c : this->_list)
            os << c.centroid().id() << '\n';
    }

private:
    void range_search(resultslist_t&, const double&) const;
    void explore(resultslist_t&, const cluster_t&, const double&) const;

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
    // Hoist query object out of the resultslist getter so each iteration
    // does not chase the centroid()/.object() chain.
    const object_t &q = _results.centroid().object();
    const uint32_t qid = _results.centroid().id();

    for(const auto &c : this->_list)
        {
            const internal_object_t &cc = c.centroid();
            const double d = this->_metric(q, cc.object());

            // Query ball intersects cluster ball -> explore bucket
            if((d - _radius) <= c.radius())
                {
                    if(d <= _radius && !cc.ghost() && cc.id() != qid)
                        _results.push(cc.object(), cc.id(), d);

                    for(const auto &o : c.bucket())
                        {
                            // Triangle-inequality prune (uses precomputed
                            // o.distance() = d(centroid, o), no metric call)
                            if((d - _radius) <= o.distance() && (d + _radius) >= o.distance())
                                {
                                    const double md = this->_metric(q, o.object());
                                    if(md <= _radius && o.id() != qid)
                                        _results.push(o.object(), o.id(), md);
                                }
                        }
                }
            if((d + _radius) <= c.radius())
                return;  // query ball totally contained
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
            if((dqc - _radius) <= o.distance() && (dqc + _radius) >= o.distance())
                {
                    const double d = this->_metric(q, o.object());
                    if(d <= _radius && o.id() != qid)
                        _results.push(o.object(), o.id(), d);
                }
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

    // Current pruning radius: the kth-best distance once we've accumulated
    // k results, otherwise infinity.
    double radius = MAX_RADIUS;
    auto refresh_radius = [&]() {
        if (results.size() >= _k)
            radius = std::prev(results.results().end())->distance();
    };

    for (const auto &c : this->_list)
        {
            const internal_object_t &cc = c.centroid();
            const double d = this->_metric(q, cc.object());

            // 1. Centroid candidate.
            if (d < radius && !cc.ghost() && cc.id() != qid)
                {
                    results.push(cc.object(), cc.id(), d);
                    refresh_radius();
                }

            // 2. Explore bucket if the query ball intersects this cluster.
            if ((d - radius) <= c.radius())
                {
                    for (const auto &m : c.bucket())
                        {
                            // Triangle-inequality prune (cheap, no metric call).
                            if ((d - radius) <= m.distance() && (d + radius) >= m.distance())
                                {
                                    const double md = this->_metric(q, m.object());
                                    if (md < radius && m.id() != qid)
                                        {
                                            results.push(m.object(), m.id(), md);
                                            refresh_radius();
                                        }
                                }
                        }
                }

            // 3. Early termination: query ball entirely inside this cluster.
            //    By construction of LC (first cluster wins for overlapping
            //    balls), no later cluster can hold a closer point.
            if ((d + radius) <= c.radius())
                break;
        }

    return results;
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

}  // namespace metric
#endif
