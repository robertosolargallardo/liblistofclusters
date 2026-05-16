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

    [[nodiscard]] resultslist_t knn_search(const object_t&, const uint32_t&, const size_t&);
    [[nodiscard]] resultslist_t range_search(const object_t&, const uint32_t&, const double&);

    [[nodiscard]] size_t size(void) const noexcept { return _list.size(); }
    [[nodiscard]] bool empty(void) const noexcept { return _list.empty(); }

    void centroids(std::ostream &os = std::cout) const
    {
        for(const auto& c : this->_list)
            os << c.centroid().id() << '\n';
    }

private:
    void range_search(resultslist_t&, const double&);
    void explore(resultslist_t&, const cluster_t&, const double&);
    double internal_distance(const internal_object_t&, const internal_object_t&);

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
listofclusters<object_t,distance_t,bucket_size,overflow>::range_search(const object_t &_object, const uint32_t &_id, const double &_radius)
{
    resultslist_t results(internal_object_t(_object, _id));
    this->range_search(results, _radius);
    return results;
}

template <class object_t, class distance_t, size_t bucket_size, size_t overflow>
    requires Metric<distance_t, object_t>
void listofclusters<object_t,distance_t,bucket_size,overflow>::range_search(resultslist_t &_results, const double &_radius)
{
    for(const auto &c : this->_list)
        {
            const double d = this->internal_distance(_results.centroid(), c.centroid());
            if((d - _radius) <= c.radius())
                this->explore(_results, c, _radius);
            if((d + _radius) <= c.radius())
                return;  // query ball totally contained
        }
}

template <class object_t, class distance_t, size_t bucket_size, size_t overflow>
    requires Metric<distance_t, object_t>
void listofclusters<object_t,distance_t,bucket_size,overflow>::explore(resultslist_t &_results, const cluster_t &_cluster, const double &_radius)
{
    const double dqc = this->internal_distance(_results.centroid(), _cluster.centroid());

    if(dqc <= _radius && !_cluster.centroid().ghost() && _cluster.centroid().id() != _results.centroid().id())
        _results.push(_cluster.centroid().object(), _cluster.centroid().id(), dqc);

    for(const auto &o : _cluster.bucket())
        {
            // Triangle-inequality prune at the per-bucket-element level.
            if((dqc - _radius) <= o.distance() && (dqc + _radius) >= o.distance())
                {
                    const double d = this->internal_distance(_results.centroid(), o);
                    if(d <= _radius && _results.centroid().id() != o.id())
                        _results.push(o.object(), o.id(), d);
                }
        }
}

// ---------------------------------------------------------------------------
// knn_search
// ---------------------------------------------------------------------------
template <class object_t, class distance_t, size_t bucket_size, size_t overflow>
    requires Metric<distance_t, object_t>
typename listofclusters<object_t,distance_t,bucket_size,overflow>::resultslist_t
listofclusters<object_t,distance_t,bucket_size,overflow>::knn_search(const object_t &_object, const uint32_t &_id, const size_t &_k)
{
    resultslist_t results(internal_object_t(_object, _id), _k);

    double radius = MAX_RADIUS;
    double min_external_radius = MAX_RADIUS;

    // Initial radius estimate from the cluster geometry.
    for(const auto &c : this->_list)
        {
            const double d    = this->internal_distance(c.centroid(), results.centroid());
            const double diff = c.radius() - d;
            if(diff > 0.0 && diff < radius)
                radius = diff;
            else if((-diff) > 0.0 && (-diff) < min_external_radius)
                min_external_radius = -diff;
        }
    radius = (radius == MAX_RADIUS) ? min_external_radius : radius;

    do
        {
            this->range_search(results, radius);
            if(radius == MAX_RADIUS || radius == 0.0) break;
            radius += radius * RADIUS_INC_PERC;
        }
    while(results.results().size() < _k);

    return results;
}

template <class object_t, class distance_t, size_t bucket_size, size_t overflow>
    requires Metric<distance_t, object_t>
double listofclusters<object_t,distance_t,bucket_size,overflow>::internal_distance(const internal_object_t &_a, const internal_object_t &_b)
{
    // Direct call. Earlier code memoized via a nested
    // std::map<uint32_t, std::map<uint32_t, double>>, but benchmarking
    // showed it had no measurable impact on kNN throughput (each query
    // touches each centroid at most a couple of times, so the map lookup
    // overhead canceled out the savings). Removed in the same commit.
    return this->_metric(_a.object(), _b.object());
}

}  // namespace metric
#endif
