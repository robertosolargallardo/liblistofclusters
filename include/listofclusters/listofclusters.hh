#ifndef _METRIC_LISTOFCLUSTERS_HH_
#define _METRIC_LISTOFCLUSTERS_HH_
#include <listofclusters/glob.hh>
#include <listofclusters/cluster.hh>
#include <listofclusters/resultslist.hh>
#include <listofclusters/internal_object.hh>

namespace metric
{

// Fixed-bucket-size variant of the List of Clusters metric index
// (Chavez & Navarro, "A compact space decomposition for effective metric
// indexing", Pattern Recognition Letters 26, 2005). The structure is a flat
// list of (center, radius, bucket) triples. The template parameter
// `bucket_size` is m* in the paper (the target bucket size for fixed-bucket-
// size construction). `overflow` was used by an earlier incremental scheme
// (now retired); it is kept in the signature so existing callers don't have
// to change but is currently unused.
template <class object_t,double (*distance)(object_t,object_t),size_t bucket_size,size_t overflow>
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
    std::map<uint32_t,std::map<uint32_t,double>> _dcache;

public:
    listofclusters(void) = default;
    listofclusters(const listofclusters&) = default;
    listofclusters(listofclusters&&) noexcept = default;
    listofclusters& operator=(const listofclusters&) = default;
    listofclusters& operator=(listofclusters&&) noexcept = default;
    ~listofclusters(void) = default;

    void insert(const object_t&,const uint32_t&);
    void remove(const object_t&,const uint32_t&);
    void clear(void);

    resultslist_t knn_search(const object_t&,const uint32_t&,const size_t&);
    resultslist_t range_search(const object_t&,const uint32_t&,const double&);

    void centroids(void) const
    {
        for(const auto& cluster : this->_list)
            std::cout << cluster.centroid().id() << std::endl;
    }

private:
    void range_search(resultslist_t&,const double&);
    void explore(resultslist_t&,const cluster_t&,const double&);
    double internal_distance(const internal_object_t&,const internal_object_t&);

    // Suppress -Wunused-template-parameter for `overflow` (kept for API
    // stability across the C-5/C-6 algorithmic fix).
    static_assert(overflow >= bucket_size, "overflow must be >= bucket_size (legacy invariant)");
};

// ---------------------------------------------------------------------------
// insert
//
// Iterative implementation of the LC fixed-bucket-size dynamic insert from
// section 5.3 of the paper:
//
//   "When inserting an element, as soon as we find its appropriate ball i,
//    the bucket will overflow. Hence we take the element of the bucket
//    which is farthest from the center c_i, remove it from the bucket
//    (modifying r_i accordingly), and continue the insertion process in
//    the tail of the list with the new element."
//
// Termination is guaranteed: each iteration either places an element (early
// return), or advances past a cluster (++it). If we reach _list.end() the
// element becomes a new cluster appended to the tail.
// ---------------------------------------------------------------------------
template <class object_t,double (*distance)(object_t,object_t),size_t bucket_size,size_t overflow>
void listofclusters<object_t,distance,bucket_size,overflow>::insert(const object_t &_object,const uint32_t &_id)
{
    object_t obj = _object;
    uint32_t id  = _id;

    for(auto it = this->_list.begin(); it != this->_list.end(); ++it)
        {
            const double d = distance(obj, it->centroid().object());
            if(d <= it->radius() || it->bucket_count() < bucket_size)
                {
                    // Element fits this ball, or the ball still has room (in
                    // which case absorbing the element will simply grow r_i).
                    it->insert(obj, id, d);
                    if(it->bucket_count() > bucket_size)
                        {
                            // Bucket overflows: eject the farthest member and
                            // continue inserting it in the tail of the list.
                            internal_object_t ejected = it->pop_farthest();
                            obj = ejected.object();
                            id  = ejected.id();
                            continue;
                        }
                    return;
                }
        }

    // No existing ball absorbed the element - it becomes the next center.
    cluster_t fresh(this->_cid++, internal_object_t(obj, id));
    this->_list.push_back(std::move(fresh));
}

// ---------------------------------------------------------------------------
// remove
//
// Find the cluster whose ball contains _object and remove _id from it. If
// the cluster becomes empty (centroid ghosted, bucket empty), erase the
// cluster from the list to keep traversals tight.
// ---------------------------------------------------------------------------
template <class object_t,double (*distance)(object_t,object_t),size_t bucket_size,size_t overflow>
void listofclusters<object_t,distance,bucket_size,overflow>::remove(const object_t &_object,const uint32_t &_id)
{
    for(auto it = this->_list.begin(); it != this->_list.end(); ++it)
        {
            const double d = distance(_object, it->centroid().object());
            if(d <= it->radius() || it->centroid().id() == _id)
                {
                    it->remove(_id);
                    if(it->empty())
                        this->_list.erase(it);
                    return;
                }
        }
}

template <class object_t,double (*distance)(object_t,object_t),size_t bucket_size,size_t overflow>
void listofclusters<object_t,distance,bucket_size,overflow>::clear(void)
{
    this->_list.clear();
    this->_dcache.clear();
    this->_cid = 0U;
}

// ---------------------------------------------------------------------------
// range_search (public)
//
// Build a resultslist around the query and run the recursive Search of
// Figure 2 of the paper. The "preempt when query ball is totally contained"
// optimization (`return` instead of `continue`) is in the inner overload.
// ---------------------------------------------------------------------------
template <class object_t,double (*distance)(object_t,object_t),size_t bucket_size,size_t overflow>
typename listofclusters<object_t,distance,bucket_size,overflow>::resultslist_t
listofclusters<object_t,distance,bucket_size,overflow>::range_search(const object_t &_object,const uint32_t &_id,const double &_radius)
{
    resultslist_t results(internal_object_t(_object, _id));
    this->range_search(results, _radius);
    this->_dcache.erase(_id);
    return results;
}

template <class object_t,double (*distance)(object_t,object_t),size_t bucket_size,size_t overflow>
void listofclusters<object_t,distance,bucket_size,overflow>::range_search(resultslist_t &_results,const double &_radius)
{
    for(auto &cluster : this->_list)
        {
            const double d = this->internal_distance(_results.centroid(), cluster.centroid());
            if((d - _radius) <= cluster.radius())
                this->explore(_results, cluster, _radius);
            if((d + _radius) <= cluster.radius())
                return;  // query ball totally contained - no need to look further
        }
}

template <class object_t,double (*distance)(object_t,object_t),size_t bucket_size,size_t overflow>
void listofclusters<object_t,distance,bucket_size,overflow>::explore(resultslist_t &_results,const cluster_t &_cluster,const double &_radius)
{
    const double dqc = this->internal_distance(_results.centroid(), _cluster.centroid());

    if(dqc <= _radius && !_cluster.centroid().ghost() && _cluster.centroid().id() != _results.centroid().id())
        _results.push(_cluster.centroid().object(), _cluster.centroid().id(), dqc);

    for(const auto &object : _cluster.bucket())
        {
            // Triangle-inequality prune at the per-bucket-element level.
            if((dqc - _radius) <= object.distance() && (dqc + _radius) >= object.distance())
                {
                    const double d = this->internal_distance(_results.centroid(), object);
                    if(d <= _radius && _results.centroid().id() != object.id())
                        _results.push(object.object(), object.id(), d);
                }
        }
}

// ---------------------------------------------------------------------------
// knn_search
//
// k-NN over LC: start with a radius derived from the cluster geometry,
// repeat range_search with an expanding radius until we have at least k
// candidates. Each range_search reuses cached distances via _dcache.
// ---------------------------------------------------------------------------
template <class object_t,double (*distance)(object_t,object_t),size_t bucket_size,size_t overflow>
typename listofclusters<object_t,distance,bucket_size,overflow>::resultslist_t
listofclusters<object_t,distance,bucket_size,overflow>::knn_search(const object_t &_object,const uint32_t &_id,const size_t &_k)
{
    resultslist_t results(internal_object_t(_object, _id), _k);

    double radius = MAX_RADIUS;
    double min_external_radius = MAX_RADIUS;

    // Initial radius estimate: smallest "margin" of any cluster around the
    // query. If the query is inside a cluster (diff>0), use the smallest
    // such inside-margin; otherwise use the smallest outside-margin.
    for(auto &cluster : this->_list)
        {
            const double d    = this->internal_distance(cluster.centroid(), results.centroid());
            const double diff = cluster.radius() - d;
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

    this->_dcache.erase(_id);
    return results;
}

template <class object_t,double (*distance)(object_t,object_t),size_t bucket_size,size_t overflow>
double listofclusters<object_t,distance,bucket_size,overflow>::internal_distance(const internal_object_t &_a,const internal_object_t &_b)
{
    auto &inner = this->_dcache[_a.id()];
    auto it = inner.find(_b.id());
    if(it == inner.end())
        it = inner.emplace(_b.id(), distance(_a.object(), _b.object())).first;
    return it->second;
}

}  // namespace metric
#endif
