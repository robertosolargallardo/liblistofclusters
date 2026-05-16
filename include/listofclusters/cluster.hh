#ifndef _METRIC_CLUSTER_HH_
#define _METRIC_CLUSTER_HH_
#include <listofclusters/glob.hh>
#include <listofclusters/internal_object.hh>

namespace metric
{

template <class object_t>
class cluster
{
public:
    typedef internal_object<object_t> internal_object_t;
    typedef std::set<internal_object_t,compare<internal_object_t>> bucket_t;

private:
    uint32_t  _id{0U};
    internal_object_t  _centroid{};
    bucket_t  _bucket{};
    double    _radius{0.0};


public:
    cluster(void) = default;
    cluster(const cluster&) = default;
    cluster(cluster&&) noexcept = default;
    cluster& operator=(const cluster&) = default;
    cluster& operator=(cluster&&) noexcept = default;
    ~cluster(void) = default;

    cluster(const uint32_t&,const internal_object_t&);

    void insert(const object_t&,const uint32_t&,const double&);
    void remove(const uint32_t&);
    bool empty(void);

    internal_object_t centroid(void) const;
    bucket_t bucket(void) const;

    // Direct, non-copying observers needed by the LC insert algorithm.
    size_t bucket_count(void) const noexcept;

    // Remove and return the bucket member with the largest distance to the
    // centroid (the LC "fixed bucket size" overflow case ejects this one
    // and continues inserting it in the tail of the list). Updates _radius
    // to the new maximum bucket distance (or 0 if the bucket becomes empty).
    internal_object_t pop_farthest(void);

    double radius(void) const;
    void radius(const double&);

    uint32_t id(void) const;

    size_t size(void) const;
    void clear(void);
};
template<class object_t>
cluster<object_t>::cluster(const uint32_t &_id,const internal_object_t &_centroid)
    : _id(_id), _centroid(_centroid)
{
}

template<class object_t>
void cluster<object_t>::insert(const object_t &_object,const uint32_t &_id,const double &_distance)
{
    if(_distance>this->_radius)
        this->_radius=_distance;
    this->_bucket.insert(internal_object_t(_object,_id,_distance));
}

template<class object_t>
typename cluster<object_t>::internal_object_t cluster<object_t>::centroid(void) const
{
    return(this->_centroid);
}

template<class object_t>
typename cluster<object_t>::bucket_t cluster<object_t>::bucket(void) const
{
    return(this->_bucket);
}
template<class object_t>
double cluster<object_t>::radius(void) const
{
    return(this->_radius);
}
template<class object_t>
void cluster<object_t>::radius(const double &_radius)
{
    this->_radius=_radius;
}

template<class object_t>
void cluster<object_t>::remove(const uint32_t &_id)
{
    if(this->_centroid.id()==_id)
        this->_centroid.ghost(true);
    else
        this->_bucket.erase(std::find_if(this->_bucket.begin(),this->_bucket.end(),[&_id](const internal_object_t &_object)->bool{return(_object.id()==_id);}));
}

template<class object_t>
bool cluster<object_t>::empty(void)
{
    return(this->_centroid.ghost() && this->_bucket.empty());
}
template<class object_t>
uint32_t cluster<object_t>::id(void) const
{
    return(this->_id);
}

template<class object_t>
size_t cluster<object_t>::size(void) const
{
    return((this->_centroid.ghost()?0:1)+this->_bucket.size());
}
template<class object_t>
void cluster<object_t>::clear(void)
{
    this->_radius=0.0;
    this->_bucket.clear();
    this->_centroid.ghost(true);
}

template<class object_t>
size_t cluster<object_t>::bucket_count(void) const noexcept
{
    return this->_bucket.size();
}

template<class object_t>
typename cluster<object_t>::internal_object_t cluster<object_t>::pop_farthest(void)
{
    // Bucket is std::set<internal_object_t, compare> sorted by (distance, id).
    // The last element has the largest (distance, id) tuple - i.e. the bucket
    // member farthest from this cluster's centroid (id breaks ties).
    auto last = std::prev(this->_bucket.end());
    internal_object_t farthest = *last;
    this->_bucket.erase(last);
    if (this->_bucket.empty())
        this->_radius = 0.0;
    else
        this->_radius = std::prev(this->_bucket.end())->distance();
    return farthest;
}
};
#endif
