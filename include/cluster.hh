#ifndef _METRIC_CLUSTER_HH_
#define _METRIC_CLUSTER_HH_
#include <glob.hh>
#include <internal_object.hh>

namespace metric
{

template <class object_t>
class cluster
{
public:
    typedef internal_object<object_t> internal_object_t;
    typedef std::set<internal_object_t,compare<internal_object_t>> bucket_t;

private:
    uint32_t  _id;
    internal_object_t  _centroid;
    bucket_t  _bucket;
    double    _radius;
    bool      _ghost;


public:
    cluster(void);
    cluster(const cluster&);
    cluster& operator=(const cluster&);
    ~cluster(void);

    cluster(const uint32_t&,const internal_object_t&);

    void insert(const object_t&,const uint32_t&,const double&);
    void remove(const uint32_t&);
    bool empty(void);

    internal_object_t centroid(void) const;
    bucket_t	bucket(void) const;

    double radius(void) const;
    void radius(const double&);

    uint32_t id(void) const;

    size_t size(void) const;
    void clear(void);

    bool ghost(void) const;

};
template<class object_t>
cluster<object_t>::cluster(void)
{
    this->_radius=0.0;
    this->_ghost=true;
}

template<class object_t>
cluster<object_t>::cluster(const cluster &_cluster)
{
    this->_id=_cluster._id;
    this->_centroid=_cluster._centroid;
    this->_bucket=_cluster._bucket;
    this->_radius=_cluster._radius;
    this->_ghost=_cluster._ghost;
}

template<class object_t>
cluster<object_t>& cluster<object_t>::operator=(const cluster &_cluster)
{
    this->_id=_cluster._id;
    this->_centroid=_cluster._centroid;
    this->_bucket=_cluster._bucket;
    this->_radius=_cluster._radius;
    this->_ghost=_cluster._ghost;
    return(*this);
}

template<class object_t>
cluster<object_t>::~cluster(void)
{
    this->_bucket.clear();
}

template<class object_t>
cluster<object_t>::cluster(const uint32_t &_id,const internal_object_t &_centroid)
{
    this->_id=_id;
    this->_centroid=_centroid;
    this->_radius=0.0;
    this->_ghost=false;
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
        this->_ghost=true;
    else
        this->_bucket.erase(std::find_if(this->_bucket.begin(),this->_bucket.end(),[&_id](const internal_object_t &_object)->bool{return(_object.id()==_id);}));
}

template<class object_t>
bool cluster<object_t>::empty(void)
{
    return(this->_ghost && this->_bucket.empty());
}
template<class object_t>
uint32_t cluster<object_t>::id(void) const
{
    return(this->_id);
}

template<class object_t>
size_t cluster<object_t>::size(void) const
{
    return((this->_ghost?0:1)+this->_bucket.size());
}
template<class object_t>
void cluster<object_t>::clear(void)
{
    this->_ghost=true;
    this->_radius=0.0;
    this->_bucket.clear();
}
template<class object_t>
bool cluster<object_t>::ghost(void) const
{
    return(this->_ghost);
}
};
#endif
