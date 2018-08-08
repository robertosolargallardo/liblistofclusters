#ifndef _METRIC_RESULTSLIST_HH_
#define _METRIC_RESULTSLIST_HH_
#include <internal_object.hh>

namespace metric
{

template<class object_t>
class resultslist
{
public:
    typedef internal_object<object_t> internal_object_t;
    typedef std::set<internal_object_t,compare<internal_object_t>> results_t;

private:
    internal_object_t 			_centroid;
    results_t        				_results;
    size_t                    	_k;
    std::map<uint32_t,double> 	_dcache;

public:
    resultslist(void);
    resultslist(const resultslist&);
    resultslist(const internal_object_t&,const size_t&);
    resultslist(const internal_object_t&);
    resultslist& operator=(const resultslist&);
    ~resultslist(void);

    internal_object_t centroid(void) const;

    void push(const object_t&,const uint32_t&,const double&);
    double dcache(const uint32_t&);
    void dcache(const uint32_t&,const double&);
    results_t results(void)
    {
        return(this->_results);
    }
};
template<class object_t>
resultslist<object_t>::resultslist(void)
{
    this->_k=std::numeric_limits<size_t>::max();
}
template<class object_t>
resultslist<object_t>::resultslist(const internal_object_t &_centroid)
{
    this->_centroid=_centroid;
    this->_k=std::numeric_limits<size_t>::max();
}
template<class object_t>
resultslist<object_t>::resultslist(const internal_object_t &_centroid,const size_t &_k)
{
    this->_centroid=_centroid;
    this->_k=_k;
}
template<class object_t>
resultslist<object_t>::resultslist(const resultslist &_results)
{
    this->_centroid=_results._centroid;
    this->_results=_results._results;
    this->_k=_results._k;
    this->_dcache=_results._dcache;
}
template<class object_t>
resultslist<object_t>& resultslist<object_t>::operator=(const resultslist &_results)
{
    this->_centroid=_results._centroid;
    this->_results=_results._results;
    this->_k=_results._k;
    this->_dcache=_results._dcache;
    return(*this);
}
template<class object_t>
resultslist<object_t>::~resultslist(void)
{
    this->_results.clear();
    this->_dcache.clear();
}
template<class object_t>
typename resultslist<object_t>::internal_object_t resultslist<object_t>::centroid(void) const
{
    return(this->_centroid);
}
template<class object_t>
void resultslist<object_t>::push(const object_t &_object,const uint32_t &_id,const double &_distance)
{
    this->_results.insert(internal_object_t(_object,_id,_distance));

    if(this->_results.size()>this->_k)
        this->_results.erase(--this->_results.end());
}
template<class object_t>
double resultslist<object_t>::dcache(const uint32_t &_id)
{
    if(!this->_dcache.count(_id))
        return(-1.0);
    return(this->_dcache[_id]);
}
template<class object_t>
void resultslist<object_t>::dcache(const uint32_t &_id,const double &_distance)
{
    this->_dcache[_id]=_distance;
}

};
#endif
