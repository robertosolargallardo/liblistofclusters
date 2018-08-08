
#ifndef _METRIC_INTERNAL_OBJECT_HH_
#define _METRIC_INTERNAL_OBJECT_HH_
#include <glob.hh>

namespace metric
{

template<class object_t> class internal_object;
template<class internal_object_t>
class compare
{
public:
    bool operator()(const internal_object_t &_a,const internal_object_t &_b) const
    {
        if(_a.id()==_b.id()) return(false);
        return(_a.distance()<=_b.distance());
    }
};

template<class object_t>
class internal_object
{
private:
    uint32_t _id;
    object_t _object;
    double   _distance;

public:
    internal_object(void)
    {
        ;
    }
    internal_object(const internal_object &_internal_object)
    {
        this->_id=_internal_object._id;
        this->_object=_internal_object._object;
        this->_distance=_internal_object._distance;
    }
    internal_object(const object_t &_object,const uint32_t &_id,const double &_distance=0.0)
    {
        this->_id=_id;
        this->_object=_object;
        this->_distance=_distance;
    }
    internal_object& operator=(const internal_object &_internal_object)
    {
        this->_id=_internal_object._id;
        this->_object=_internal_object._object;
        this->_distance=_internal_object._distance;
        return(*this);
    }
    ~internal_object(void)
    {
        ;
    }
    uint32_t id(void) const
    {
        return(this->_id);
    }
    object_t object(void) const
    {
        return(this->_object);
    }
    double distance(void) const
    {
        return(this->_distance);
    }
    void distance(const double &_distance)
    {
        this->_distance=_distance;
    }
};
}
#endif
