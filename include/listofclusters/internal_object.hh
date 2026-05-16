
#ifndef _METRIC_INTERNAL_OBJECT_HH_
#define _METRIC_INTERNAL_OBJECT_HH_
#include <listofclusters/glob.hh>

namespace metric
{

template<class object_t> class internal_object;
template<class internal_object_t>
class compare
{
public:
    bool operator()(const internal_object_t &_a,const internal_object_t &_b) const noexcept
    {
        // Treat objects with the same id as equivalent (dedup-by-id semantics)
        if(_a.id()==_b.id()) return(false);
        // Order by distance, breaking ties on id to satisfy strict weak ordering
        return(std::tuple{_a.distance(),_a.id()} < std::tuple{_b.distance(),_b.id()});
    }
};

template<class object_t>
class internal_object
{
private:
    uint32_t _id{0U};
    object_t _object{};
    double   _distance{0.0};
    bool     _ghost{true};

public:
    internal_object(void) = default;
    internal_object(const internal_object&) = default;
    internal_object(internal_object&&) noexcept = default;
    internal_object& operator=(const internal_object&) = default;
    internal_object& operator=(internal_object&&) noexcept = default;
    ~internal_object(void) = default;

    internal_object(const object_t &_object,const uint32_t &_id,const double &_distance=0.0)
        : _id(_id), _object(_object), _distance(_distance), _ghost(false) {}
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
    bool ghost(void) const
    {
        return(this->_ghost);
    }

    void distance(const double &_distance)
    {
        this->_distance=_distance;
    }
    void ghost(const bool &_ghost)
    {
        this->_ghost=_ghost;
    }
};
}
#endif
