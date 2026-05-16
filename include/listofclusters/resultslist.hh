#ifndef _METRIC_RESULTSLIST_HH_
#define _METRIC_RESULTSLIST_HH_
#include <listofclusters/internal_object.hh>

namespace metric
{

template<class object_t>
class resultslist
{
public:
    typedef internal_object<object_t> internal_object_t;
    typedef std::set<internal_object_t,compare<internal_object_t>> results_t;

private:
    internal_object_t _centroid{};
    results_t         _results{};
    size_t            _k{std::numeric_limits<size_t>::max()};

public:
    resultslist(void) = default;
    resultslist(const resultslist&) = default;
    resultslist(resultslist&&) noexcept = default;
    resultslist& operator=(const resultslist&) = default;
    resultslist& operator=(resultslist&&) noexcept = default;
    ~resultslist(void) = default;

    explicit resultslist(const internal_object_t &_c) : _centroid(_c) {}
    resultslist(const internal_object_t &_c, const size_t &_kk) : _centroid(_c), _k(_kk) {}

    [[nodiscard]] const internal_object_t& centroid(void) const noexcept { return _centroid; }
    [[nodiscard]] const results_t& results(void) const noexcept { return _results; }
    [[nodiscard]] size_t size(void) const noexcept { return _results.size(); }

    void push(const object_t &_object, const uint32_t &_id, const double &_distance)
    {
        this->_results.emplace(_object, _id, _distance);
        if(this->_results.size() > this->_k)
            this->_results.erase(std::prev(this->_results.end()));
    }
};

}  // namespace metric
#endif
