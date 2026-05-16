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
    // Sorted std::vector ordered ascending by (distance, id). For typical
    // k (10-100) this beats std::set on every operation because the data
    // is contiguous and the constant factors of pointer-chasing through a
    // red-black tree dwarf the asymptotics. push() is O(k) but k is tiny.
    typedef std::vector<internal_object_t> results_t;

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

    explicit resultslist(const internal_object_t &_c) : _centroid(_c)
    {
        // _k stays at max for unbounded range search; reserve a small cap.
        _results.reserve(16);
    }
    resultslist(const internal_object_t &_c, const size_t &_kk) : _centroid(_c), _k(_kk)
    {
        _results.reserve(_kk + 1);
    }

    [[nodiscard]] const internal_object_t& centroid(void) const noexcept { return _centroid; }
    [[nodiscard]] const results_t& results(void) const noexcept { return _results; }
    [[nodiscard]] size_t size(void) const noexcept { return _results.size(); }

    void push(const object_t &_object, const uint32_t &_id, const double &_distance)
    {
        // Sorted insert with dedup-by-id (matches the prior std::set
        // contract). For small k the lower_bound + insert is O(k) but
        // entirely in cache, dominating the std::set's O(log k) tree path.
        const auto key = std::tuple{_distance, _id};
        auto it = std::lower_bound(_results.begin(), _results.end(), key,
            [](const internal_object_t &m, const std::tuple<double, uint32_t> &k) noexcept {
                return std::tuple{m.distance(), m.id()} < k;
            });
        if (it != _results.end() && it->id() == _id) return;  // dedup
        _results.emplace(it, _object, _id, _distance);
        if (_results.size() > _k)
            _results.pop_back();
    }
};

}  // namespace metric
#endif
