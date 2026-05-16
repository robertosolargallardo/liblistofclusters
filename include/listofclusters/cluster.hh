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
    // Bucket: sorted std::vector by ascending (distance, id). std::vector
    // is cache-friendly for sequential iteration, which is the dominant
    // access pattern during knn/range search. Prior std::set storage
    // caused a pointer chase per node and made bucket scans cache-bound.
    typedef std::vector<internal_object_t> bucket_t;

private:
    uint32_t           _id{0U};
    internal_object_t  _centroid{};
    bucket_t           _bucket{};
    double             _radius{0.0};


public:
    cluster(void) = default;
    cluster(const cluster&) = default;
    cluster(cluster&&) noexcept = default;
    cluster& operator=(const cluster&) = default;
    cluster& operator=(cluster&&) noexcept = default;
    ~cluster(void) = default;

    cluster(const uint32_t&, const internal_object_t&);

    void insert(const object_t&, const uint32_t&, const double&);
    void remove(const uint32_t&);
    [[nodiscard]] bool empty(void) const noexcept;

    [[nodiscard]] const internal_object_t& centroid(void) const noexcept { return _centroid; }
    [[nodiscard]] const bucket_t& bucket(void) const noexcept { return _bucket; }

    [[nodiscard]] size_t bucket_count(void) const noexcept { return _bucket.size(); }

    // Remove and return the bucket member with the largest distance to the
    // centroid. Bucket is kept sorted by ascending distance, so this is the
    // back element. Updates _radius to the new max bucket distance.
    internal_object_t pop_farthest(void);

    [[nodiscard]] double radius(void) const noexcept { return _radius; }
    void radius(const double &_r) noexcept { _radius = _r; }

    [[nodiscard]] uint32_t id(void) const noexcept { return _id; }

    [[nodiscard]] size_t size(void) const noexcept { return (_centroid.ghost() ? 0U : 1U) + _bucket.size(); }
    void clear(void);
};
template<class object_t>
cluster<object_t>::cluster(const uint32_t &_id, const internal_object_t &_centroid)
    : _id(_id), _centroid(_centroid)
{
}

template<class object_t>
void cluster<object_t>::insert(const object_t &_object, const uint32_t &_id, const double &_distance)
{
    if (_distance > this->_radius)
        this->_radius = _distance;

    // Sorted insert: keep _bucket ordered by ascending (distance, id) so
    // the std::vector matches the prior std::set contract.
    internal_object_t obj(_object, _id, _distance);
    const auto key = std::tuple{_distance, _id};
    auto it = std::lower_bound(this->_bucket.begin(), this->_bucket.end(), key,
        [](const internal_object_t &m, const std::tuple<double, uint32_t> &k) noexcept {
            return std::tuple{m.distance(), m.id()} < k;
        });
    // Drop true duplicates (same id) to preserve the old dedup-by-id semantics.
    if (it != this->_bucket.end() && it->id() == _id) return;
    this->_bucket.insert(it, std::move(obj));
}

template<class object_t>
void cluster<object_t>::remove(const uint32_t &_id)
{
    if (this->_centroid.id() == _id) {
        this->_centroid.ghost(true);
        return;
    }
    auto it = std::find_if(this->_bucket.begin(), this->_bucket.end(),
        [_id](const internal_object_t &o) noexcept { return o.id() == _id; });
    if (it != this->_bucket.end())
        this->_bucket.erase(it);
}

template<class object_t>
bool cluster<object_t>::empty(void) const noexcept
{
    return this->_centroid.ghost() && this->_bucket.empty();
}

template<class object_t>
void cluster<object_t>::clear(void)
{
    this->_radius = 0.0;
    this->_bucket.clear();
    this->_centroid.ghost(true);
}

template<class object_t>
typename cluster<object_t>::internal_object_t cluster<object_t>::pop_farthest(void)
{
    // Bucket is sorted ascending by (distance, id) - the back element is the
    // farthest from the centroid. pop_back is O(1).
    internal_object_t farthest = std::move(this->_bucket.back());
    this->_bucket.pop_back();
    this->_radius = this->_bucket.empty() ? 0.0 : this->_bucket.back().distance();
    return farthest;
}

}  // namespace metric
#endif
