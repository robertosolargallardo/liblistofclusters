// nanobind bindings for liblistofclusters.
//
// Exposes a single `Index` class with the same online + batch API the C++
// library provides, plus a module-level `available_metrics()` function that
// surfaces the C++ `metric::available_metrics[]` compile-time registry.
//
// Design notes:
// - object_t is bound as std::vector<double> (runtime D). NumPy arrays are
//   converted at the API boundary; D is determined from the array's last
//   dimension and is not fixed at compile time.
// - bucket_size is a C++ template parameter so it's compile-time fixed.
//   v0.1 hardcodes 20 (the empirical sweet spot per the bench's
//   bucket-size sweep).
// - Multi-metric support via a type-erased base class + templated concrete
//   subclass per metric. A factory function picks the right concrete type
//   from the user-provided metric name string.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <listofclusters/listofclusters.hh>
#include <listofclusters/metrics.hh>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace nb = nanobind;
using vec_t = std::vector<double>;

namespace {

constexpr std::size_t kBucketSize = 20;
constexpr std::size_t kOverflow   = 80;

// Type-erased interface so a single Python `Index` class can hold any of
// the supported metric template instantiations.
class IndexBase
{
public:
    virtual ~IndexBase() = default;
    virtual std::string metric_name() const = 0;
    virtual std::size_t cluster_count() const = 0;
    virtual bool empty() const = 0;
    virtual void clear() = 0;
    virtual void insert(const vec_t&, std::uint32_t) = 0;
    virtual void remove(const vec_t&, std::uint32_t) = 0;
    virtual void batch_insert(const std::vector<vec_t>&, const std::vector<std::uint32_t>&) = 0;
    virtual void batch_remove(const std::vector<vec_t>&, const std::vector<std::uint32_t>&) = 0;
    virtual void bulk_build(const std::vector<vec_t>&, const std::vector<std::uint32_t>&) = 0;
    virtual std::pair<std::vector<std::uint32_t>, std::vector<double>>
        knn(const vec_t&, std::size_t k) = 0;
    virtual std::pair<std::vector<std::uint32_t>, std::vector<double>>
        range(const vec_t&, double radius) = 0;
    virtual std::pair<std::vector<std::vector<std::uint32_t>>,
                      std::vector<std::vector<double>>>
        batch_knn(const std::vector<vec_t>&, std::size_t k, unsigned nthreads) = 0;
};

template <class Metric>
class IndexImpl final : public IndexBase
{
    metric::listofclusters<vec_t, Metric, kBucketSize, kOverflow> _idx;
    std::string _name;

    static std::uint32_t fresh_qid(const decltype(_idx)& idx) {
        // Use a high constant to avoid colliding with user-inserted ids -
        // the index's own knn filters out matches with this qid.
        (void)idx;
        return std::numeric_limits<std::uint32_t>::max() - 1U;
    }

public:
    explicit IndexImpl(std::string name) : _name(std::move(name)) {}

    std::string metric_name() const override { return _name; }
    std::size_t cluster_count() const override { return _idx.size(); }
    bool empty() const override { return _idx.empty(); }
    void clear() override { _idx.clear(); }

    void insert(const vec_t& v, std::uint32_t id) override { _idx.insert(v, id); }
    void remove(const vec_t& v, std::uint32_t id) override { _idx.remove(v, id); }

    void batch_insert(const std::vector<vec_t>& objs, const std::vector<std::uint32_t>& ids) override {
        _idx.insert(objs, ids);
    }
    void batch_remove(const std::vector<vec_t>& objs, const std::vector<std::uint32_t>& ids) override {
        _idx.remove(objs, ids);
    }
    void bulk_build(const std::vector<vec_t>& objs, const std::vector<std::uint32_t>& ids) override {
        _idx.bulk_build(objs, ids);
    }

    std::pair<std::vector<std::uint32_t>, std::vector<double>>
    knn(const vec_t& q, std::size_t k) override {
        auto res = _idx.knn_search(q, fresh_qid(_idx), k);
        std::vector<std::uint32_t> ids;
        std::vector<double> dists;
        ids.reserve(res.results().size());
        dists.reserve(res.results().size());
        for (const auto& r : res.results()) {
            ids.push_back(r.id());
            dists.push_back(r.distance());
        }
        return {std::move(ids), std::move(dists)};
    }

    std::pair<std::vector<std::uint32_t>, std::vector<double>>
    range(const vec_t& q, double radius) override {
        auto res = _idx.range_search(q, fresh_qid(_idx), radius);
        std::vector<std::uint32_t> ids;
        std::vector<double> dists;
        ids.reserve(res.results().size());
        dists.reserve(res.results().size());
        for (const auto& r : res.results()) {
            ids.push_back(r.id());
            dists.push_back(r.distance());
        }
        return {std::move(ids), std::move(dists)};
    }

    std::pair<std::vector<std::vector<std::uint32_t>>,
              std::vector<std::vector<double>>>
    batch_knn(const std::vector<vec_t>& queries, std::size_t k, unsigned nthreads) override {
        auto results = _idx.batch_knn(queries, /*start_qid=*/std::numeric_limits<std::uint32_t>::max() - 1U - static_cast<std::uint32_t>(queries.size()),
                                      k, nthreads);
        std::vector<std::vector<std::uint32_t>> ids_2d;
        std::vector<std::vector<double>>        dists_2d;
        ids_2d.reserve(results.size());
        dists_2d.reserve(results.size());
        for (const auto& r : results) {
            std::vector<std::uint32_t> ids;
            std::vector<double> dists;
            ids.reserve(r.results().size());
            dists.reserve(r.results().size());
            for (const auto& x : r.results()) {
                ids.push_back(x.id());
                dists.push_back(x.distance());
            }
            ids_2d.push_back(std::move(ids));
            dists_2d.push_back(std::move(dists));
        }
        return {std::move(ids_2d), std::move(dists_2d)};
    }
};

std::unique_ptr<IndexBase> make_index(const std::string& metric_name)
{
    if (metric_name == "euclidean")
        return std::make_unique<IndexImpl<metric::euclidean>>(metric_name);
    if (metric_name == "manhattan")
        return std::make_unique<IndexImpl<metric::manhattan>>(metric_name);
    if (metric_name == "chebyshev")
        return std::make_unique<IndexImpl<metric::chebyshev>>(metric_name);
    if (metric_name == "canberra")
        return std::make_unique<IndexImpl<metric::canberra>>(metric_name);
    throw std::invalid_argument("unsupported metric: '" + metric_name +
        "' (supported: euclidean, manhattan, chebyshev, canberra)");
}

// Adapter: 1D numpy array -> std::vector<double>. Accepts any contiguous
// 1D array of floating type; values are copied (zero-copy is awkward when
// the C++ side stores std::vector internally, which it must to satisfy the
// LC algorithm's stable address requirement).
vec_t to_vec(nb::ndarray<const double, nb::ndim<1>, nb::c_contig> arr) {
    const double* p = arr.data();
    return vec_t(p, p + arr.shape(0));
}

// 2D numpy array -> vector<vector<double>>.
std::vector<vec_t> to_vec2d(nb::ndarray<const double, nb::ndim<2>, nb::c_contig> arr) {
    const std::size_t n = arr.shape(0);
    const std::size_t d = arr.shape(1);
    const double* p = arr.data();
    std::vector<vec_t> out;
    out.reserve(n);
    for (std::size_t i = 0; i < n; ++i)
        out.emplace_back(p + i * d, p + i * d + d);
    return out;
}

std::vector<std::uint32_t> to_ids(nb::ndarray<const std::uint32_t, nb::ndim<1>, nb::c_contig> arr) {
    const std::uint32_t* p = arr.data();
    return std::vector<std::uint32_t>(p, p + arr.shape(0));
}

}  // namespace

NB_MODULE(_listofclusters, m)
{
    m.doc() = "Python bindings for liblistofclusters: exact metric-space NN index.";

    // Index: type-erased over the metric. v0.1 supports euclidean, manhattan,
    // chebyshev, canberra (the four LC-friendly numeric vector metrics).
    // bucket_size is hardcoded to 20 in this build.
    nb::class_<IndexBase>(m, "Index")
        .def(nb::new_([](const std::string& metric) -> IndexBase* {
                return make_index(metric).release();
            }),
            nb::arg("metric") = std::string("euclidean"),
            "Construct an empty index. `metric` must be one of "
            "available_metrics_supported().")

        .def_prop_ro("metric", &IndexBase::metric_name,
            "Metric this index uses.")
        .def_prop_ro("cluster_count", &IndexBase::cluster_count,
            "Number of LC clusters currently in the index.")
        .def("__len__", &IndexBase::cluster_count)
        .def_prop_ro("empty", &IndexBase::empty)
        .def("clear", &IndexBase::clear,
            "Reset the index to empty.")

        // Online insert: one point.
        .def("insert",
            [](IndexBase& self, nb::ndarray<const double, nb::ndim<1>, nb::c_contig> v,
               std::uint32_t id) {
                self.insert(to_vec(v), id);
            },
            nb::arg("v"), nb::arg("id"),
            "Insert one point with the given id.")

        // Batch insert: (N, D) array + (N,) ids.
        .def("insert_batch",
            [](IndexBase& self,
               nb::ndarray<const double, nb::ndim<2>, nb::c_contig> V,
               nb::ndarray<const std::uint32_t, nb::ndim<1>, nb::c_contig> ids) {
                if (V.shape(0) != ids.shape(0))
                    throw std::invalid_argument("V.shape[0] must equal ids.shape[0]");
                self.batch_insert(to_vec2d(V), to_ids(ids));
            },
            nb::arg("V"), nb::arg("ids"),
            "Insert a batch of points from an (N, D) array with (N,) ids.")

        // Online remove.
        .def("remove",
            [](IndexBase& self, nb::ndarray<const double, nb::ndim<1>, nb::c_contig> v,
               std::uint32_t id) {
                self.remove(to_vec(v), id);
            },
            nb::arg("v"), nb::arg("id"),
            "Remove the (v, id) pair from the index.")

        // Batch remove.
        .def("remove_batch",
            [](IndexBase& self,
               nb::ndarray<const double, nb::ndim<2>, nb::c_contig> V,
               nb::ndarray<const std::uint32_t, nb::ndim<1>, nb::c_contig> ids) {
                if (V.shape(0) != ids.shape(0))
                    throw std::invalid_argument("V.shape[0] must equal ids.shape[0]");
                self.batch_remove(to_vec2d(V), to_ids(ids));
            },
            nb::arg("V"), nb::arg("ids"),
            "Remove a batch of (V, ids) pairs.")

        // bulk_build (canonical LC construction; replaces state).
        .def("bulk_build",
            [](IndexBase& self,
               nb::ndarray<const double, nb::ndim<2>, nb::c_contig> V,
               nb::ndarray<const std::uint32_t, nb::ndim<1>, nb::c_contig> ids) {
                if (V.shape(0) != ids.shape(0))
                    throw std::invalid_argument("V.shape[0] must equal ids.shape[0]");
                self.bulk_build(to_vec2d(V), to_ids(ids));
            },
            nb::arg("V"), nb::arg("ids"),
            "Canonical LC construction. Replaces any existing index state.")

        // kNN, single query. Returns (ids, dists) tuple.
        .def("knn",
            [](IndexBase& self, nb::ndarray<const double, nb::ndim<1>, nb::c_contig> q,
               std::size_t k) {
                return self.knn(to_vec(q), k);
            },
            nb::arg("q"), nb::arg("k") = 10,
            "k nearest neighbors of q. Returns (ids, distances) as Python lists.")

        // Range search.
        .def("range",
            [](IndexBase& self, nb::ndarray<const double, nb::ndim<1>, nb::c_contig> q,
               double radius) {
                return self.range(to_vec(q), radius);
            },
            nb::arg("q"), nb::arg("radius"),
            "Points within `radius` of q (exact). Returns (ids, distances).")

        // Batch kNN, parallel across worker threads.
        .def("batch_knn",
            [](IndexBase& self,
               nb::ndarray<const double, nb::ndim<2>, nb::c_contig> Q,
               std::size_t k, unsigned nthreads) {
                return self.batch_knn(to_vec2d(Q), k, nthreads);
            },
            nb::arg("Q"), nb::arg("k") = 10, nb::arg("nthreads") = 0,
            "k nearest neighbors of every row in Q, parallelized across "
            "`nthreads` worker threads (0 -> hardware concurrency).");

    // Module-level: enumerate available metrics. Pulls from the C++
    // metric::available_metrics registry so the lists stay in sync.
    m.def("available_metrics",
        []() {
            std::vector<std::tuple<std::string, std::string, std::string>> out;
            out.reserve(metric::available_metrics_count);
            for (std::size_t i = 0; i < metric::available_metrics_count; ++i) {
                const auto& md = metric::available_metrics[i];
                out.emplace_back(std::string(md.name),
                                 std::string(md.description),
                                 std::string(md.domain));
            }
            return out;
        },
        "Full list of metrics shipped by the C++ library, as "
        "(name, description, domain) tuples. The Python wrapper "
        "currently supports a subset; see Index() for what's bindable.");

    m.def("supported_metrics",
        []() {
            // The subset that the Python factory currently knows how to
            // instantiate. Subset of available_metrics().
            return std::vector<std::string>{"euclidean", "manhattan", "chebyshev", "canberra"};
        },
        "Metrics that this Python wrapper can construct via Index(metric=...).");
}
