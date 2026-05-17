// Self-contained microbenchmarks for liblistofclusters.
//
// No external benchmark framework - just std::chrono with median-of-runs
// reporting. Build with -O3, no sanitizers (the Makefile in this directory
// takes care of the flags). The point is to establish baseline numbers
// before phase 4 (threading) and phase 5 (perf pass) so wins can be
// measured against a known reference.
//
// Phase 5.2: payload type is std::array<double, BENCH_D> (compile-time
// fixed dimensionality). With inline storage, each internal_object holds
// the full vector in place, so iterating a cluster's bucket is a single
// sequential walk through contiguous memory - no per-element heap pointer
// chase, which was the dominant cost with the prior std::vector payload.
//
// To bench a different D, change BENCH_D and recompile (`make clean`
// inside bench/ to force the rebuild). The N and Q workload sizes remain
// runtime CLI args.
//
// Usage:
//   make bench         # build and run
//   ./bench_main       # run directly
//   ./bench_main N D Q gen   # D is checked against BENCH_D
//
// What's reported:
//   - Insert throughput  (LC build cost)
//   - kNN throughput     (LC vs brute force baseline)
//   - Range throughput   (LC vs brute force baseline)
//   - Recall@k vs brute  (sanity check that the LC results match)
//   - bucket_size sweep

#include <listofclusters/listofclusters.hh>
#include <listofclusters/metrics.hh>

// Vendored hnswlib for the approximate-NN baseline (phase 5b). Header-only,
// MIT licensed; see third_party/hnswlib/LICENSE and third_party/hnswlib/VENDOR.txt.
#include <hnswlib/hnswlib.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#ifndef BENCH_D
#define BENCH_D 8
#endif

inline constexpr std::size_t kD = BENCH_D;
using vec_t  = std::array<double, kD>;
using clock_t_ = std::chrono::steady_clock;

struct euclid {
    [[nodiscard]] double operator()(const vec_t &a, const vec_t &b) const noexcept
    {
        double s = 0.0;
        for (std::size_t i = 0; i < kD; ++i) {
            const double d = a[i] - b[i];
            s += d * d;
        }
        return std::sqrt(s);
    }
};

// bucket_size=20 is the published default ("Engineering efficient metric
// indexes", Chavez et al.) and empirically best for our N=10k workload.
// At larger N (>= 100k) or high D, 16 is slightly better; see the sweep
// section at the end of the bench output.
template <std::size_t bucket = 20, std::size_t overflow = 80>
using idx_t = metric::listofclusters<vec_t, euclid, bucket, overflow>;

// SIMD-specialized version for the direct comparison row.
template <std::size_t bucket = 20, std::size_t overflow = 80>
using idx_t_simd = metric::listofclusters<vec_t, metric::euclidean_simd<kD>, bucket, overflow>;

// -----------------------------------------------------------------------------
// Timing helpers
// -----------------------------------------------------------------------------

template <class F>
[[nodiscard]] static double time_ns(F &&fn)
{
    const auto t0 = clock_t_::now();
    fn();
    const auto t1 = clock_t_::now();
    return std::chrono::duration<double, std::nano>(t1 - t0).count();
}

struct Stats {
    double median_ns;
    double min_ns;
    double max_ns;
    double per_op_ns;
    double ops_per_s;
};

static Stats summarize(std::vector<double> samples_ns, std::size_t ops_per_sample)
{
    std::sort(samples_ns.begin(), samples_ns.end());
    const double med = samples_ns[samples_ns.size() / 2];
    const double mn  = samples_ns.front();
    const double mx  = samples_ns.back();
    const double per_op = med / static_cast<double>(ops_per_sample);
    return Stats{med, mn, mx, per_op, 1e9 / per_op};
}

static void print_header()
{
    std::printf("%-32s %10s %12s %14s\n",
                "benchmark", "ops/sample", "per-op (us)", "throughput (M/s)");
    std::printf("%s\n", std::string(80, '-').c_str());
}

static void print_row(const std::string &name, std::size_t ops_per_sample, const Stats &s)
{
    std::printf("%-32s %10zu %12.3f %14.3f\n",
                name.c_str(), ops_per_sample, s.per_op_ns / 1000.0, s.ops_per_s / 1e6);
}

// -----------------------------------------------------------------------------
// Workloads
// -----------------------------------------------------------------------------

[[nodiscard]] static std::vector<vec_t>
make_dataset(std::size_t N, std::uint32_t seed)
{
    std::vector<vec_t> db(N);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    for (auto &v : db)
        for (auto &x : v) x = u(rng);
    return db;
}

// Mixture-of-Gaussians: points form `clusters` clusters, each centered at a
// uniformly-random point in [-1,1]^D, with isotropic gaussian noise of
// stdev = noise_sigma. Models real-world data with intrinsic dimensionality
// well below the nominal D - which is where LC's pruning actually pays off.
[[nodiscard]] static std::vector<vec_t>
make_clustered_dataset(std::size_t N, std::size_t n_clusters,
                       double noise_sigma, std::uint32_t seed)
{
    std::vector<vec_t> centers(n_clusters);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    for (auto &c : centers) for (auto &x : c) x = u(rng);

    std::vector<vec_t> db(N);
    std::uniform_int_distribution<std::size_t> pick(0, n_clusters - 1);
    std::normal_distribution<double> noise(0.0, noise_sigma);
    for (auto &v : db) {
        const auto &c = centers[pick(rng)];
        for (std::size_t i = 0; i < kD; ++i) v[i] = c[i] + noise(rng);
    }
    return db;
}

// Build a fresh index from `db` once; return mean time per insert.
[[nodiscard]] static Stats bench_insert(const std::vector<vec_t> &db, int repeats)
{
    std::vector<double> samples_ns;
    samples_ns.reserve(repeats);
    for (int r = 0; r < repeats; ++r) {
        idx_t<> idx;
        const double ns = time_ns([&] {
            for (std::uint32_t i = 0; i < db.size(); ++i)
                idx.insert(db[i], i);
        });
        samples_ns.push_back(ns);
    }
    return summarize(std::move(samples_ns), db.size());
}

// Build the index once, then run `Q` kNN queries against it `repeats` times.
[[nodiscard]] static Stats
bench_knn(const std::vector<vec_t> &db, const std::vector<vec_t> &queries,
          std::size_t k, int repeats)
{
    idx_t<> idx;
    for (std::uint32_t i = 0; i < db.size(); ++i) idx.insert(db[i], i);

    std::vector<double> samples_ns;
    samples_ns.reserve(repeats);
    volatile std::size_t sink = 0;
    for (int r = 0; r < repeats; ++r) {
        const double ns = time_ns([&] {
            for (std::uint32_t q = 0; q < queries.size(); ++q) {
                auto res = idx.knn_search(queries[q], static_cast<std::uint32_t>(db.size() + q), k);
                sink += res.results().size();
            }
        });
        samples_ns.push_back(ns);
    }
    (void)sink;
    return summarize(std::move(samples_ns), queries.size());
}

// kNN brute-force baseline (single threaded), for the same workload.
[[nodiscard]] static Stats
bench_knn_brute(const std::vector<vec_t> &db, const std::vector<vec_t> &queries,
                std::size_t k, int repeats)
{
    std::vector<double> samples_ns;
    samples_ns.reserve(repeats);
    volatile double sink = 0.0;
    euclid m;
    for (int r = 0; r < repeats; ++r) {
        const double ns = time_ns([&] {
            std::vector<std::pair<double, std::uint32_t>> heap;
            heap.reserve(db.size());
            for (const auto &q : queries) {
                heap.clear();
                for (std::uint32_t i = 0; i < db.size(); ++i)
                    heap.emplace_back(m(q, db[i]), i);
                std::partial_sort(heap.begin(), heap.begin() + std::min<std::size_t>(k, heap.size()), heap.end());
                for (std::size_t i = 0; i < std::min<std::size_t>(k, heap.size()); ++i)
                    sink += heap[i].first;
            }
        });
        samples_ns.push_back(ns);
    }
    (void)sink;
    return summarize(std::move(samples_ns), queries.size());
}

// Bulk build via Index::bulk_build (canonical LC static construction).
[[nodiscard]] static Stats
bench_bulk_build(const std::vector<vec_t> &db, int repeats)
{
    std::vector<std::uint32_t> ids(db.size());
    for (std::uint32_t i = 0; i < db.size(); ++i) ids[i] = i;

    std::vector<double> samples_ns;
    samples_ns.reserve(repeats);
    for (int r = 0; r < repeats; ++r) {
        idx_t<> idx;
        const double ns = time_ns([&] {
            idx.bulk_build(db, ids);
        });
        samples_ns.push_back(ns);
    }
    return summarize(std::move(samples_ns), db.size());
}

// kNN throughput against an index built by bulk_build (vs incremental).
[[nodiscard]] static Stats
bench_knn_bulk(const std::vector<vec_t> &db, const std::vector<vec_t> &queries,
               std::size_t k, int repeats)
{
    std::vector<std::uint32_t> ids(db.size());
    for (std::uint32_t i = 0; i < db.size(); ++i) ids[i] = i;
    idx_t<> idx;
    idx.bulk_build(db, ids);

    std::vector<double> samples_ns;
    samples_ns.reserve(repeats);
    volatile std::size_t sink = 0;
    for (int r = 0; r < repeats; ++r) {
        const double ns = time_ns([&] {
            for (std::uint32_t q = 0; q < queries.size(); ++q) {
                auto res = idx.knn_search(queries[q], static_cast<std::uint32_t>(db.size() + q), k);
                sink += res.results().size();
            }
        });
        samples_ns.push_back(ns);
    }
    (void)sink;
    return summarize(std::move(samples_ns), queries.size());
}

// kNN LC with the explicit NEON/AVX2 Euclidean (vs the auto-vec scalar).
[[nodiscard]] static Stats
bench_knn_simd(const std::vector<vec_t> &db, const std::vector<vec_t> &queries,
               std::size_t k, int repeats)
{
    idx_t_simd<> idx;
    for (std::uint32_t i = 0; i < db.size(); ++i) idx.insert(db[i], i);

    std::vector<double> samples_ns;
    samples_ns.reserve(repeats);
    volatile std::size_t sink = 0;
    for (int r = 0; r < repeats; ++r) {
        const double ns = time_ns([&] {
            for (std::uint32_t q = 0; q < queries.size(); ++q) {
                auto res = idx.knn_search(queries[q], static_cast<std::uint32_t>(db.size() + q), k);
                sink += res.results().size();
            }
        });
        samples_ns.push_back(ns);
    }
    (void)sink;
    return summarize(std::move(samples_ns), queries.size());
}

// -----------------------------------------------------------------------------
// KD-tree (exact NN). Recursive split on the axis of widest spread.
// Tight bounding boxes per node so the search prunes via per-axis distance.
// Falls back to brute force at small leaf size (16 here). Classic structure
// for low-D Euclidean; degrades sharply past ~20 dimensions but a useful
// reference at our D=8.
// -----------------------------------------------------------------------------
struct kdtree_t {
    struct node {
        std::size_t axis;
        double split;
        std::size_t left, right;       // children (0 = null)
        std::size_t lo, hi;            // index range in `order` (leaves only)
        std::array<double, kD> bb_lo, bb_hi;  // bounding box
    };
    std::vector<node> nodes;
    std::vector<std::uint32_t> order;  // permutation of [0, N) sorted into leaves
    const std::vector<vec_t>* data = nullptr;
    static constexpr std::size_t kLeafSize = 16;

    [[nodiscard]] std::size_t build_node(std::size_t lo, std::size_t hi)
    {
        const std::size_t idx = nodes.size();
        nodes.emplace_back();
        node &n = nodes.back();
        n.lo = lo; n.hi = hi; n.left = 0; n.right = 0;

        // Bounding box.
        for (std::size_t d = 0; d < kD; ++d) {
            n.bb_lo[d] = (*data)[order[lo]][d];
            n.bb_hi[d] = n.bb_lo[d];
        }
        for (std::size_t i = lo + 1; i < hi; ++i)
            for (std::size_t d = 0; d < kD; ++d) {
                const double v = (*data)[order[i]][d];
                if (v < n.bb_lo[d]) n.bb_lo[d] = v;
                if (v > n.bb_hi[d]) n.bb_hi[d] = v;
            }

        if (hi - lo <= kLeafSize) {
            n.axis = 0; n.split = 0.0;
            return idx;
        }

        // Pick split axis (widest extent), median-of-axis split.
        std::size_t best_axis = 0;
        double best_extent = -1.0;
        for (std::size_t d = 0; d < kD; ++d) {
            const double ext = n.bb_hi[d] - n.bb_lo[d];
            if (ext > best_extent) { best_extent = ext; best_axis = d; }
        }
        n.axis = best_axis;

        const std::size_t mid = lo + (hi - lo) / 2;
        std::nth_element(order.begin() + lo, order.begin() + mid, order.begin() + hi,
            [&](std::uint32_t a, std::uint32_t b) {
                return (*data)[a][best_axis] < (*data)[b][best_axis];
            });
        n.split = (*data)[order[mid]][best_axis];

        const std::size_t left  = build_node(lo, mid);
        const std::size_t right = build_node(mid, hi);
        nodes[idx].left  = left;
        nodes[idx].right = right;
        return idx;
    }

    void build(const std::vector<vec_t> &db)
    {
        data = &db;
        order.resize(db.size());
        std::iota(order.begin(), order.end(), std::uint32_t{0});
        nodes.clear();
        nodes.reserve(2 * db.size() / kLeafSize + 1);
        if (!db.empty()) (void)build_node(0, db.size());
    }

    // Squared L2 from q to the bounding box of `n`.
    [[nodiscard]] double bb_dist2(const vec_t &q, const node &n) const noexcept {
        double s = 0.0;
        for (std::size_t d = 0; d < kD; ++d) {
            if (q[d] < n.bb_lo[d])      { const double x = n.bb_lo[d] - q[d]; s += x * x; }
            else if (q[d] > n.bb_hi[d]) { const double x = q[d] - n.bb_hi[d]; s += x * x; }
        }
        return s;
    }

    void search(std::size_t idx, const vec_t &q,
                std::size_t k,
                std::vector<std::pair<double, std::uint32_t>> &heap) const
    {
        const node &n = nodes[idx];
        if (n.left == 0) {  // leaf
            euclid m;
            for (std::size_t i = n.lo; i < n.hi; ++i) {
                const std::uint32_t id = order[i];
                const double d = m(q, (*data)[id]);
                if (heap.size() < k) {
                    heap.emplace_back(d, id);
                    std::push_heap(heap.begin(), heap.end());
                } else if (d < heap.front().first) {
                    std::pop_heap(heap.begin(), heap.end());
                    heap.back() = {d, id};
                    std::push_heap(heap.begin(), heap.end());
                }
            }
            return;
        }
        // Branch nearer-first for tighter pruning.
        const std::size_t first  = (q[n.axis] < n.split) ? n.left  : n.right;
        const std::size_t second = (q[n.axis] < n.split) ? n.right : n.left;
        search(first, q, k, heap);
        const node &n2 = nodes[second];
        const double cur_worst2 = heap.size() < k
            ? std::numeric_limits<double>::infinity()
            : heap.front().first * heap.front().first;
        if (bb_dist2(q, n2) <= cur_worst2)
            search(second, q, k, heap);
    }

    [[nodiscard]] std::vector<std::uint32_t> knn(const vec_t &q, std::size_t k) const
    {
        std::vector<std::pair<double, std::uint32_t>> heap;
        heap.reserve(k + 1);
        if (!nodes.empty()) search(0, q, k, heap);
        std::sort(heap.begin(), heap.end());
        std::vector<std::uint32_t> out;
        out.reserve(heap.size());
        for (const auto &h : heap) out.push_back(h.second);
        return out;
    }
};

// Ball tree (general metric-space exact NN). Each internal node has a
// centroid + radius covering all members; pruning uses the triangle
// inequality d(q, c) - r > tau to skip subtrees. Build splits by the
// farthest-pair direction (cheap approximation of widest-spread).
// -----------------------------------------------------------------------------
struct balltree_t {
    struct node {
        vec_t centroid;
        double radius;
        std::size_t left, right;
        std::size_t lo, hi;
    };
    std::vector<node> nodes;
    std::vector<std::uint32_t> order;
    const std::vector<vec_t>* data = nullptr;
    static constexpr std::size_t kLeafSize = 16;

    [[nodiscard]] std::size_t build_node(std::size_t lo, std::size_t hi)
    {
        const std::size_t idx = nodes.size();
        nodes.emplace_back();
        node &n = nodes.back();
        n.lo = lo; n.hi = hi; n.left = 0; n.right = 0;

        // Centroid = mean of members.
        for (std::size_t d = 0; d < kD; ++d) n.centroid[d] = 0.0;
        for (std::size_t i = lo; i < hi; ++i)
            for (std::size_t d = 0; d < kD; ++d)
                n.centroid[d] += (*data)[order[i]][d];
        const double inv = 1.0 / static_cast<double>(hi - lo);
        for (std::size_t d = 0; d < kD; ++d) n.centroid[d] *= inv;

        // Radius = max distance from centroid to any member.
        euclid m;
        double r = 0.0;
        for (std::size_t i = lo; i < hi; ++i) {
            const double d = m(n.centroid, (*data)[order[i]]);
            if (d > r) r = d;
        }
        n.radius = r;

        if (hi - lo <= kLeafSize) return idx;

        // Split direction: find the farthest pair (cheap two-pivot heuristic).
        std::uint32_t a = order[lo];
        std::size_t i_far_a = lo;
        double dmax = -1.0;
        for (std::size_t i = lo; i < hi; ++i) {
            const double d = m((*data)[a], (*data)[order[i]]);
            if (d > dmax) { dmax = d; i_far_a = i; }
        }
        std::swap(order[lo], order[i_far_a]);
        std::uint32_t p1 = order[lo];

        std::size_t i_far_b = lo;
        dmax = -1.0;
        for (std::size_t i = lo; i < hi; ++i) {
            const double d = m((*data)[p1], (*data)[order[i]]);
            if (d > dmax) { dmax = d; i_far_b = i; }
        }
        std::swap(order[lo + 1], order[i_far_b]);
        std::uint32_t p2 = order[lo + 1];

        // Project onto the (p1, p2) axis; split at the median projection.
        std::vector<std::pair<double, std::uint32_t>> proj;
        proj.reserve(hi - lo);
        for (std::size_t i = lo; i < hi; ++i) {
            // Use distance to p1 as the sort key.
            const double dp1 = m((*data)[order[i]], (*data)[p1]);
            proj.emplace_back(dp1, order[i]);
        }
        std::sort(proj.begin(), proj.end());
        const std::size_t mid = (hi - lo) / 2;
        for (std::size_t i = 0; i < proj.size(); ++i)
            order[lo + i] = proj[i].second;
        (void)p2;

        const std::size_t left  = build_node(lo, lo + mid);
        const std::size_t right = build_node(lo + mid, hi);
        nodes[idx].left  = left;
        nodes[idx].right = right;
        return idx;
    }

    void build(const std::vector<vec_t> &db)
    {
        data = &db;
        order.resize(db.size());
        std::iota(order.begin(), order.end(), std::uint32_t{0});
        nodes.clear();
        nodes.reserve(2 * db.size() / kLeafSize + 1);
        if (!db.empty()) (void)build_node(0, db.size());
    }

    void search(std::size_t idx, const vec_t &q, std::size_t k,
                std::vector<std::pair<double, std::uint32_t>> &heap) const
    {
        euclid m;
        const node &n = nodes[idx];

        // Triangle-inequality prune at the subtree level.
        const double d_qc = m(q, n.centroid);
        const double cur_worst = heap.size() < k
            ? std::numeric_limits<double>::infinity()
            : heap.front().first;
        if (d_qc - n.radius > cur_worst) return;

        if (n.left == 0) {  // leaf
            for (std::size_t i = n.lo; i < n.hi; ++i) {
                const std::uint32_t id = order[i];
                const double d = m(q, (*data)[id]);
                if (heap.size() < k) {
                    heap.emplace_back(d, id);
                    std::push_heap(heap.begin(), heap.end());
                } else if (d < heap.front().first) {
                    std::pop_heap(heap.begin(), heap.end());
                    heap.back() = {d, id};
                    std::push_heap(heap.begin(), heap.end());
                }
            }
            return;
        }
        // Visit nearer child first.
        const double d_l = m(q, nodes[n.left].centroid);
        const double d_r = m(q, nodes[n.right].centroid);
        if (d_l < d_r) {
            search(n.left,  q, k, heap);
            search(n.right, q, k, heap);
        } else {
            search(n.right, q, k, heap);
            search(n.left,  q, k, heap);
        }
    }

    [[nodiscard]] std::vector<std::uint32_t> knn(const vec_t &q, std::size_t k) const
    {
        std::vector<std::pair<double, std::uint32_t>> heap;
        heap.reserve(k + 1);
        if (!nodes.empty()) search(0, q, k, heap);
        std::sort(heap.begin(), heap.end());
        std::vector<std::uint32_t> out;
        out.reserve(heap.size());
        for (const auto &h : heap) out.push_back(h.second);
        return out;
    }
};

// Common helper: brute-force ground truth for one query.
[[nodiscard]] static std::vector<std::uint32_t>
ground_truth(const std::vector<vec_t> &db, const vec_t &q, std::size_t k)
{
    euclid m;
    std::vector<std::pair<double, std::uint32_t>> bf;
    bf.reserve(db.size());
    for (std::uint32_t i = 0; i < db.size(); ++i)
        bf.emplace_back(m(q, db[i]), i);
    std::partial_sort(bf.begin(), bf.begin() + std::min<std::size_t>(k, bf.size()), bf.end());
    std::vector<std::uint32_t> out;
    out.reserve(std::min<std::size_t>(k, bf.size()));
    for (std::size_t i = 0; i < std::min<std::size_t>(k, bf.size()); ++i) out.push_back(bf[i].second);
    return out;
}

// Generic bench: build a tree-like structure, run queries, report recall.
template <class Tree>
[[nodiscard]] static Stats
bench_knn_tree(const std::vector<vec_t> &db, const std::vector<vec_t> &queries,
               std::size_t k, int repeats, double &recall_out)
{
    Tree t;
    t.build(db);

    std::vector<std::vector<std::uint32_t>> gt(queries.size());
    for (std::size_t i = 0; i < queries.size(); ++i)
        gt[i] = ground_truth(db, queries[i], k);

    std::vector<double> samples_ns;
    samples_ns.reserve(repeats);
    std::size_t total_hits = 0;
    std::size_t total_expected = 0;
    volatile std::size_t sink = 0;

    for (int r = 0; r < repeats; ++r) {
        std::size_t hits_this_run = 0;
        const double ns = time_ns([&] {
            for (std::size_t q = 0; q < queries.size(); ++q) {
                auto got = t.knn(queries[q], k);
                sink += got.size();
                if (r == 0) {
                    for (auto id : gt[q])
                        if (std::find(got.begin(), got.end(), id) != got.end()) ++hits_this_run;
                }
            }
        });
        samples_ns.push_back(ns);
        if (r == 0) {
            total_hits = hits_this_run;
            for (const auto &g : gt) total_expected += g.size();
        }
    }
    (void)sink;
    recall_out = total_expected ? static_cast<double>(total_hits) / static_cast<double>(total_expected) : 0.0;
    return summarize(std::move(samples_ns), queries.size());
}

// -----------------------------------------------------------------------------
// IVFFlat baseline (a la Faiss IndexIVFFlat, scaled down)
//
// Build: sample nlist random db points as centroids, assign each db point to
//        its nearest centroid -> a flat std::vector per centroid ("inverted
//        list").
// Query: distance to all nlist centroids, sort to find the top `nprobe`,
//        exhaustively scan their inverted lists for top-k.
//
// Compared to LC: same general family (compact partitioning) but without LC's
// triangle-inequality pruning between cluster ball and query ball. IVFFlat
// just probes the top-nprobe lists by centroid distance and accepts whatever
// lies in those.
// -----------------------------------------------------------------------------
struct ivfflat_t {
    std::vector<vec_t>                                     centroids;
    std::vector<std::vector<std::pair<vec_t, std::uint32_t>>> lists;

    void build(const std::vector<vec_t> &db, std::size_t nlist, std::uint32_t seed)
    {
        nlist = std::min(nlist, db.size());
        std::vector<std::size_t> idx(db.size());
        std::iota(idx.begin(), idx.end(), std::size_t{0});
        std::mt19937 rng(seed);
        std::shuffle(idx.begin(), idx.end(), rng);
        idx.resize(nlist);

        centroids.clear();
        centroids.reserve(nlist);
        for (auto i : idx) centroids.push_back(db[i]);

        lists.assign(nlist, {});
        euclid m;
        for (std::uint32_t i = 0; i < db.size(); ++i) {
            std::size_t best = 0;
            double best_d = std::numeric_limits<double>::infinity();
            for (std::size_t c = 0; c < nlist; ++c) {
                const double d = m(db[i], centroids[c]);
                if (d < best_d) { best_d = d; best = c; }
            }
            lists[best].emplace_back(db[i], i);
        }
    }

    [[nodiscard]] std::vector<std::uint32_t>
    knn(const vec_t &q, std::size_t k, std::size_t nprobe) const
    {
        euclid m;
        const std::size_t nlist = centroids.size();
        std::vector<std::pair<double, std::size_t>> cdists;
        cdists.reserve(nlist);
        for (std::size_t c = 0; c < nlist; ++c)
            cdists.emplace_back(m(q, centroids[c]), c);
        const std::size_t np = std::min(nprobe, nlist);
        std::partial_sort(cdists.begin(), cdists.begin() + np, cdists.end());

        std::vector<std::pair<double, std::uint32_t>> cands;
        for (std::size_t i = 0; i < np; ++i) {
            const auto &list = lists[cdists[i].second];
            for (const auto &[v, id] : list)
                cands.emplace_back(m(q, v), id);
        }
        const std::size_t take = std::min(k, cands.size());
        std::partial_sort(cands.begin(), cands.begin() + take, cands.end());
        std::vector<std::uint32_t> out;
        out.reserve(take);
        for (std::size_t i = 0; i < take; ++i) out.push_back(cands[i].second);
        return out;
    }
};

// IVFFlat bench helper. Reports throughput AND recall@k vs brute force.
[[nodiscard]] static Stats
bench_knn_ivfflat(const std::vector<vec_t> &db, const std::vector<vec_t> &queries,
                  std::size_t k, int repeats,
                  std::size_t nlist, std::size_t nprobe,
                  double &recall_out)
{
    ivfflat_t ivf;
    ivf.build(db, nlist, /*seed=*/42);

    // Ground truth.
    std::vector<std::vector<std::uint32_t>> gt(queries.size());
    {
        euclid m;
        for (std::size_t q = 0; q < queries.size(); ++q) {
            std::vector<std::pair<double, std::uint32_t>> bf;
            bf.reserve(db.size());
            for (std::uint32_t i = 0; i < db.size(); ++i)
                bf.emplace_back(m(queries[q], db[i]), i);
            std::partial_sort(bf.begin(), bf.begin() + std::min<std::size_t>(k, bf.size()), bf.end());
            for (std::size_t i = 0; i < std::min<std::size_t>(k, bf.size()); ++i)
                gt[q].push_back(bf[i].second);
        }
    }

    std::vector<double> samples_ns;
    samples_ns.reserve(repeats);
    std::size_t total_hits = 0;
    std::size_t total_expected = 0;
    volatile std::size_t sink = 0;

    for (int r = 0; r < repeats; ++r) {
        std::size_t hits_this_run = 0;
        const double ns = time_ns([&] {
            for (std::size_t q = 0; q < queries.size(); ++q) {
                auto got = ivf.knn(queries[q], k, nprobe);
                sink += got.size();
                if (r == 0) {
                    for (auto id : gt[q])
                        if (std::find(got.begin(), got.end(), id) != got.end()) ++hits_this_run;
                }
            }
        });
        samples_ns.push_back(ns);
        if (r == 0) {
            total_hits = hits_this_run;
            for (const auto &g : gt) total_expected += g.size();
        }
    }
    (void)sink;
    recall_out = total_expected ? static_cast<double>(total_hits) / static_cast<double>(total_expected) : 0.0;
    return summarize(std::move(samples_ns), queries.size());
}

// kNN LC, batched across worker threads via Index::batch_knn.
[[nodiscard]] static Stats
bench_knn_lc_parallel(const std::vector<vec_t> &db, const std::vector<vec_t> &queries,
                      std::size_t k, int repeats, unsigned nthreads)
{
    idx_t<> idx;
    for (std::uint32_t i = 0; i < db.size(); ++i) idx.insert(db[i], i);

    std::vector<double> samples_ns;
    samples_ns.reserve(repeats);
    volatile std::size_t sink = 0;
    for (int r = 0; r < repeats; ++r) {
        const double ns = time_ns([&] {
            auto res = idx.batch_knn(queries,
                                     static_cast<std::uint32_t>(db.size()),
                                     k, nthreads);
            for (auto &q : res) sink += q.results().size();
        });
        samples_ns.push_back(ns);
    }
    (void)sink;
    return summarize(std::move(samples_ns), queries.size());
}

// kNN brute-force baseline, multi-threaded. Splits the query set across
// hardware concurrency. The honest threaded reference for a real-world
// "is your fancy index worth it" comparison.
[[nodiscard]] static Stats
bench_knn_brute_parallel(const std::vector<vec_t> &db, const std::vector<vec_t> &queries,
                         std::size_t k, int repeats, unsigned nthreads)
{
    std::vector<double> samples_ns;
    samples_ns.reserve(repeats);
    std::vector<double> per_query_sums(queries.size(), 0.0);
    euclid m;

    auto worker = [&](std::size_t q0, std::size_t q1) {
        std::vector<std::pair<double, std::uint32_t>> heap;
        heap.reserve(db.size());
        for (std::size_t q = q0; q < q1; ++q) {
            heap.clear();
            const auto &qv = queries[q];
            for (std::uint32_t i = 0; i < db.size(); ++i)
                heap.emplace_back(m(qv, db[i]), i);
            std::partial_sort(heap.begin(),
                              heap.begin() + std::min<std::size_t>(k, heap.size()),
                              heap.end());
            double acc = 0.0;
            for (std::size_t i = 0; i < std::min<std::size_t>(k, heap.size()); ++i)
                acc += heap[i].first;
            per_query_sums[q] = acc;
        }
    };

    for (int r = 0; r < repeats; ++r) {
        const double ns = time_ns([&] {
            std::vector<std::thread> ts;
            ts.reserve(nthreads);
            const std::size_t Q = queries.size();
            const std::size_t chunk = (Q + nthreads - 1) / nthreads;
            for (unsigned t = 0; t < nthreads; ++t) {
                const std::size_t q0 = t * chunk;
                const std::size_t q1 = std::min(q0 + chunk, Q);
                if (q0 < q1) ts.emplace_back(worker, q0, q1);
            }
            for (auto &t : ts) t.join();
        });
        samples_ns.push_back(ns);
    }
    volatile double sink = 0.0;
    for (double v : per_query_sums) sink += v;
    (void)sink;
    return summarize(std::move(samples_ns), queries.size());
}

// HNSW (approximate) baseline via vendored hnswlib. Returns throughput and
// fills `recall_out` with the recall@k against brute-force ground truth.
[[nodiscard]] static Stats
bench_knn_hnsw(const std::vector<vec_t> &db, const std::vector<vec_t> &queries,
               std::size_t k, int repeats,
               std::size_t M, std::size_t ef_construction, std::size_t ef_search,
               double &recall_out)
{
    // hnswlib's L2Space expects a flat float buffer.
    std::vector<float> flat(db.size() * kD);
    for (std::size_t i = 0; i < db.size(); ++i)
        for (std::size_t j = 0; j < kD; ++j)
            flat[i * kD + j] = static_cast<float>(db[i][j]);

    hnswlib::L2Space space(kD);
    hnswlib::HierarchicalNSW<float> index(&space, db.size(), M, ef_construction);
    for (std::size_t i = 0; i < db.size(); ++i)
        index.addPoint(flat.data() + i * kD, static_cast<hnswlib::labeltype>(i));
    index.setEf(ef_search);

    std::vector<float> qflat(queries.size() * kD);
    for (std::size_t i = 0; i < queries.size(); ++i)
        for (std::size_t j = 0; j < kD; ++j)
            qflat[i * kD + j] = static_cast<float>(queries[i][j]);

    // Ground truth ids for recall calculation (brute force).
    std::vector<std::vector<std::uint32_t>> gt(queries.size());
    {
        euclid m;
        for (std::size_t q = 0; q < queries.size(); ++q) {
            std::vector<std::pair<double, std::uint32_t>> heap;
            heap.reserve(db.size());
            for (std::uint32_t i = 0; i < db.size(); ++i)
                heap.emplace_back(m(queries[q], db[i]), i);
            std::partial_sort(heap.begin(), heap.begin() + std::min<std::size_t>(k, heap.size()), heap.end());
            for (std::size_t i = 0; i < std::min<std::size_t>(k, heap.size()); ++i)
                gt[q].push_back(heap[i].second);
        }
    }

    std::vector<double> samples_ns;
    samples_ns.reserve(repeats);
    std::size_t total_hits = 0;
    std::size_t total_expected = 0;

    for (int r = 0; r < repeats; ++r) {
        std::size_t hits_this_run = 0;
        const double ns = time_ns([&] {
            for (std::size_t q = 0; q < queries.size(); ++q) {
                auto res = index.searchKnn(qflat.data() + q * kD, k);
                if (r == 0) {
                    std::vector<std::uint32_t> got;
                    while (!res.empty()) { got.push_back(static_cast<std::uint32_t>(res.top().second)); res.pop(); }
                    for (auto id : gt[q])
                        if (std::find(got.begin(), got.end(), id) != got.end()) ++hits_this_run;
                }
            }
        });
        samples_ns.push_back(ns);
        if (r == 0) {
            total_hits += hits_this_run;
            for (auto &g : gt) total_expected += g.size();
        }
    }
    recall_out = total_expected ? static_cast<double>(total_hits) / static_cast<double>(total_expected) : 0.0;
    return summarize(std::move(samples_ns), queries.size());
}

// Range search (LC), radius chosen so each query returns roughly `target_count` neighbors.
[[nodiscard]] static Stats
bench_range(const std::vector<vec_t> &db, const std::vector<vec_t> &queries,
            double radius, int repeats)
{
    idx_t<> idx;
    for (std::uint32_t i = 0; i < db.size(); ++i) idx.insert(db[i], i);

    std::vector<double> samples_ns;
    samples_ns.reserve(repeats);
    volatile std::size_t sink = 0;
    for (int r = 0; r < repeats; ++r) {
        const double ns = time_ns([&] {
            for (std::uint32_t q = 0; q < queries.size(); ++q) {
                auto res = idx.range_search(queries[q], static_cast<std::uint32_t>(db.size() + q), radius);
                sink += res.results().size();
            }
        });
        samples_ns.push_back(ns);
    }
    (void)sink;
    return summarize(std::move(samples_ns), queries.size());
}

// -----------------------------------------------------------------------------
// Bucket-size sweep
//
// Phase 5.1: walk bucket_size across {16, 20, 32, 64, 128} on a fixed
// workload and report insert + kNN throughput for each. The literature
// (Chavez et al., "Engineering efficient metric indexes") nominates 20 as
// a good default - this surfaces the empirical optimum for our test data.
// -----------------------------------------------------------------------------

template <std::size_t bucket_sz>
[[nodiscard]] static std::pair<Stats, Stats>
sweep_bucket(const std::vector<vec_t> &db, const std::vector<vec_t> &queries, std::size_t k)
{
    // `overflow` is currently a no-op constant in the LC algorithm but is
    // still asserted >= bucket_size. Use 4x bucket_size to keep the
    // static_assert happy across the sweep.
    using idx_b = metric::listofclusters<vec_t, euclid, bucket_sz, 4 * bucket_sz>;

    // Insert phase
    std::vector<double> ins_samples;
    ins_samples.reserve(3);
    for (int r = 0; r < 3; ++r) {
        idx_b idx;
        const double ns = time_ns([&] {
            for (std::uint32_t i = 0; i < db.size(); ++i)
                idx.insert(db[i], i);
        });
        ins_samples.push_back(ns);
    }
    Stats ins = summarize(std::move(ins_samples), db.size());

    // kNN phase
    idx_b idx;
    for (std::uint32_t i = 0; i < db.size(); ++i) idx.insert(db[i], i);

    std::vector<double> knn_samples;
    knn_samples.reserve(5);
    volatile std::size_t sink = 0;
    for (int r = 0; r < 5; ++r) {
        const double ns = time_ns([&] {
            for (std::uint32_t q = 0; q < queries.size(); ++q) {
                auto res = idx.knn_search(queries[q], static_cast<std::uint32_t>(db.size() + q), k);
                sink += res.results().size();
            }
        });
        knn_samples.push_back(ns);
    }
    (void)sink;
    Stats knn = summarize(std::move(knn_samples), queries.size());

    return {ins, knn};
}

static void print_sweep_row(std::size_t b, const Stats &ins, const Stats &knn)
{
    std::printf("  %5zu  %12.3f  %12.3f\n",
                b, ins.per_op_ns / 1000.0, knn.per_op_ns / 1000.0);
}

// Recall@k of LC vs brute force - sanity number, not a benchmark.
[[nodiscard]] static double
recall_at_k(const std::vector<vec_t> &db, const std::vector<vec_t> &queries, std::size_t k)
{
    idx_t<> idx;
    for (std::uint32_t i = 0; i < db.size(); ++i) idx.insert(db[i], i);

    euclid m;
    std::size_t total_hits = 0;
    std::size_t total_expected = 0;

    for (std::uint32_t q = 0; q < queries.size(); ++q) {
        std::vector<std::pair<double, std::uint32_t>> bf;
        bf.reserve(db.size());
        for (std::uint32_t i = 0; i < db.size(); ++i)
            bf.emplace_back(m(queries[q], db[i]), i);
        std::partial_sort(bf.begin(), bf.begin() + std::min<std::size_t>(k, bf.size()), bf.end());

        std::vector<std::uint32_t> expected;
        for (std::size_t i = 0; i < std::min<std::size_t>(k, bf.size()); ++i)
            expected.push_back(bf[i].second);

        auto res = idx.knn_search(queries[q], static_cast<std::uint32_t>(db.size() + q), k);
        std::vector<std::uint32_t> got;
        for (const auto &r : res.results()) got.push_back(r.id());

        std::sort(expected.begin(), expected.end());
        std::sort(got.begin(), got.end());

        for (auto id : expected) {
            if (std::find(got.begin(), got.end(), id) != got.end()) ++total_hits;
            ++total_expected;
        }
    }
    return total_expected ? static_cast<double>(total_hits) / static_cast<double>(total_expected) : 0.0;
}

// -----------------------------------------------------------------------------
// main
// -----------------------------------------------------------------------------

int main(int argc, char **argv)
{
    std::size_t N = (argc > 1) ? static_cast<std::size_t>(std::atoll(argv[1])) : 10000;
    std::size_t D_arg = (argc > 2) ? static_cast<std::size_t>(std::atoll(argv[2])) : kD;
    std::size_t Q = (argc > 3) ? static_cast<std::size_t>(std::atoll(argv[3])) : 200;
    std::size_t k = 10;
    // Set GEN=clustered (env var or 4th positional) to use the mixture-of-
    // Gaussians generator. The uniform case is LC's worst case: LC's
    // triangle-inequality pruning needs structure to be effective.
    std::string gen = (argc > 4) ? std::string(argv[4]) : std::string(std::getenv("GEN") ? std::getenv("GEN") : "uniform");

    if (D_arg != kD) {
        std::fprintf(stderr, "warning: D=%zu requested but binary was built with BENCH_D=%zu; recompile with -DBENCH_D=%zu\n",
                     D_arg, kD, D_arg);
    }

    std::printf("# liblistofclusters microbenchmarks\n");
    std::printf("#   dataset: N=%zu, D=%zu (compile-time, std::array payload), gen=%s\n", N, kD, gen.c_str());
    std::printf("#   queries: Q=%zu, k=%zu\n", Q, k);
    std::printf("#   index:   bucket_size=20, overflow=80\n");
    std::printf("#\n");

    std::vector<vec_t> db, queries;
    if (gen == "clustered") {
        // 64 clusters of ~N/64 points each, gaussian noise tight enough that
        // points within a cluster are clearly closer to each other than to
        // any other cluster (sigma << inter-cluster distance).
        db      = make_clustered_dataset(N, /*n_clusters=*/64, /*sigma=*/0.05, /*seed=*/42);
        // Queries drawn from the SAME distribution - the realistic case.
        queries = make_clustered_dataset(Q, /*n_clusters=*/64, /*sigma=*/0.05, /*seed=*/9999);
    } else {
        db      = make_dataset(N, /*seed=*/42);
        queries = make_dataset(Q, /*seed=*/9999);
    }

    print_header();

    // 1. Build (insert all N), incremental.
    const Stats s_insert = bench_insert(db, /*repeats=*/5);
    print_row("insert (LC, incremental)", N, s_insert);

    // 1b. Build via bulk_build (canonical static LC construction).
    const Stats s_bulk = bench_bulk_build(db, /*repeats=*/3);
    print_row("insert (LC, bulk_build)", N, s_bulk);

    // 2. kNN throughput (LC, incrementally built), queries only.
    const Stats s_knn_lc = bench_knn(db, queries, k, /*repeats=*/5);
    print_row("knn k=10 (LC incr, 1T)", Q, s_knn_lc);

    // 2a. kNN throughput against a bulk-built index. Same queries, same N,
    //     same metric - the only difference is the cluster shapes from the
    //     two build strategies.
    const Stats s_knn_bulk = bench_knn_bulk(db, queries, k, /*repeats=*/5);
    print_row("knn k=10 (LC bulk, 1T)", Q, s_knn_bulk);

    // 2c. kNN with the explicit NEON/AVX2 Euclidean. Same incremental build,
    //     only the metric functor differs from row 2.
    const Stats s_knn_simd = bench_knn_simd(db, queries, k, /*repeats=*/5);
    print_row("knn k=10 (LC incr, SIMD, 1T)", Q, s_knn_simd);

    // 2b. kNN throughput (LC) via batch_knn parallelized across hw threads.
    const unsigned nthreads = std::max(1u, std::thread::hardware_concurrency());
    const Stats s_knn_lc_par = bench_knn_lc_parallel(db, queries, k, /*repeats=*/5, nthreads);
    {
        char lbl[64];
        std::snprintf(lbl, sizeof lbl, "knn k=10 (LC batch, %uT)", nthreads);
        print_row(lbl, Q, s_knn_lc_par);
    }

    // 3. kNN throughput (brute force baseline). Brute force has no build phase.
    const Stats s_knn_bf = bench_knn_brute(db, queries, k, /*repeats=*/3);
    print_row("knn k=10 (brute force, 1T)", Q, s_knn_bf);

    // 3b. Multi-threaded brute force. The real-world baseline once threads
    //     are on the table.
    const Stats s_knn_bf_par = bench_knn_brute_parallel(db, queries, k, /*repeats=*/3, nthreads);
    {
        char lbl[64];
        std::snprintf(lbl, sizeof lbl, "knn k=10 (brute force, %uT)", nthreads);
        print_row(lbl, Q, s_knn_bf_par);
    }

    // 3c. HNSW approximate baseline. Reports throughput AND recall@k so the
    //     speed/recall tradeoff is on the page.
    double hnsw_recall = 0.0;
    const Stats s_knn_hnsw = bench_knn_hnsw(db, queries, k, /*repeats=*/3,
        /*M=*/16, /*ef_construction=*/200, /*ef_search=*/50, hnsw_recall);
    print_row("knn k=10 (HNSW, approx)", Q, s_knn_hnsw);

    // 3e. KD-tree exact baseline (low-D Euclidean specialist).
    double kd_recall = 0.0;
    const Stats s_knn_kd = bench_knn_tree<kdtree_t>(db, queries, k, /*repeats=*/3, kd_recall);
    print_row("knn k=10 (KD-tree, 1T)", Q, s_knn_kd);

    // 3f. Ball-tree exact baseline (general metric-space, like LC).
    double bt_recall = 0.0;
    const Stats s_knn_bt = bench_knn_tree<balltree_t>(db, queries, k, /*repeats=*/3, bt_recall);
    print_row("knn k=10 (ball tree, 1T)", Q, s_knn_bt);

    // 3d. IVFFlat approximate baseline (inverted-file / k-cell). nlist =
    //     sqrt(N) (the Faiss default), nprobe = nlist / 10 (a moderate
    //     speed/recall point on the IVFFlat tradeoff curve).
    const std::size_t ivf_nlist  = std::max<std::size_t>(
        static_cast<std::size_t>(std::sqrt(static_cast<double>(N))), 4);
    const std::size_t ivf_nprobe = std::max<std::size_t>(ivf_nlist / 10, 1);
    double ivf_recall = 0.0;
    const Stats s_knn_ivf = bench_knn_ivfflat(db, queries, k, /*repeats=*/3,
        ivf_nlist, ivf_nprobe, ivf_recall);
    {
        char lbl[80];
        std::snprintf(lbl, sizeof lbl, "knn k=10 (IVFFlat nlist=%zu nprobe=%zu)",
                      ivf_nlist, ivf_nprobe);
        print_row(lbl, Q, s_knn_ivf);
    }

    // 4. Range search (LC). Radius picked to yield ~5-15 results on uniform data.
    const double radius = 0.4;
    const Stats s_range_lc = bench_range(db, queries, radius, /*repeats=*/5);
    print_row("range r=0.4 (LC, query only)", Q, s_range_lc);

    // 5. Recall@k sanity check (cheap, on a subset of queries).
    const double r = recall_at_k(db,
        std::vector<vec_t>(queries.begin(), queries.begin() + std::min<std::size_t>(50, Q)), k);

    // Honest build-amortization summary. LC pays a one-time build cost
    // (insert * N) and saves time vs brute force on each subsequent query
    // IF its per-query time is lower. Print build-time, per-query savings,
    // and the break-even query count.
    const double build_total_us  = (s_insert.per_op_ns * static_cast<double>(N)) / 1000.0;
    const double saved_per_q_us  = (s_knn_bf.per_op_ns - s_knn_lc.per_op_ns) / 1000.0;
    std::printf("\nbuild amortization (kNN, LC vs single-thread brute):\n");
    std::printf("  one-time build:        %12.1f us  (insert * N)\n", build_total_us);
    std::printf("  per-query brute (1T):  %12.3f us\n", s_knn_bf.per_op_ns / 1000.0);
    std::printf("  per-query brute (%uT): %12.3f us\n", nthreads, s_knn_bf_par.per_op_ns / 1000.0);
    std::printf("  per-query LC:          %12.3f us\n", s_knn_lc.per_op_ns / 1000.0);
    std::printf("  per-query HNSW (approx): %10.3f us  (recall@%zu = %.3f)\n",
                s_knn_hnsw.per_op_ns / 1000.0, k, hnsw_recall);
    std::printf("  per-query KD-tree:     %12.3f us  (recall@%zu = %.3f)\n",
                s_knn_kd.per_op_ns / 1000.0, k, kd_recall);
    std::printf("  per-query ball tree:   %12.3f us  (recall@%zu = %.3f)\n",
                s_knn_bt.per_op_ns / 1000.0, k, bt_recall);
    std::printf("  per-query IVFFlat:     %12.3f us  (recall@%zu = %.3f, nlist=%zu nprobe=%zu)\n",
                s_knn_ivf.per_op_ns / 1000.0, k, ivf_recall, ivf_nlist, ivf_nprobe);
    if (saved_per_q_us > 0.0) {
        const double breakeven = build_total_us / saved_per_q_us;
        std::printf("  LC saves per query:    %12.3f us\n", saved_per_q_us);
        std::printf("  break-even at queries: %12.0f\n", breakeven);
    } else {
        std::printf("  LC saves per query:    %12.3f us  (NEGATIVE - brute force wins)\n", saved_per_q_us);
        std::printf("  break-even at queries: never at this N\n");
    }

    std::printf("\nrecall@%zu vs brute force (50 queries): %.3f\n", k, r);

    // Bucket-size sweep on the same workload.
    std::printf("\nbucket_size sweep (same dataset, %s):\n", gen.c_str());
    std::printf("  %5s  %12s  %12s\n", "bucket", "insert (us)", "knn (us)");
    {
        auto [ins, knn] = sweep_bucket<16>(db, queries, k);  print_sweep_row(16, ins, knn);
    }
    {
        auto [ins, knn] = sweep_bucket<20>(db, queries, k);  print_sweep_row(20, ins, knn);
    }
    {
        auto [ins, knn] = sweep_bucket<32>(db, queries, k);  print_sweep_row(32, ins, knn);
    }
    {
        auto [ins, knn] = sweep_bucket<64>(db, queries, k);  print_sweep_row(64, ins, knn);
    }
    {
        auto [ins, knn] = sweep_bucket<128>(db, queries, k); print_sweep_row(128, ins, knn);
    }

    return 0;
}
