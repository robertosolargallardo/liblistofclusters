// Self-contained microbenchmarks for liblistofclusters.
//
// No external benchmark framework - just std::chrono with median-of-runs
// reporting. Build with -O3, no sanitizers (the Makefile in this directory
// takes care of the flags). The point is to establish baseline numbers
// before phase 4 (threading) and phase 5 (perf pass) so wins can be
// measured against a known reference.
//
// Usage:
//   make bench         # build and run
//   ./bench_main       # run directly
//   ./bench_main N D   # override default workload size
//
// What's reported:
//   - Insert throughput  (LC build cost)
//   - kNN throughput     (LC vs brute force baseline)
//   - Range throughput   (LC vs brute force baseline)
//   - Recall@k vs brute  (sanity check that the LC results match)

#include <listofclusters/listofclusters.hh>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using vec_t  = std::vector<double>;
using clock_t_ = std::chrono::steady_clock;

struct euclid {
    [[nodiscard]] double operator()(const vec_t &a, const vec_t &b) const noexcept
    {
        double s = 0.0;
        for (std::size_t i = 0; i < a.size(); ++i) {
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
make_dataset(std::size_t N, std::size_t D, std::uint32_t seed)
{
    std::vector<vec_t> db(N, vec_t(D));
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
make_clustered_dataset(std::size_t N, std::size_t D, std::size_t n_clusters,
                       double noise_sigma, std::uint32_t seed)
{
    std::vector<vec_t> centers(n_clusters, vec_t(D));
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    for (auto &c : centers) for (auto &x : c) x = u(rng);

    std::vector<vec_t> db(N, vec_t(D));
    std::uniform_int_distribution<std::size_t> pick(0, n_clusters - 1);
    std::normal_distribution<double> noise(0.0, noise_sigma);
    for (auto &v : db) {
        const auto &c = centers[pick(rng)];
        for (std::size_t i = 0; i < D; ++i) v[i] = c[i] + noise(rng);
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
    std::size_t D = (argc > 2) ? static_cast<std::size_t>(std::atoll(argv[2])) : 8;
    std::size_t Q = (argc > 3) ? static_cast<std::size_t>(std::atoll(argv[3])) : 200;
    std::size_t k = 10;
    // Set GEN=clustered (env var or 4th positional) to use the mixture-of-
    // Gaussians generator. The uniform case is LC's worst case: LC's
    // triangle-inequality pruning needs structure to be effective.
    std::string gen = (argc > 4) ? std::string(argv[4]) : std::string(std::getenv("GEN") ? std::getenv("GEN") : "uniform");

    std::printf("# liblistofclusters microbenchmarks\n");
    std::printf("#   dataset: N=%zu, D=%zu, gen=%s\n", N, D, gen.c_str());
    std::printf("#   queries: Q=%zu, k=%zu\n", Q, k);
    std::printf("#   index:   bucket_size=20, overflow=80\n");
    std::printf("#\n");

    std::vector<vec_t> db, queries;
    if (gen == "clustered") {
        // 64 clusters of ~N/64 points each, gaussian noise tight enough that
        // points within a cluster are clearly closer to each other than to
        // any other cluster (sigma << inter-cluster distance).
        db      = make_clustered_dataset(N, D, /*n_clusters=*/64, /*sigma=*/0.05, /*seed=*/42);
        // Queries drawn from the SAME distribution - the realistic case.
        queries = make_clustered_dataset(Q, D, /*n_clusters=*/64, /*sigma=*/0.05, /*seed=*/9999);
    } else {
        db      = make_dataset(N, D, /*seed=*/42);
        queries = make_dataset(Q, D, /*seed=*/9999);
    }

    print_header();

    // 1. Build (insert all N).
    const Stats s_insert = bench_insert(db, /*repeats=*/5);
    print_row("insert (LC)", N, s_insert);

    // 2. kNN throughput (LC), queries only - excludes build.
    const Stats s_knn_lc = bench_knn(db, queries, k, /*repeats=*/5);
    print_row("knn k=10 (LC, query only)", Q, s_knn_lc);

    // 3. kNN throughput (brute force baseline). Brute force has no build phase.
    const Stats s_knn_bf = bench_knn_brute(db, queries, k, /*repeats=*/3);
    print_row("knn k=10 (brute force)", Q, s_knn_bf);

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
    std::printf("\nbuild amortization (kNN):\n");
    std::printf("  one-time build:        %12.1f us  (insert * N)\n", build_total_us);
    std::printf("  per-query brute:       %12.3f us\n", s_knn_bf.per_op_ns / 1000.0);
    std::printf("  per-query LC:          %12.3f us\n", s_knn_lc.per_op_ns / 1000.0);
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
