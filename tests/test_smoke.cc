#include <listofclusters/listofclusters.hh>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

using vec_t = std::vector<double>;

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

// Brute-force ground truth (used by the correctness test below).
[[nodiscard]] static double bf_dist(const vec_t &a, const vec_t &b) noexcept
{
    return euclid{}(a, b);
}

using idx_t = metric::listofclusters<vec_t, euclid, 4, 10>;

static void test_build_and_knn()
{
    idx_t idx;

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> u(-1.0, 1.0);

    constexpr std::uint32_t N = 64U;
    constexpr std::size_t D = 8U;

    std::vector<vec_t> db(N);
    for (std::uint32_t i = 0; i < N; ++i) {
        db[i].resize(D);
        for (std::size_t j = 0; j < D; ++j) db[i][j] = u(rng);
        idx.insert(db[i], i);
    }

    const vec_t q = db[7];
    const std::uint32_t qid = N;
    const std::size_t k = 5;

    auto results = idx.knn_search(q, qid, k);

    std::vector<std::pair<double, std::uint32_t>> brute;
    brute.reserve(N);
    for (std::uint32_t i = 0; i < N; ++i) brute.emplace_back(bf_dist(q, db[i]), i);
    std::sort(brute.begin(), brute.end());

    const auto &got = results.results();
    assert(got.size() >= k);

    std::vector<std::uint32_t> got_ids;
    for (const auto &r : got) got_ids.push_back(r.id());
    std::sort(got_ids.begin(), got_ids.end());

    std::vector<std::uint32_t> expected_ids;
    for (std::size_t i = 0; i < k; ++i) expected_ids.push_back(brute[i].second);
    std::sort(expected_ids.begin(), expected_ids.end());

    for (auto id : expected_ids) {
        assert(std::find(got_ids.begin(), got_ids.end(), id) != got_ids.end()
               && "knn_search missed an expected nearest neighbor");
    }
    std::cout << "  test_build_and_knn: OK (" << got.size() << " results, " << k << " expected)\n";
}

static void test_range_search()
{
    idx_t idx;

    constexpr std::uint32_t N = 50U;
    constexpr std::size_t D = 4U;

    std::vector<vec_t> db(N);
    std::mt19937 rng(123);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    for (std::uint32_t i = 0; i < N; ++i) {
        db[i].resize(D);
        for (std::size_t j = 0; j < D; ++j) db[i][j] = u(rng);
        idx.insert(db[i], i);
    }

    const vec_t q = db[0];
    const double r = 0.5;

    auto results = idx.range_search(q, N, r);

    std::vector<std::uint32_t> brute_ids;
    for (std::uint32_t i = 0; i < N; ++i)
        if (bf_dist(q, db[i]) <= r) brute_ids.push_back(i);

    std::vector<std::uint32_t> got_ids;
    for (const auto &x : results.results()) got_ids.push_back(x.id());

    std::sort(brute_ids.begin(), brute_ids.end());
    std::sort(got_ids.begin(), got_ids.end());

    for (auto id : brute_ids) {
        assert(std::find(got_ids.begin(), got_ids.end(), id) != got_ids.end()
               && "range_search missed a point inside the radius");
    }
    std::cout << "  test_range_search: OK (" << got_ids.size() << " found, " << brute_ids.size() << " expected)\n";
}

static void test_equal_distance_strict_weak_ordering()
{
    idx_t idx;

    const vec_t origin = {0.0, 0.0};
    idx.insert(origin, 0U);
    idx.insert(vec_t{1.0,  0.0}, 1U);
    idx.insert(vec_t{0.0,  1.0}, 2U);
    idx.insert(vec_t{-1.0, 0.0}, 3U);
    idx.insert(vec_t{0.0, -1.0}, 4U);

    auto results = idx.knn_search(origin, 99U, 4);
    assert(results.results().size() == 4U
           && "knn_search dropped objects with equal distances (C-1 regression)");
    std::cout << "  test_equal_distance_strict_weak_ordering: OK\n";
}

// Regression test for C-5 and C-6: the previous incremental insert
// implementation had a "supercluster" splitting scheme not described in the
// LC paper. It (a) recursed deep enough to stack-overflow at N=64 and
// (b) diverged into an unbounded queue cycle at N>=100 even after the
// recursion was made iterative. Rewriting around the canonical fixed-
// bucket-size algorithm (Chavez & Navarro 2005, Section 5.3) fixed both:
// inserts are now O(|list|) per element with deterministic termination.
static void test_large_n_no_crash()
{
    idx_t idx;

    constexpr std::uint32_t N = 500U;
    constexpr std::size_t D = 8U;

    std::vector<vec_t> db(N);
    std::mt19937 rng(2025);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    for (std::uint32_t i = 0; i < N; ++i) {
        db[i].resize(D);
        for (std::size_t j = 0; j < D; ++j) db[i][j] = u(rng);
        idx.insert(db[i], i);
    }

    constexpr int probes = 32;
    std::uniform_int_distribution<std::uint32_t> pick(0, N - 1);
    int hits = 0;
    for (int p = 0; p < probes; ++p) {
        const std::uint32_t qid = pick(rng);
        auto results = idx.knn_search(db[qid], N + p, 1);
        for (const auto &r : results.results()) {
            if (r.id() == qid) { ++hits; break; }
        }
    }
    assert(hits == probes && "C-5/C-6 regression: some inserted objects not retrievable");
    std::cout << "  test_large_n_no_crash: OK (" << probes
              << " probes, all found in N=" << N << ")\n";
}

// Correctness vs brute force: for several queries the LC knn result must
// contain exactly the k nearest neighbors according to a full O(N) scan.
static void test_knn_matches_brute_force()
{
    idx_t idx;

    constexpr std::uint32_t N = 300U;
    constexpr std::size_t D = 6U;
    constexpr std::size_t k = 5U;

    std::vector<vec_t> db(N);
    std::mt19937 rng(11);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    for (std::uint32_t i = 0; i < N; ++i) {
        db[i].resize(D);
        for (std::size_t j = 0; j < D; ++j) db[i][j] = u(rng);
        idx.insert(db[i], i);
    }

    constexpr int queries = 10;
    int total_recall_hits = 0;
    for (int q = 0; q < queries; ++q) {
        vec_t query(D);
        for (std::size_t j = 0; j < D; ++j) query[j] = u(rng);
        const std::uint32_t qid = N + q;

        // Brute force ground truth.
        std::vector<std::pair<double, std::uint32_t>> bf;
        bf.reserve(N);
        for (std::uint32_t i = 0; i < N; ++i) bf.emplace_back(bf_dist(query, db[i]), i);
        std::sort(bf.begin(), bf.end());

        std::vector<std::uint32_t> expected;
        for (std::size_t i = 0; i < k; ++i) expected.push_back(bf[i].second);

        auto results = idx.knn_search(query, qid, k);
        std::vector<std::uint32_t> got;
        for (const auto &r : results.results()) got.push_back(r.id());

        std::sort(expected.begin(), expected.end());
        std::sort(got.begin(), got.end());

        for (auto id : expected) {
            if (std::find(got.begin(), got.end(), id) != got.end()) ++total_recall_hits;
        }
    }
    const int total_expected = queries * static_cast<int>(k);
    assert(total_recall_hits == total_expected && "LC knn missed brute-force neighbors");
    std::cout << "  test_knn_matches_brute_force: OK (" << total_recall_hits
              << "/" << total_expected << " recall over " << queries
              << " queries, N=" << N << " k=" << k << ")\n";
}

int main()
{
    std::cout << "liblistofclusters smoke tests:\n";
    test_build_and_knn();
    test_range_search();
    test_equal_distance_strict_weak_ordering();
    test_large_n_no_crash();
    test_knn_matches_brute_force();
    std::cout << "all tests passed\n";
    return 0;
}
