#include <listofclusters/listofclusters.hh>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

using vec_t = std::vector<double>;

double euclid(vec_t a, vec_t b)
{
    double s = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const double d = a[i] - b[i];
        s += d * d;
    }
    return std::sqrt(s);
}

static void test_build_and_knn()
{
    // Index parameters: bucket_size=4, overflow=10.
    // We deliberately stay below the supercluster overflow threshold (N < overflow)
    // because the cascading-overflow path in listofclusters::insert hits pathological
    // recursion - tracked separately as C-5. This test exercises the no-overflow path.
    metric::listofclusters<vec_t, euclid, 4, 10> idx;

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> u(-1.0, 1.0);

    constexpr std::uint32_t N = 9U;
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
    for (std::uint32_t i = 0; i < N; ++i) brute.emplace_back(euclid(q, db[i]), i);
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
    metric::listofclusters<vec_t, euclid, 4, 10> idx;

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
        if (euclid(q, db[i]) <= r) brute_ids.push_back(i);

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
    metric::listofclusters<vec_t, euclid, 4, 10> idx;

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

int main()
{
    std::cout << "liblistofclusters smoke tests:\n";
    test_build_and_knn();
    test_range_search();
    test_equal_distance_strict_weak_ordering();
    std::cout << "all tests passed\n";
    return 0;
}
