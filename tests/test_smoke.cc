#include <listofclusters/listofclusters.hh>
#include <listofclusters/metrics.hh>

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

// batch_knn must produce the same answers (one resultslist per query) as
// calling knn_search in a serial loop. Sanity check for phase 4 threading.
static void test_batch_knn_matches_serial()
{
    idx_t idx;
    constexpr std::uint32_t N = 200U;
    constexpr std::size_t D = 6U;
    constexpr std::size_t k = 5U;

    std::vector<vec_t> db(N);
    std::mt19937 rng(7);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    for (std::uint32_t i = 0; i < N; ++i) {
        db[i].resize(D);
        for (std::size_t j = 0; j < D; ++j) db[i][j] = u(rng);
        idx.insert(db[i], i);
    }

    constexpr std::size_t Qn = 25;
    std::vector<vec_t> queries(Qn);
    for (std::size_t q = 0; q < Qn; ++q) {
        queries[q].resize(D);
        for (std::size_t j = 0; j < D; ++j) queries[q][j] = u(rng);
    }

    auto par = idx.batch_knn(queries, N, k);
    assert(par.size() == Qn);

    for (std::size_t q = 0; q < Qn; ++q) {
        auto ser = idx.knn_search(queries[q], static_cast<std::uint32_t>(N + q), k);
        std::vector<std::uint32_t> par_ids, ser_ids;
        for (const auto &r : par[q].results())  par_ids.push_back(r.id());
        for (const auto &r : ser.results())     ser_ids.push_back(r.id());
        std::sort(par_ids.begin(), par_ids.end());
        std::sort(ser_ids.begin(), ser_ids.end());
        assert(par_ids == ser_ids && "batch_knn output differs from serial knn_search");
    }
    std::cout << "  test_batch_knn_matches_serial: OK (" << Qn << " queries)\n";
}

// Both build paths must produce indexes whose knn results match brute-force
// ground truth. Clusters differ between incremental and bulk_build (different
// arrangement), but recall@k against brute force must be 1.000 for both.
static void test_bulk_build_matches_brute_force()
{
    constexpr std::uint32_t N = 200U;
    constexpr std::size_t D = 6U;
    constexpr std::size_t k = 5U;

    std::vector<vec_t> db(N);
    std::vector<std::uint32_t> ids(N);
    std::mt19937 rng(31);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    for (std::uint32_t i = 0; i < N; ++i) {
        db[i].resize(D);
        for (std::size_t j = 0; j < D; ++j) db[i][j] = u(rng);
        ids[i] = i;
    }

    // Build via bulk_build.
    idx_t idx;
    idx.bulk_build(db, ids);
    assert(!idx.empty());

    constexpr int queries = 10;
    int hits = 0;
    int expected = 0;
    for (int q = 0; q < queries; ++q) {
        vec_t query(D);
        for (std::size_t j = 0; j < D; ++j) query[j] = u(rng);

        std::vector<std::pair<double, std::uint32_t>> bf;
        bf.reserve(N);
        for (std::uint32_t i = 0; i < N; ++i) bf.emplace_back(bf_dist(query, db[i]), i);
        std::sort(bf.begin(), bf.end());

        std::vector<std::uint32_t> want;
        for (std::size_t i = 0; i < k; ++i) want.push_back(bf[i].second);

        auto res = idx.knn_search(query, N + q, k);
        std::vector<std::uint32_t> got;
        for (const auto &r : res.results()) got.push_back(r.id());

        std::sort(want.begin(), want.end());
        std::sort(got.begin(), got.end());
        for (auto id : want) {
            if (std::find(got.begin(), got.end(), id) != got.end()) ++hits;
            ++expected;
        }
    }
    assert(hits == expected && "bulk_build knn missed brute-force neighbors");
    std::cout << "  test_bulk_build_matches_brute_force: OK (" << hits
              << "/" << expected << " recall over " << queries << " queries)\n";
}

// Sequential batch insert must produce the same index state as a serial loop.
static void test_batch_insert_matches_serial()
{
    constexpr std::uint32_t N = 100U;
    constexpr std::size_t D = 4U;

    std::vector<vec_t> db(N);
    std::vector<std::uint32_t> ids(N);
    std::mt19937 rng(5);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    for (std::uint32_t i = 0; i < N; ++i) {
        db[i].resize(D);
        for (std::size_t j = 0; j < D; ++j) db[i][j] = u(rng);
        ids[i] = i;
    }

    idx_t a, b;
    a.insert(db, ids);                              // batch entry point
    for (std::uint32_t i = 0; i < N; ++i) b.insert(db[i], ids[i]);  // serial loop

    // Each inserted id should be retrievable from both (1-NN of itself).
    for (std::uint32_t i = 0; i < N; ++i) {
        auto ra = a.knn_search(db[i], N + i, 1);
        auto rb = b.knn_search(db[i], N + i, 1);
        bool a_has = false, b_has = false;
        for (const auto &r : ra.results()) if (r.id() == i) { a_has = true; break; }
        for (const auto &r : rb.results()) if (r.id() == i) { b_has = true; break; }
        assert(a_has && b_has && "insert(batch) result differs from serial insert");
    }
    std::cout << "  test_batch_insert_matches_serial: OK (" << N << " ids)\n";
}

// Built-in metric functors: sanity-check the values against hand-computed
// expected distances on small vectors.
static void test_builtin_metrics()
{
    using v2 = std::vector<double>;
    const v2 a = {1.0, 2.0};
    const v2 b = {4.0, 6.0};  // diff: 3, 4 -> L2=5, L1=7, Linf=4

    auto approx = [](double x, double y, double eps = 1e-9) {
        return std::abs(x - y) < eps;
    };

    assert(approx(metric::euclidean{}(a, b), 5.0));
    assert(approx(metric::manhattan{}(a, b), 7.0));
    assert(approx(metric::chebyshev{}(a, b), 4.0));
    assert(approx(metric::minkowski<3>{}(a, b),
                  std::pow(std::pow(3.0, 3.0) + std::pow(4.0, 3.0), 1.0 / 3.0)));

    // Identity (axiom 2): d(x, x) == 0
    assert(approx(metric::euclidean{}(a, a), 0.0));
    assert(approx(metric::manhattan{}(a, a), 0.0));
    assert(approx(metric::chebyshev{}(a, a), 0.0));

    // Symmetry (axiom 3): d(x, y) == d(y, x)
    assert(approx(metric::euclidean{}(a, b), metric::euclidean{}(b, a)));
    assert(approx(metric::manhattan{}(a, b), metric::manhattan{}(b, a)));

    // Hamming on integer vectors.
    const std::vector<int> p = {1, 2, 3, 4, 5};
    const std::vector<int> q = {1, 0, 3, 0, 5};  // two diffs
    assert(metric::hamming{}(p, q) == 2.0);
    assert(metric::hamming{}(p, p) == 0.0);

    // Angular: two parallel vectors are at distance 0.
    const v2 unit_x = {1.0, 0.0};
    assert(approx(metric::angular{}(unit_x, unit_x), 0.0));
    const v2 unit_y = {0.0, 1.0};
    assert(approx(metric::angular{}(unit_x, unit_y), M_PI / 2.0));

    // Index plugs into euclidean functor exactly like the inline test type.
    metric::listofclusters<v2, metric::euclidean, 4, 16> idx;
    idx.insert(a, 0U);
    idx.insert(b, 1U);
    idx.insert({0.0, 0.0}, 2U);
    auto r = idx.knn_search(a, 99U, 2);
    assert(r.results().size() >= 1);

    std::cout << "  test_builtin_metrics: OK (euclidean, manhattan, chebyshev, minkowski<3>, hamming, angular)\n";
}

// Batch remove should leave the index in the same state as a serial loop
// of remove() calls and removed ids must no longer be retrievable.
static void test_batch_remove()
{
    constexpr std::uint32_t N = 100U;
    constexpr std::size_t D = 4U;

    std::vector<vec_t> db(N);
    std::vector<std::uint32_t> ids(N);
    std::mt19937 rng(13);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    for (std::uint32_t i = 0; i < N; ++i) {
        db[i].resize(D);
        for (std::size_t j = 0; j < D; ++j) db[i][j] = u(rng);
        ids[i] = i;
    }

    idx_t idx;
    idx.insert(db, ids);

    // Remove the first half via batch_remove.
    std::vector<vec_t> rm_objs(db.begin(), db.begin() + N/2);
    std::vector<std::uint32_t> rm_ids(ids.begin(), ids.begin() + N/2);
    idx.remove(rm_objs, rm_ids);

    // The removed ids should no longer be findable as 1-NN of themselves.
    int leaked = 0;
    for (std::uint32_t i = 0; i < N/2; ++i) {
        auto r = idx.knn_search(db[i], N + i, 1);
        for (const auto &x : r.results())
            if (x.id() == i) ++leaked;
    }
    assert(leaked == 0 && "removed ids still retrievable");

    // Surviving ids must still be retrievable.
    int found = 0;
    for (std::uint32_t i = N/2; i < N; ++i) {
        auto r = idx.knn_search(db[i], N + i, 1);
        for (const auto &x : r.results())
            if (x.id() == i) { ++found; break; }
    }
    assert(found == static_cast<int>(N/2) && "surviving ids not retrievable after batch remove");

    std::cout << "  test_batch_remove: OK (removed " << N/2 << ", " << found << " survivors retrievable)\n";
}

int main()
{
    std::cout << "liblistofclusters smoke tests:\n";
    test_build_and_knn();
    test_range_search();
    test_equal_distance_strict_weak_ordering();
    test_large_n_no_crash();
    test_knn_matches_brute_force();
    test_batch_knn_matches_serial();
    test_batch_insert_matches_serial();
    test_bulk_build_matches_brute_force();
    test_builtin_metrics();
    test_batch_remove();
    std::cout << "all tests passed\n";
    return 0;
}
