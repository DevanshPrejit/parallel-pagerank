#include "pagerank.hpp"
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>

void check(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
void near(double a, double b, double tolerance = 1e-11) {
    check(std::abs(a - b) < tolerance, "Numeric mismatch");
}
pr::Graph graph(const std::string& text) {
    std::istringstream in(text);
    return pr::load_graph(in);
}
void rejects(const std::string& text) {
    bool failed = false;
    try { (void)graph(text); } catch (const std::exception&) { failed = true; }
    check(failed, "Malformed input accepted");
}
void probability(const pr::Result& r) {
    check(r.converged, "Did not converge");
    near(std::accumulate(r.ranks.begin(), r.ranks.end(), 0.0), 1.0);
    for (double p : r.ranks) check(std::isfinite(p) && p >= 0, "Invalid probability");
}
int main(int argc, char** argv) {
    try {
        const auto csr = graph("# comment\n4 4\n2 3\n0 1\n2 0\n1 2 # inline\n");
        check(csr.offsets == std::vector<std::size_t>({0,1,2,3,4}), "CSR offsets");
        check(csr.sources == std::vector<std::size_t>({2,0,1,2}), "CSR sources");
        check(csr.out_degree == std::vector<std::size_t>({1,1,2,0}), "Out degrees");
        auto cycle = pr::pagerank(graph("3 3\n0 1\n1 2\n2 0\n"));
        probability(cycle);
        for (double p : cycle.ranks) near(p, 1.0/3);
        auto empty_edges = pr::pagerank(graph("3 0\n"));
        probability(empty_edges);
        for (double p : empty_edges.ranks) near(p, 1.0/3);
        probability(pr::pagerank(graph("1 1\n0 0\n")));
        // Analytic solution for 0 -> 1, with vertex 1 dangling:
        // p0 = 1/(2+d), p1 = (1+d)/(2+d).
        const auto chain = graph("2 1\n0 1\n");
        pr::Options precise; precise.tolerance = 1e-13;
        auto analytic = pr::pagerank(chain, precise);
        probability(analytic);
        near(analytic.ranks[0], 1.0/2.85);
        near(analytic.ranks[1], 1.85/2.85);
        pr::Options once; once.max_iterations = 1; once.tolerance = 1e-15;
        auto first = pr::pagerank(chain, once);
        near(first.ranks[0], .2875); near(first.ranks[1], .7125);
        near(first.l1_difference, .425);
        check(!first.converged && first.iterations == 1, "Iteration cap");
        auto duplicates = graph("3 3\n0 1\n0 1\n0 2\n");
        check(duplicates.out_degree[0] == 3 && duplicates.offsets[2] == 2, "Duplicate edges");
        auto dup_first = pr::pagerank(duplicates, once);
        near(dup_first.ranks[1] - dup_first.ranks[2], .85/9);
        pr::Options zero; zero.damping = 0;
        for (double p : pr::pagerank(csr, zero).ranks) near(p, .25);
        for (const auto& bad : {"", "0 0", "-1 0", "2 1\n0 2", "2 2\n0 1",
             "2 0\n0 1", "2 1\n0 x", "2 1\n0 1 2", "2 0 extra",
             "184467440737095516160 0"}) rejects(bad);
        for (int i = 0; i < 5; ++i) {
            pr::Options invalid;
            if (i == 0) invalid.damping = 1;
            if (i == 1) invalid.damping = -.1;
            if (i == 2) invalid.tolerance = 0;
            if (i == 3) invalid.max_iterations = 0;
            if (i == 4) invalid.damping = std::numeric_limits<double>::quiet_NaN();
            bool failed = false;
            try { (void)pr::pagerank(chain, invalid); }
            catch (const std::invalid_argument&) { failed = true; }
            check(failed, "Invalid options accepted");
        }
        // Independent dense push reference, using original fixture edges.
        const auto fixture = pr::load_graph(std::string(argc > 1 ? argv[1] : "data/small_graph.txt"));
        auto actual = pr::pagerank(fixture, precise);
        probability(actual);
        check(fixture.size() == 4, "Wrong fixture");
        std::vector<double> p(4, .25);
        for (int i = 0; i < 1000; ++i) {
            std::vector<double> q(4, .15/4 + .85*p[3]/4);
            q[1] += .85*p[0]; q[2] += .85*p[1];
            q[0] += .85*p[2]/2; q[3] += .85*p[2]/2;
            p.swap(q);
        }
        for (std::size_t v = 0; v < 4; ++v) near(actual.ranks[v], p[v]);
        check(actual.l1_difference < precise.tolerance, "L1 stopping criterion");
        std::cout << "All correctness tests passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n'; return 1;
    }
}
