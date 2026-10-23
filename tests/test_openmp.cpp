#include "pagerank_openmp.hpp"
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <omp.h>
#include <sstream>
#include <stdexcept>

void check(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
pr::Graph graph(const std::string& text) {
    std::istringstream in(text); return pr::load_graph(in);
}
void compare(const pr::Graph& g, int threads, const pr::Options& o = {}) {
    const auto a = pr::pagerank(g, o);
    const auto b = pr::pagerank_openmp(g, threads, o);
    check(a.converged == b.converged, "Convergence disagreement");
    check(b.iterations > 0 && b.iterations <= o.max_iterations, "Iteration count");
    double l1 = 0;
    for (std::size_t v = 0; v < g.size(); ++v) {
        check(std::isfinite(b.ranks[v]) && b.ranks[v] >= 0, "Invalid probability");
        l1 += std::abs(a.ranks[v] - b.ranks[v]);
    }
    check(l1 < 1e-9, "Ranks disagree");
    check(std::abs(std::accumulate(b.ranks.begin(), b.ranks.end(), 0.0) - 1) < 1e-11,
          "Rank mass not conserved");
    if (b.converged) check(b.l1_difference < o.tolerance, "L1 stopping rule");
    if (o.max_iterations == 1) check(b.iterations == 1, "Iteration cap");
}
int main(int argc, char** argv) {
    try {
        omp_set_dynamic(0);
        for (int threads : {1, 2, 4, 8}) {
            for (const auto& text : {"1 0", "1 1\n0 0", "3 0", "3 3\n0 1\n1 2\n2 0",
                 "2 1\n0 1", "4 5\n0 1\n0 1\n1 1\n1 2\n2 0"}) {
                const auto g = graph(text);
                compare(g, threads);
                pr::Options once; once.max_iterations = 1; once.tolerance = 1e-15;
                compare(g, threads, once);
                pr::Options zero; zero.damping = 0;
                compare(g, threads, zero);
            }
            // Deterministic irregular graph with isolated and dangling nodes.
            std::ostringstream input;
            input << "200 600\n";
            for (int e = 0; e < 600; ++e)
                input << (e * 17 % 150) << ' ' << ((e * e + 11) % 190) << '\n';
            compare(graph(input.str()), threads);
            for (int repeat = 0; repeat < 5; ++repeat) compare(graph("2 1\n0 1"), threads);
            if (argc > 1) compare(pr::load_graph(std::string(argv[1])), threads);
        }
        for (int kind = 0; kind < 7; ++kind) {
            pr::Options o;
            int threads = 2;
            if (kind == 0) threads = 0;
            if (kind == 1) threads = -1;
            if (kind == 2) o.damping = 1;
            if (kind == 3) o.damping = std::numeric_limits<double>::quiet_NaN();
            if (kind == 4) o.tolerance = 0;
            if (kind == 5) o.max_iterations = 0;
            bool rejected = false;
            try { (void)pr::pagerank_openmp(kind == 6 ? pr::Graph{} : graph("1 0"), threads, o); }
            catch (const std::invalid_argument&) { rejected = true; }
            check(rejected, "Invalid options accepted");
        }
        std::cout << "All OpenMP correctness tests passed (1, 2, 4, 8 threads)\n";
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n'; return 1;
    }
}
