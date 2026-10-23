#include "pagerank_cuda.hpp"
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>

void check(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
pr::Graph graph(const std::string& text) { std::istringstream in(text); return pr::load_graph(in); }
void compare(const pr::Graph& g, int block, const pr::Options& o, pr::CudaMapping mapping) {
    const auto cpu = pr::pagerank(g, o);
    const auto gpu = pr::pagerank_cuda(g, block, o, mapping);
    const auto& r = gpu.result;
    check(cpu.converged == r.converged, "Convergence disagreement");
    check(r.iterations > 0 && r.iterations <= o.max_iterations, "Iteration bounds");
    if (o.max_iterations == 1) check(r.iterations == 1, "Iteration cap");
    double error = 0;
    for (std::size_t v = 0; v < g.size(); ++v) {
        check(std::isfinite(r.ranks[v]) && r.ranks[v] >= 0, "Invalid probability");
        error += std::abs(cpu.ranks[v] - r.ranks[v]);
    }
    check(error < 1e-9, "Rank disagreement");
    check(std::abs(std::accumulate(r.ranks.begin(), r.ranks.end(), 0.0) - 1) < 1e-9, "Mass conservation");
    if (r.converged) check(r.l1_difference < o.tolerance, "L1 stopping rule");
    check(gpu.compute_ms >= 0 && gpu.solver_end_to_end_ms >= gpu.compute_ms, "Timing bounds");
}
int main(int argc, char** argv) {
    try {
        for (int block : {128, 256, 512}) {
          for (auto mapping : {pr::CudaMapping::thread_per_vertex, pr::CudaMapping::warp_per_vertex}) {
            for (const auto& text : {"1 0", "1 1\n0 0", "3 0", "3 3\n0 1\n1 2\n2 0",
                 "2 1\n0 1", "4 5\n0 1\n0 1\n1 1\n1 2\n2 0"}) {
                const auto g = graph(text);
                compare(g, block, {}, mapping);
                pr::Options once; once.max_iterations = 1; once.tolerance = 1e-15;
                compare(g, block, once, mapping);
                pr::Options zero; zero.damping = 0; compare(g, block, zero, mapping);
            }
            // Non-multiple block boundary, skewed degree and multiple reduction blocks.
            std::ostringstream data; data << "1003 3000\n";
            for (int e = 0; e < 3000; ++e)
                data << (e * 17 % 800) << ' ' << ((e * e + 11) % 990) << '\n';
            compare(graph(data.str()), block, {}, mapping);
            // Long adjacency list, tail warp and capped-grid stride coverage.
            std::ostringstream star; star << "70003 70000\n";
            for (int u = 1; u <= 70000; ++u) star << u << " 0\n";
            compare(graph(star.str()), block, {}, mapping);
            if (argc > 1) compare(pr::load_graph(std::string(argv[1])), block, {}, mapping);
          }
        }
        for (int kind = 0; kind < 6; ++kind) {
            pr::Options o; int block = 256;
            if (kind == 0) block = 100;
            if (kind == 1) o.damping = 1;
            if (kind == 2) o.damping = std::numeric_limits<double>::quiet_NaN();
            if (kind == 3) o.tolerance = 0;
            if (kind == 4) o.max_iterations = 0;
            bool rejected = false;
            try { (void)pr::pagerank_cuda(kind == 5 ? pr::Graph{} : graph("1 0"), block, o); }
            catch (const std::invalid_argument&) { rejected = true; }
            check(rejected, "Invalid options accepted");
        }
        std::cout << "All CUDA correctness tests passed (thread and warp mappings; 128, 256, 512 threads/block)\n";
    } catch (const std::exception& e) { std::cerr << "FAIL: " << e.what() << '\n'; return 1; }
}
