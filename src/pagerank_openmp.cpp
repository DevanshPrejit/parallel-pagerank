#include "pagerank_openmp.hpp"
#include <cmath>
#include <stdexcept>

namespace pr {
Result pagerank_openmp(const Graph& g, int threads, const Options& o) {
    if (threads <= 0 || g.size() == 0 || !std::isfinite(o.damping) ||
        o.damping < 0 || o.damping >= 1 || !std::isfinite(o.tolerance) ||
        o.tolerance <= 0 || o.max_iterations == 0)
        throw std::invalid_argument("Require threads>0, N>0, 0<=damping<1, finite tolerance>0, max_iterations>0");
    const auto n = g.size();
    const double count = static_cast<double>(n);
    Result r;
    r.ranks.assign(n, 1.0 / count);
    std::vector<double> next(n);
    double dangling = 0, difference = 0, base = 0;
    bool done = false;
    // Keep one team alive across iterations. Each work-sharing loop ends
    // with a barrier; single also publishes swaps and the stopping decision.
    #pragma omp parallel num_threads(threads) shared(r, next, dangling, difference, base, done)
    {
        for (std::size_t iteration = 0; iteration < o.max_iterations; ++iteration) {
            #pragma omp single
            { dangling = 0; difference = 0; }
            #pragma omp for schedule(static) reduction(+:dangling)
            for (std::size_t u = 0; u < n; ++u)
                if (g.out_degree[u] == 0) dangling += r.ranks[u];
            #pragma omp single
            { base = (1.0 - o.damping + o.damping * dangling) / count; }
            #pragma omp for schedule(static) reduction(+:difference)
            for (std::size_t v = 0; v < n; ++v) {
                double incoming = 0;
                for (auto e = g.offsets[v]; e < g.offsets[v + 1]; ++e) {
                    const auto u = g.sources[e];
                    incoming += r.ranks[u] / static_cast<double>(g.out_degree[u]);
                }
                next[v] = base + o.damping * incoming;
                difference += std::abs(next[v] - r.ranks[v]);
            }
            #pragma omp single
            {
                r.ranks.swap(next);
                r.iterations = iteration + 1;
                r.l1_difference = difference;
                done = difference < o.tolerance;
                r.converged = done;
            }
            if (done) break;
        }
    }
    return r;
}
}
