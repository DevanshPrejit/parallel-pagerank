#include "pagerank_openmp.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <omp.h>
#include <stdexcept>

struct Measurement {
    std::vector<double> milliseconds;
    std::size_t iterations = 0;
    double max_l1 = 0;
};
int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: benchmark_cpu.exe graph.txt > results.csv\n"; return 1;
    }
    try {
        omp_set_dynamic(0);
        const auto g = pr::load_graph(std::string(argv[1]));
        const auto reference = pr::pagerank(g); // Sequential warm-up.
        if (!reference.converged) throw std::runtime_error("Reference did not converge");
        std::vector<int> counts{0}; // 0 selects the sequential implementation.
        for (int t : {1, 2, 4, 8}) if (t <= omp_get_num_procs()) counts.push_back(t);
        std::vector<int> actual(counts.size(), 1);
        for (std::size_t i = 1; i < counts.size(); ++i) {
            int observed = 0;
            #pragma omp parallel num_threads(counts[i])
            {
                #pragma omp single
                observed = omp_get_num_threads();
            }
            actual[i] = observed;
            (void)pr::pagerank_openmp(g, counts[i]); // Warm up each team.
        }
        std::vector<Measurement> measurements(counts.size());
        // Rotate order across five repetitions to reduce fixed-order bias.
        for (std::size_t repeat = 0; repeat < 5; ++repeat) {
            for (std::size_t j = 0; j < counts.size(); ++j) {
                const auto i = (j + repeat) % counts.size();
                const auto start = std::chrono::steady_clock::now();
                const auto result = counts[i] == 0 ? pr::pagerank(g) :
                    pr::pagerank_openmp(g, counts[i]);
                const auto stop = std::chrono::steady_clock::now();
                // Time the whole solver call: allocations, initialization,
                // thread-region entry, iterations and convergence checks.
                // File loading, warm-ups and rank verification are excluded.
                if (!result.converged) throw std::runtime_error("Solver did not converge");
                double error = 0;
                for (std::size_t v = 0; v < g.size(); ++v) {
                    if (!std::isfinite(result.ranks[v]) || result.ranks[v] < 0)
                        throw std::runtime_error("Invalid rank");
                    error += std::abs(result.ranks[v] - reference.ranks[v]);
                }
                if (error > 1e-9 || std::abs(std::accumulate(result.ranks.begin(), result.ranks.end(), 0.0) - 1) > 1e-9)
                    throw std::runtime_error("Correctness validation failed");
                auto& m = measurements[i];
                m.milliseconds.push_back(std::chrono::duration<double, std::milli>(stop - start).count());
                m.iterations = result.iterations;
                m.max_l1 = std::max(m.max_l1, error);
            }
        }
        for (auto& m : measurements) std::sort(m.milliseconds.begin(), m.milliseconds.end());
        const double baseline = measurements[0].milliseconds[2];
        std::cout << "implementation,requested_threads,observed_warmup_threads,vertices,edges,repeats,iterations,median_solver_ms,min_solver_ms,max_solver_ms,speedup,efficiency,max_rank_l1_vs_seq\n";
        std::cout << std::setprecision(12);
        for (std::size_t i = 0; i < counts.size(); ++i) {
            const auto& m = measurements[i];
            const double speedup = baseline / m.milliseconds[2];
            std::cout << (counts[i] == 0 ? "sequential" : "openmp") << ','
                      << (counts[i] == 0 ? 1 : counts[i]) << ',' << actual[i] << ','
                      << g.size() << ',' << g.sources.size() << ",5," << m.iterations << ','
                      << m.milliseconds[2] << ',' << m.milliseconds.front() << ','
                      << m.milliseconds.back() << ',' << speedup << ','
                      << speedup / actual[i] << ',' << m.max_l1 << '\n';
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n'; return 1;
    }
}
