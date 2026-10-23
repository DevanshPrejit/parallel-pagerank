#include "pagerank_cuda.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>

struct Samples {
    std::vector<double> compute, end_to_end;
    std::size_t iterations = 0;
    double max_error = 0;
};
int main(int argc, char** argv) {
    if (argc != 2) { std::cerr << "Usage: benchmark_cuda.exe graph.txt > results.csv\n"; return 1; }
    try {
        const auto load_start = std::chrono::steady_clock::now();
        const auto g = pr::load_graph(std::string(argv[1]));
        const double load_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - load_start).count();
        cudaDeviceProp properties{};
        if (cudaGetDeviceProperties(&properties, 0) != cudaSuccess) throw std::runtime_error("Cannot query GPU");
        std::cerr << "GPU: " << properties.name << "; input_load_and_csr_ms=" << load_ms
                  << "; fp64; damping=0.85; tolerance=1e-10; cap=1000; warmup=1/config; repeats=5\n";
        const auto reference = pr::pagerank(g);
        if (!reference.converged) throw std::runtime_error("Reference did not converge");
        for (int block : {128, 256, 512})
            for (auto mapping : {pr::CudaMapping::thread_per_vertex, pr::CudaMapping::warp_per_vertex})
                (void)pr::pagerank_cuda(g, block, {}, mapping);
        Samples samples[7]; // sequential, three thread and three warp configurations
        for (int repeat = 0; repeat < 5; ++repeat) {
            for (int j = 0; j < 7; ++j) {
                const int i = (j + repeat) % 7;
                const auto start = std::chrono::steady_clock::now();
                pr::Result r;
                double compute, total;
                if (i == 0) {
                    r = pr::pagerank(g);
                    total = compute = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
                } else {
                    auto cuda = pr::pagerank_cuda(g, 128 << ((i - 1) % 3), {},
                        i <= 3 ? pr::CudaMapping::thread_per_vertex : pr::CudaMapping::warp_per_vertex);
                    compute = cuda.compute_ms; total = cuda.solver_end_to_end_ms;
                    r = std::move(cuda.result);
                }
                if (!r.converged) throw std::runtime_error("Did not converge");
                double error = 0;
                for (std::size_t v = 0; v < g.size(); ++v) {
                    if (!std::isfinite(r.ranks[v]) || r.ranks[v] < 0) throw std::runtime_error("Invalid rank");
                    error += std::abs(r.ranks[v] - reference.ranks[v]);
                }
                if (error > 1e-9 || std::abs(std::accumulate(r.ranks.begin(), r.ranks.end(), 0.0) - 1) > 1e-9)
                    throw std::runtime_error("Rank validation failed");
                samples[i].compute.push_back(compute); samples[i].end_to_end.push_back(total);
                samples[i].iterations = r.iterations;
                samples[i].max_error = std::max(samples[i].max_error, error);
            }
        }
        for (auto& s : samples) {
            std::sort(s.compute.begin(), s.compute.end());
            std::sort(s.end_to_end.begin(), s.end_to_end.end());
        }
        const double baseline = samples[0].end_to_end[2];
        std::cout << std::setprecision(12)
                  << "implementation,mapping,block_size,vertices,edges,repeats,iterations,median_compute_ms,min_compute_ms,max_compute_ms,median_solver_end_to_end_ms,min_solver_end_to_end_ms,max_solver_end_to_end_ms,compute_speedup,solver_speedup,max_rank_l1_vs_seq\n";
        for (int i = 0; i < 7; ++i) {
            const auto& s = samples[i];
            std::cout << (i == 0 ? "sequential_msvc" : "cuda") << ','
                      << (i == 0 ? "none" : i <= 3 ? "thread" : "warp") << ','
                      << (i == 0 ? 0 : 128 << ((i - 1) % 3))
                      << ',' << g.size() << ',' << g.sources.size() << ",5," << s.iterations << ','
                      << s.compute[2] << ',' << s.compute.front() << ',' << s.compute.back() << ','
                      << s.end_to_end[2] << ',' << s.end_to_end.front() << ',' << s.end_to_end.back() << ','
                      << baseline / s.compute[2] << ',' << baseline / s.end_to_end[2] << ',' << s.max_error << '\n';
        }
    } catch (const std::exception& e) { std::cerr << "Error: " << e.what() << '\n'; return 1; }
}
