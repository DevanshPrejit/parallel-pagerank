#include "pagerank_openmp.hpp"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <limits>
#include <omp.h>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: pagerank_omp.exe graph.txt threads\n";
        return 1;
    }
    try {
        const std::string value(argv[2]);
        if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos)
            throw std::invalid_argument("threads must be a positive integer");
        const auto parsed = std::stoull(value);
        if (parsed == 0 || parsed > static_cast<unsigned long long>(std::numeric_limits<int>::max()))
            throw std::invalid_argument("threads out of range");
        const int threads = static_cast<int>(parsed);
        omp_set_dynamic(0);
        const auto graph = pr::load_graph(std::string(argv[1]));
        const auto start = std::chrono::steady_clock::now();
        const auto result = pr::pagerank_openmp(graph, threads);
        const double ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();
        std::cout << std::setprecision(15)
                  << "vertices=" << graph.size() << " edges=" << graph.sources.size()
                  << " requested_threads=" << threads << " iterations=" << result.iterations
                  << " converged=" << std::boolalpha << result.converged
                  << " l1_difference=" << result.l1_difference << " solver_ms=" << ms << '\n';
        for (std::size_t v = 0; v < graph.size(); ++v)
            std::cout << v << ' ' << result.ranks[v] << '\n';
        return result.converged ? 0 : 2;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n'; return 1;
    }
}
