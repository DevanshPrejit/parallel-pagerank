#include "pagerank.hpp"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: pagerank_seq.exe graph.txt\n";
        return 1;
    }
    try {
        const auto graph = pr::load_graph(std::string(argv[1]));
        const auto start = std::chrono::steady_clock::now();
        const auto result = pr::pagerank(graph);
        const double ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();
        std::cout << std::setprecision(15)
                  << "vertices=" << graph.size() << " edges=" << graph.sources.size()
                  << " iterations=" << result.iterations
                  << " converged=" << std::boolalpha << result.converged
                  << " l1_difference=" << result.l1_difference
                  << " compute_ms=" << ms << '\n';
        for (std::size_t v = 0; v < graph.size(); ++v)
            std::cout << v << ' ' << result.ranks[v] << '\n';
        return result.converged ? 0 : 2;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}
