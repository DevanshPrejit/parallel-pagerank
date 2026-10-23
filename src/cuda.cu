#include "pagerank_cuda.hpp"
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
    if (argc < 2 || argc > 4) {
        std::cerr << "Usage: pagerank_cuda.exe graph.txt [block_size: 128|256|512] [thread|warp]\n"; return 1;
    }
    try {
        int block = 256;
        if (argc >= 3) {
            const std::string value(argv[2]);
            if (value != "128" && value != "256" && value != "512")
                throw std::invalid_argument("block_size must be 128, 256 or 512");
            block = std::stoi(value);
        }
        const std::string mapping = argc == 4 ? argv[3] : "thread";
        if (mapping != "thread" && mapping != "warp") throw std::invalid_argument("mapping must be thread or warp");
        const auto graph = pr::load_graph(std::string(argv[1]));
        const auto gpu = pr::pagerank_cuda(graph, block, {}, mapping == "warp" ?
            pr::CudaMapping::warp_per_vertex : pr::CudaMapping::thread_per_vertex);
        const auto& r = gpu.result;
        std::cout << std::setprecision(15) << "vertices=" << graph.size()
                  << " edges=" << graph.sources.size() << " block_size=" << block << " mapping=" << mapping
                  << " iterations=" << r.iterations << " converged=" << std::boolalpha << r.converged
                  << " l1_difference=" << r.l1_difference << " compute_ms=" << gpu.compute_ms
                  << " solver_end_to_end_ms=" << gpu.solver_end_to_end_ms << '\n';
        for (std::size_t v = 0; v < graph.size(); ++v) std::cout << v << ' ' << r.ranks[v] << '\n';
        return r.converged ? 0 : 2;
    } catch (const std::exception& e) { std::cerr << "Error: " << e.what() << '\n'; return 1; }
}
