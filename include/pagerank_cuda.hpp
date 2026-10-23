#pragma once
#include "pagerank.hpp"

namespace pr {
struct CudaResult {
    Result result;
    // Wall-clock iterative solve, including convergence transfers/checks.
    double compute_ms = 0;
    // Complete call: context setup, allocation, CSR upload, initialization,
    // solve, result download and cleanup. Input parsing/CSR construction excluded.
    double solver_end_to_end_ms = 0;
};
enum class CudaMapping { thread_per_vertex, warp_per_vertex };
CudaResult pagerank_cuda(const Graph& graph, int block_size = 256, const Options& options = {},
                         CudaMapping mapping = CudaMapping::thread_per_vertex);
}
