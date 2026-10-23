#pragma once
#include "pagerank.hpp"

namespace pr {
// threads must be positive; graph must satisfy the shared loader contract.
Result pagerank_openmp(const Graph& graph, int threads, const Options& options = {});
}
