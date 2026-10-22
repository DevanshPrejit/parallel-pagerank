#pragma once
#include "graph.hpp"

namespace pr {
struct Options {
    double damping = 0.85;
    double tolerance = 1e-10;
    std::size_t max_iterations = 1000;
};
struct Result {
    std::vector<double> ranks;
    std::size_t iterations = 0;
    double l1_difference = 0;
    bool converged = false;
};
// Graph must be produced by load_graph (consistent CSR and degree arrays).
Result pagerank(const Graph& graph, const Options& options = {});
}
