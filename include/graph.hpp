#pragma once
#include <cstddef>
#include <istream>
#include <string>
#include <vector>

namespace pr {
// Incoming sources for v occupy offsets[v] .. offsets[v+1].
struct Graph {
    std::vector<std::size_t> offsets, sources, out_degree;
    std::size_t size() const { return out_degree.size(); }
};
// First non-comment line: N M. Then exactly M directed edges: source target.
// IDs are zero-based; duplicate edges and self-loops are retained.
Graph load_graph(std::istream& input);
Graph load_graph(const std::string& path);
}
