#include "graph.hpp"
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace pr {
namespace {
bool record(std::istream& in, std::string& line) {
    while (std::getline(in, line)) {
        line.erase(line.find('#') == std::string::npos ? line.size() : line.find('#'));
        if (line.find_first_not_of(" \t\r\n") != std::string::npos) return true;
    }
    if (in.bad()) throw std::runtime_error("Graph read failed");
    return false;
}
std::size_t number(const std::string& token) {
    if (token.empty()) throw std::runtime_error("Missing integer");
    std::size_t n = 0;
    for (char c : token) {
        if (c < '0' || c > '9') throw std::runtime_error("Expected nonnegative integer");
        const auto digit = static_cast<std::size_t>(c - '0');
        if (n > (std::numeric_limits<std::size_t>::max() - digit) / 10)
            throw std::runtime_error("Integer overflow");
        n = n * 10 + digit;
    }
    return n;
}
std::pair<std::size_t, std::size_t> pair(const std::string& line) {
    std::istringstream row(line);
    std::string a, b, extra;
    if (!(row >> a >> b) || (row >> extra))
        throw std::runtime_error("Expected exactly two integers per record");
    return {number(a), number(b)};
}
}
Graph load_graph(std::istream& in) {
    std::string line;
    if (!record(in, line)) throw std::runtime_error("Missing N M header");
    const auto [n, m] = pair(line);
    if (n == 0 || n == std::numeric_limits<std::size_t>::max())
        throw std::runtime_error("Vertex count must be positive and representable");
    Graph g;
    g.offsets.assign(n + 1, 0);
    g.out_degree.assign(n, 0);
    std::vector<std::pair<std::size_t, std::size_t>> edges;
    for (std::size_t i = 0; i < m; ++i) {
        if (!record(in, line)) throw std::runtime_error("Too few edges");
        const auto [u, v] = pair(line);
        if (u >= n || v >= n) throw std::runtime_error("Vertex ID outside [0,N)");
        edges.emplace_back(u, v);
        ++g.out_degree[u];
        ++g.offsets[v + 1];
    }
    if (record(in, line)) throw std::runtime_error("Too many edge records");
    for (std::size_t v = 0; v < n; ++v) g.offsets[v + 1] += g.offsets[v];
    g.sources.resize(m);
    auto cursor = g.offsets;
    for (const auto& [u, v] : edges) g.sources[cursor[v]++] = u;
    return g;
}
Graph load_graph(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Cannot open graph: " + path);
    return load_graph(in);
}
}
