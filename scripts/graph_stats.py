"""Report degree skew for an already converted N M graph, without third-party packages."""
import argparse
from array import array
import json
from pathlib import Path


def stats(path):
    with path.open(encoding='ascii') as stream:
        n, m = map(int, next(stream).split())
        if n <= 0 or m < 0:
            raise ValueError('Invalid graph size')
        incoming = array('Q', [0]) * n
        outgoing = array('Q', [0]) * n
        count = 0
        for line in stream:
            u, v = map(int, line.split())
            if not (0 <= u < n and 0 <= v < n):
                raise ValueError('Vertex outside declared range')
            incoming[v] += 1
            outgoing[u] += 1
            count += 1
        if count != m:
            raise ValueError('Edge count does not match header')
    ordered = sorted(incoming)
    return {
        'input': path.name, 'vertices': n, 'edges': m,
        'mean_in_degree': m / n,
        'median_in_degree': (ordered[(n - 1) // 2] + ordered[n // 2]) / 2,
        'p99_in_degree': ordered[int(.99 * (n - 1))],
        'max_in_degree': ordered[-1],
        'top_one_percent_vertices_edge_fraction': sum(ordered[-max(1, n // 100):]) / m if m else 0,
        'vertices_with_in_degree_at_least_1024': sum(d >= 1024 for d in incoming),
        'dangling_vertices': sum(d == 0 for d in outgoing),
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('graph', type=Path)
    args = parser.parse_args()
    print(json.dumps(stats(args.graph), indent=2))
