"""Convert a directed SNAP edge list to the project's N M format.

Only vertices appearing in edges are included. No isolated vertices are
invented from gaps in the original IDs. Duplicate edges and self-loops stay.
Outputs are created exclusively: existing files are never overwritten.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path


def edges(path):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as stream:
        for line_number, line in enumerate(stream, 1):
            line = line.partition('#')[0].strip()
            if not line:
                continue
            tokens = line.split()
            if len(tokens) != 2 or any(not t.isascii() or not t.isdecimal() for t in tokens):
                raise ValueError(f'{path}:{line_number}: expected two nonnegative integer IDs')
            yield int(tokens[0]), int(tokens[1])


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def convert(source, output):
    mapping_path = output.with_suffix('.ids.tsv')
    metadata_path = output.with_suffix('.json')
    for path in (output, mapping_path, metadata_path):
        if path.exists():
            raise FileExistsError(f'Refusing to overwrite {path}')
    original_hash = digest(source)
    ids = {}
    degrees = []
    count = loops = 0
    for u, v in edges(source):
        for node in (u, v):
            if node not in ids:
                ids[node] = len(ids)
                degrees.append(0)
        degrees[ids[u]] += 1
        count += 1
        loops += u == v
    if not ids:
        raise ValueError('No vertices found')
    output.parent.mkdir(parents=True, exist_ok=True)
    created = []
    try:
        with output.open('x', encoding='ascii', newline='\n') as stream:
            created.append(output)
            stream.write(f'{len(ids)} {count}\n')
            written = 0
            for u, v in edges(source):
                stream.write(f'{ids[u]} {ids[v]}\n')
                written += 1
        if written != count or digest(source) != original_hash:
            raise RuntimeError('Source changed during conversion')
        with mapping_path.open('x', encoding='ascii', newline='\n') as stream:
            created.append(mapping_path)
            stream.write('new_id\toriginal_id\n')
            for original, new in ids.items():
                stream.write(f'{new}\t{original}\n')
        metadata = {
            'source': source.name, 'source_sha256': original_hash,
            'vertices': len(ids), 'edges': count,
            'dangling_vertices': sum(d == 0 for d in degrees),
            'maximum_out_degree': max(degrees), 'self_loops': loops,
            'directed': True, 'duplicates': 'preserved; not deduplicated',
            'vertex_policy': 'IDs appearing in edges; first appearance order',
            'processed_sha256': digest(output),
        }
        with metadata_path.open('x', encoding='utf-8') as stream:
            created.append(metadata_path)
            json.dump(metadata, stream, indent=2)
            stream.write('\n')
    except BaseException:
        for path in created:
            path.unlink()
        raise
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    convert(args.source, args.output)
