"""Validate saved evaluation CSVs and build tables, static charts and an HTML report."""
import argparse
import csv
import html
import json
import math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NAMES=['wiki-Vote','web-Stanford','web-Google','web-BerkStan']


def read(path):
    with path.open(encoding='utf-8-sig',newline='') as stream:
        return list(csv.DictReader(stream))


def table(headers, rows):
    return '<table><thead><tr>'+''.join(f'<th>{html.escape(str(x))}</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join(f'<td>{html.escape(str(x))}</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table>'


def summarize(folder):
    normalized=[]; summary=[]; cpu_series=[]
    graph_stats=json.loads((folder/'graph_statistics.json').read_text())
    graphs={x['input'].removesuffix('.txt'):x for x in graph_stats}
    for name in NAMES:
        cpu=read(folder/'raw'/f'{name}-cpu.csv'); gpu=read(folder/'raw'/f'{name}-cuda.csv')
        if len(cpu)!=5 or len(gpu)!=7: raise ValueError(f'Incomplete results for {name}')
        if {int(r['requested_threads']) for r in cpu[1:]}!={1,2,4,8}: raise ValueError('Missing CPU configuration')
        if {(r['mapping'],int(r['block_size'])) for r in gpu[1:]}!={(m,b) for m in ['thread','warp'] for b in [128,256,512]}: raise ValueError('Missing CUDA configuration')
        iterations=set()
        for kind,rows in [('cpu',cpu),('cuda',gpu)]:
            for row in rows:
                if int(row['vertices'])!=graphs[name]['vertices'] or int(row['edges'])!=graphs[name]['edges']: raise ValueError('Graph size mismatch')
                if int(row['repeats'])!=5: raise ValueError('Unexpected repeat count')
                error=float(row['max_rank_l1_vs_seq'])
                if not math.isfinite(error) or error>1e-9: raise ValueError('Rank error exceeds threshold')
                iterations.add(int(row['iterations']))
                prefix='solver' if kind=='cpu' else 'solver_end_to_end'
                median=float(row[f'median_{prefix}_ms']); lo=float(row[f'min_{prefix}_ms']); hi=float(row[f'max_{prefix}_ms'])
                if not 0<lo<=median<=hi: raise ValueError('Invalid timing range')
                config=row['requested_threads'] if kind=='cpu' else row['block_size']
                normalized.append({'dataset':name,'implementation':row['implementation'],'mapping':row.get('mapping','none'),
                    'configuration':config,'median_solver_ms':median,'min_solver_ms':lo,'max_solver_ms':hi,
                    'speedup_own_compiler_baseline':row['speedup'] if kind=='cpu' else row['solver_speedup'],
                    'max_rank_l1_vs_seq':error,'iterations':row['iterations'],'source':f'raw/{name}-{kind}.csv'})
        if len(iterations)!=1: raise ValueError('Iteration counts differ; inspect convergence before comparing')
        thread=min((r for r in gpu if r.get('mapping')=='thread'),key=lambda r:float(r['median_solver_end_to_end_ms']))
        warp=min((r for r in gpu if r.get('mapping')=='warp'),key=lambda r:float(r['median_solver_end_to_end_ms']))
        omp=next(r for r in cpu if r['implementation']=='openmp' and r['requested_threads']=='8')
        best=min([thread,warp],key=lambda r:float(r['median_solver_end_to_end_ms']))
        item={'dataset':name,'gcc_seq_ms':float(cpu[0]['median_solver_ms']),'omp8_ms':float(omp['median_solver_ms']),
              'omp8_speedup':float(omp['speedup']),'omp8_efficiency':float(omp['efficiency']),
              'msvc_seq_ms':float(gpu[0]['median_solver_end_to_end_ms']),
              'cuda_thread_ms':float(thread['median_solver_end_to_end_ms']),'cuda_thread_block':thread['block_size'],
              'cuda_warp_ms':float(warp['median_solver_end_to_end_ms']),'cuda_warp_block':warp['block_size'],
              'best_cuda_mapping':best['mapping'],'best_cuda_block':best['block_size'],
              'best_cuda_speedup_vs_msvc':float(best['solver_speedup']),
              'iterations':next(iter(iterations)), 'largest_rank_l1':max(float(r['max_rank_l1_vs_seq']) for r in cpu+gpu)}
        summary.append(item); cpu_series.append(cpu[1:])
    for name,data in [('all_configurations.csv',normalized),('summary.csv',summary)]:
        with (folder/name).open('w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    for ax,title,keys in [(axes[0],'CPU solver time (GCC)',[('gcc_seq_ms','Sequential'),('omp8_ms','OpenMP, 8 threads')]),
                          (axes[1],'GPU solver time (MSVC baseline)',[('msvc_seq_ms','Sequential'),('cuda_thread_ms','CUDA: best thread mapping'),('cuda_warp_ms','CUDA: best warp mapping')])]:
        width=.24
        for j,(key,label) in enumerate(keys):
            ax.bar([i+(j-(len(keys)-1)/2)*width for i in range(4)],[s[key] for s in summary],width,label=label)
        ax.set_xticks(range(4),NAMES,rotation=20,ha='right');ax.set_yscale('log');ax.set_ylabel('Median solver time (ms, logarithmic scale)');ax.set_title(title);ax.legend(fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.28));ax.grid(axis='y',alpha=.2)
    fig.suptitle('Four-graph evaluation — five measured repetitions after warm-up')
    fig.set_size_inches(12,6.2)
    fig.tight_layout();fig.savefig(folder/'runtime_comparison.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.5))
    best_rows=[]
    for s in summary:
        rows=read(folder/'raw'/f"{s['dataset']}-cuda.csv")
        best_rows.append(next(r for r in rows if r['mapping']==s['best_cuda_mapping'] and r['block_size']==s['best_cuda_block']))
    ax.bar([i-.18 for i in range(4)],[float(r['median_compute_ms']) for r in best_rows],.36,label='Iterative compute incl. convergence')
    ax.bar([i+.18 for i in range(4)],[float(r['median_solver_end_to_end_ms']) for r in best_rows],.36,label='Complete solver incl. setup/transfers')
    ax.set_xticks(range(4),NAMES);ax.set_yscale('log');ax.set_ylabel('Median time (ms, logarithmic scale)');ax.set_title('CUDA timing boundaries — fastest observed configuration per graph');ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(folder/'cuda_timing_boundaries.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.5))
    for name,rows in zip(NAMES,cpu_series):ax.plot([int(r['requested_threads']) for r in rows],[float(r['speedup']) for r in rows],'o-',label=name)
    ax.plot([1,8],[1,8],'--',color='gray',label='Ideal linear speedup');ax.axhline(1,color='gray',linewidth=.6)
    ax.set(xlabel='Requested CPU threads',ylabel='Speedup over GCC sequential',title='OpenMP scaling',xticks=[1,2,4,8]);ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(folder/'openmp_scaling.png',dpi=180);plt.close(fig)
    memory=read(folder/'memory_measurements.csv')
    fig,ax=plt.subplots(figsize=(9,4.5))
    configs=['sequential_gcc','openmp_8','cuda_thread_128','cuda_warp_128']
    for j,config in enumerate(configs):
        rows=[next(r for r in memory if r['dataset']==name and r['configuration']==config) for name in NAMES]
        ax.bar([i+(j-1.5)*.2 for i in range(4)],[float(r['host_peak_working_set_mib_observed']) for r in rows],.2,label=config)
    ax.set_xticks(range(4),NAMES);ax.set_ylabel('Observed host peak working set (MiB)');ax.set_title('Separate memory runs — includes loading and solver');ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(folder/'host_memory.png',dpi=180);plt.close(fig)
    env=json.loads((folder/'environment.json').read_text())
    graph_table=table(['Graph','Vertices','Edges','Dangling','Median in-degree','Max in-degree','Edges into top 1%'],[
        [n,f"{graphs[n]['vertices']:,}",f"{graphs[n]['edges']:,}",graphs[n]['dangling_vertices'],graphs[n]['median_in_degree'],f"{graphs[n]['max_in_degree']:,}",f"{100*graphs[n]['top_one_percent_vertices_edge_fraction']:.1f}%"] for n in NAMES])
    result_table=table(['Graph','GCC seq ms','OMP 8 ms','OMP speedup','MSVC seq ms','CUDA thread ms','CUDA warp ms'],[
        [s['dataset'],*[f'{s[k]:,.2f}' for k in ['gcc_seq_ms','omp8_ms','omp8_speedup','msvc_seq_ms','cuda_thread_ms','cuda_warp_ms']]] for s in summary])
    memory_table=table(['Graph','Configuration','Host peak observed MiB','GPU device delta sampled MiB'],[
        [r['dataset'],r['configuration'],f"{float(r['host_peak_working_set_mib_observed']):.1f}",f"{float(r['gpu_device_peak_delta_mib_sampled']):.1f}" if r['gpu_device_peak_delta_mib_sampled'] else '—'] for r in memory])
    berk=next(s for s in summary if s['dataset']=='web-BerkStan'); google=next(s for s in summary if s['dataset']=='web-Google')
    report=f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Parallel PageRank evaluation</title>
<style>body{{font:16px/1.6 system-ui,sans-serif;color:#172331;max-width:1120px;margin:40px auto;padding:0 24px;background:#fafbfc}}h1,h2{{line-height:1.2}}h2{{margin-top:40px}}table{{border-collapse:collapse;font-size:13px;width:100%;background:white}}th,td{{padding:9px;text-align:left;border-bottom:1px solid #dbe2e9}}th{{background:#eaf0f6}}img{{width:100%;height:auto}}pre{{white-space:pre-wrap;background:#edf1f5;padding:16px}}a{{color:#1859a0}}.lead{{font-size:20px}}</style>
<h1>Parallel PageRank: evaluation of the completed implementation</h1><p>23 October 2026 · Four fixed graphs · FP64 · RTX 4060 Laptop GPU</p>
<p class="lead">The completed experiments show graph-dependent CPU and GPU performance. Parallelism helps larger workloads, while the best GPU work assignment depends on incoming-degree structure.</p>
<h2>Scope and methods</h2><p>Sequential pull PageRank, static-schedule OpenMP at 1/2/4/8 threads, and CUDA thread-per-vertex and warp-per-vertex mappings at 128/256/512 threads per block. Uniform initial rank and teleportation, uniform dangling redistribution, damping 0.85, L1 difference &lt; 1e-10 and a 1000-iteration cap are shared. Each benchmark uses one warm-up per configuration followed by five measured repetitions. The CSVs retain median, minimum and maximum; these ranges are not confidence intervals.</p>
<p>Solver timing excludes file parsing and CSR construction. CPU timing includes solver allocation and initialization. CUDA end-to-end solver timing includes context setup, allocations, CSR upload, initialization, iterations, result download and cleanup; compute timing includes synchronization, scalar convergence transfers and host checks. Warmed measurements exclude cold-start context cost. Input loading is reported separately in available environment logs and is not included in solver speedups.</p>
<p>OpenMP speedups use the GCC sequential baseline from the same run. CUDA speedups use the NVCC/MSVC sequential baseline from the same run. Absolute CPU and GPU times can be displayed together, but do not treat cross-compiler ratios as controlled algorithm-only speedups. CUDA entries below select the fastest observed block size within each mapping; this is tuning on these same datasets, not held-out validation.</p>
<h2>Graphs and preprocessing</h2>{graph_table}<p>SNAP directed edge lists were remapped to consecutive IDs in first-appearance order. Directions, duplicate edges and self-loops were retained. Only vertices appearing in edges were included; missing isolated vertices cannot be inferred. Exact source and processed checksums are retained in each dataset's metadata. No new graphs were added for this evaluation.</p>
<p>Sources: <a href="https://snap.stanford.edu/data/wiki-Vote.html">wiki-Vote</a>, <a href="https://snap.stanford.edu/data/web-Stanford.html">Stanford</a>, <a href="https://snap.stanford.edu/data/web-Google.html">Google</a>, <a href="https://snap.stanford.edu/data/web-BerkStan.html">Berkeley–Stanford</a>. Dataset pages supply the original-paper citations.</p>
<h2>Runtime and scaling</h2>{result_table}<img src="runtime_comparison.png" alt="Runtime comparisons with separate GCC and MSVC baselines"><img src="openmp_scaling.png" alt="OpenMP speedup by thread count">
<h2>Findings supported by these experiments</h2><ul>
<li>Small workloads incur parallel overhead: evaluate wiki-Vote's scaling curve rather than assuming more threads help.</li>
<li>OpenMP improves the larger graph workloads, with sublinear thread scaling.</li>
<li>On Berkeley–Stanford, cooperative CUDA takes {berk['cuda_warp_ms']:.1f} ms versus {berk['cuda_thread_ms']:.1f} ms for the original mapping, a {berk['cuda_thread_ms']/berk['cuda_warp_ms']:.2f}× improvement between best tested configurations. Its median incoming degree is {graphs['web-BerkStan']['median_in_degree']:.0f}, but the maximum is {graphs['web-BerkStan']['max_in_degree']:,}. This is consistent with a workload-balancing benefit; profiling would be needed to isolate imbalance from memory-access effects.</li>
<li>On Google, cooperative CUDA takes {google['cuda_warp_ms']:.1f} ms versus {google['cuda_thread_ms']:.1f} ms for the original mapping. Cooperation is not uniformly beneficial.</li>
<li>CPU/GPU setup and transfers must be distinguished from iterative compute. Both measurements remain in the detailed CUDA tables.</li></ul>
<h2>Numerical agreement</h2><p>All selected benchmark rows meet the 1e-9 L1 rank-agreement threshold against their sequential reference. The largest recorded L1 disagreement is {max(s['largest_rank_l1'] for s in summary):.3g}. All configurations agree on iteration count within each graph. Existing correctness tests additionally cover dangling nodes, all-dangling graphs, self-loops, duplicate edges, iteration limits and malformed inputs. NVIDIA memory checking passed the CUDA synthetic test suite with zero errors; this is not a full performance profile.</p>
<h2>Memory measurements</h2>{memory_table}<img src="host_memory.png" alt="Observed process peak working set including input loading">
<p>These are separate single-run resource observations, not benchmark timing samples. Host values are OS peak working sets observed during 5 ms polling; a final short-lived peak can be missed. They include parser/CSR temporary storage, solver allocations and runtime overhead. GPU values are sampled device-wide NVML usage above a pre-run baseline; they are estimates, may miss transient allocations and may include other activity. They are not per-process VRAM peaks. The same 128-thread block size is used for both GPU mappings in these resource runs; it is not necessarily the fastest configuration.</p>
<h2>Hardware and software</h2><pre>{html.escape(json.dumps({k:env[k] for k in ['platform','cpu','physical_cores','logical_cpus','ram_gib','gpu','gcc','cuda']},indent=2))}</pre><p>Build flags: GCC C++17 -O2 -Wall -Wextra -Wpedantic (-fopenmp for OpenMP). CUDA C++17 -O2 -arch=sm_89 --fmad=false -Xcompiler=/Zc:preprocessor, with the installed Visual Studio host compiler. Source hashes are in environment.json.</p>
<h2>Provenance and limits</h2><p>CPU rows use the saved wiki-Vote, user-run Stanford, Google and Berkeley–Stanford CSVs. Google and Berkeley–Stanford CUDA rows use the later combined-mapping runs. The earlier noisy Berkeley–Stanford run is archived and not pooled into the selected run. CUDA wiki-Vote and Stanford rows were collected to complete the matrix. Measurements were made across sessions, with uncontrolled background load, laptop power/thermal state and CPU placement; they are exploratory local results. No claim of statistical significance, universal GPU superiority, or an optimized-framework comparison is supported.</p>
<p>The four graphs contain up to 875,713 vertices and 7,600,595 edges. Describe this measured scale explicitly rather than implying billion-edge evaluation. The synopsis's implementation and evaluation dimensions are represented, with approximate resource measurements and timing/compiler limitations disclosed. No additional kernels or datasets are required to write the project report.</p>
<h2>Detailed outputs</h2><p><a href="summary.csv">Summary table</a> · <a href="all_configurations.csv">All configurations and source paths</a> · <a href="memory_measurements.csv">Memory observations</a> · <a href="graph_statistics.json">Graph statistics</a> · <a href="environment.json">Hardware, methods and source hashes</a>. Selected raw CSVs are in raw/; earlier results are preserved in original_results/.</p></html>'''
    report=report.replace('<h2>Findings supported', '<img src="cuda_timing_boundaries.png" alt="GPU compute and complete solver timing"><h2>Findings supported')
    report=report.replace('<h2>Memory measurements</h2>', table(['Graph','Iterations','Largest recorded rank L1 difference'],[[s['dataset'],s['iterations'],f"{s['largest_rank_l1']:.3g}"] for s in summary])+'<h2>Memory measurements</h2>')
    (folder/'report.html').write_text(report,encoding='utf-8')
    print('Validated 48 configuration rows; report and charts saved to',folder)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('folder',type=Path)
    summarize(parser.parse_args().folder)
