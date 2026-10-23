"""Collect the four-graph evaluation without changing solver implementations.

Requires psutil. Memory runs are separate from benchmark timings. GPU memory
is device-wide NVML usage above a pre-run baseline, not per-process usage.
"""
import argparse
import csv
import ctypes
import importlib.util
import json
import platform
import shutil
import subprocess
import time
from pathlib import Path
import psutil

DATASETS = ['wiki-Vote', 'web-Stanford', 'web-Google', 'web-BerkStan']


class GpuMemory:
    class Memory(ctypes.Structure):
        _fields_ = [('total', ctypes.c_ulonglong), ('free', ctypes.c_ulonglong), ('used', ctypes.c_ulonglong)]

    def __init__(self):
        self.api = ctypes.WinDLL('nvml.dll')
        if self.api.nvmlInit_v2() != 0:
            raise RuntimeError('NVML initialization failed')
        self.handle = ctypes.c_void_p()
        if self.api.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(self.handle)) != 0:
            raise RuntimeError('NVML device query failed')

    def used(self):
        memory = self.Memory()
        if self.api.nvmlDeviceGetMemoryInfo(self.handle, ctypes.byref(memory)) != 0:
            raise RuntimeError('NVML memory query failed')
        return memory.used


def run(args, destination, errors=None):
    with destination.open('x', encoding='utf-8') as out:
        if errors:
            with errors.open('x', encoding='utf-8') as err:
                subprocess.run(args, stdout=out, stderr=err, check=True)
        else:
            subprocess.run(args, stdout=out, check=True)


def measure(args, gpu):
    base = gpu.used() if gpu else None
    device_peak = base
    host_peak = samples = 0
    start = time.perf_counter()
    with subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE) as process:
        observed = psutil.Process(process.pid)
        while process.poll() is None:
            try:
                memory = observed.memory_info()
                host_peak = max(host_peak, memory.peak_wset)
                samples += 1
            except psutil.NoSuchProcess:
                pass
            if gpu:
                device_peak = max(device_peak, gpu.used())
            time.sleep(.005)
        error = process.stderr.read().decode(errors='replace')
        if process.returncode:
            raise RuntimeError(f'{args}: {error}')
    return {'host_peak_working_set_mib_observed': host_peak / 2**20,
            'gpu_device_peak_delta_mib_sampled': (device_peak - base) / 2**20 if gpu else '',
            'memory_samples': samples, 'memory_run_wall_ms_not_benchmark': (time.perf_counter()-start)*1000}


def collect(repo, output):
    output.mkdir(parents=True, exist_ok=False)
    raw = output / 'raw'; raw.mkdir()
    environment = {
        'platform': platform.platform(), 'logical_cpus': psutil.cpu_count(),
        'physical_cores': psutil.cpu_count(logical=False),
        'ram_gib': psutil.virtual_memory().total / 2**30,
        'timing_policy': 'Retain saved runs; add CUDA benchmarks only for wiki-Vote and web-Stanford. Five repetitions after warm-up. Input/CSR loading excluded from solver timings.',
        'memory_policy': 'Separate single-run measurements per dataset/configuration. Poll every 5ms; host OS peak working set observed while alive, a lower bound if final peak is missed. Includes input parsing, CSR, solver and rank output to null. NVML device-wide sampled peak minus pre-run baseline; may include other GPU activity and miss short peaks. No memory measurement timings are used for speedups.',
        'baseline_policy': 'OpenMP speedups use GCC sequential in the same CSV. CUDA speedups use NVCC/MSVC sequential in the same CSV. Cross-compiler absolute timings are descriptive, not controlled CPU/GPU speedup estimates.',
    }
    for key, command in {
        'cpu': ['powershell','-NoProfile','-Command','Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors | ConvertTo-Json'],
        'gpu': ['nvidia-smi','--query-gpu=name,driver_version,memory.total,compute_cap','--format=csv,noheader'],
        'gcc': ['g++','--version'], 'cuda': ['nvcc','--version'],
    }.items():
        result = subprocess.run(command, capture_output=True, text=True)
        environment[key] = result.stdout.strip() if result.returncode == 0 else result.stderr.strip()
    environment['source_files_sha256'] = {}
    import hashlib
    for path in list((repo/'src').glob('*')) + list((repo/'include').glob('*')):
        if path.is_file(): environment['source_files_sha256'][str(path.relative_to(repo))] = hashlib.sha256(path.read_bytes()).hexdigest()
    (output/'environment.json').write_text(json.dumps(environment,indent=2),encoding='utf-8')
    selections = {
        'wiki-Vote': ('day2-benchmark-wiki-Vote.csv', None),
        'web-Stanford': ('stanford-cpu.csv', None),
        'web-Google': ('google-cpu.csv', 'google-cuda-mapping.csv'),
        'web-BerkStan': ('berkstan-cpu.csv', 'berkstan-cuda-mapping.csv'),
    }
    # Archive all original result CSVs too, including the noisy first run.
    archive = output/'original_results'; archive.mkdir()
    for path in (repo/'results').iterdir():
        if path.is_file() and path.suffix in {'.csv','.json','.txt'}:
            shutil.copy2(path, archive/path.name)
    spec = importlib.util.spec_from_file_location('graph_stats', repo/'scripts/graph_stats.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    graphs = []
    measurements = []
    gpu = GpuMemory()
    for name in DATASETS:
        print(f'Collecting {name}', flush=True)
        graph = repo/'data/processed'/f'{name}.txt'
        graphs.append(module.stats(graph))
        cpu_source, cuda_source = selections[name]
        shutil.copy2(repo/'results'/cpu_source, raw/f'{name}-cpu.csv')
        if cuda_source:
            shutil.copy2(repo/'results'/cuda_source, raw/f'{name}-cuda.csv')
        else:
            run([str(repo/'build/benchmark_cuda.exe'),str(graph)],raw/f'{name}-cuda.csv',raw/f'{name}-cuda-environment.txt')
        for label, executable, options in [
            ('sequential_gcc','pagerank_seq.exe',[]),
            ('openmp_8','pagerank_omp.exe',['8']),
            ('cuda_thread_128','pagerank_cuda.exe',['128','thread']),
            ('cuda_warp_128','pagerank_cuda.exe',['128','warp']),
        ]:
            memory = measure([str(repo/'build'/executable),str(graph),*options],gpu if label.startswith('cuda') else None)
            measurements.append({'dataset':name,'configuration':label,**memory})
    (output/'graph_statistics.json').write_text(json.dumps(graphs,indent=2),encoding='utf-8')
    with (output/'memory_measurements.csv').open('x',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(measurements[0])); writer.writeheader(); writer.writerows(measurements)
    print('Collection complete',flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('repo',type=Path); parser.add_argument('output',type=Path)
    args=parser.parse_args(); collect(args.repo,args.output)
