$ErrorActionPreference = 'Stop'
Push-Location (Split-Path -Parent $PSScriptRoot)
try {
    New-Item -ItemType Directory -Force build | Out-Null
    $common = @('-std=c++17', '-O2', '-Wall', '-Wextra', '-Wpedantic', '-fopenmp', '-Iinclude', 'src/graph.cpp', 'src/pagerank.cpp', 'src/pagerank_openmp.cpp')
    foreach ($target in @(
        @('src/openmp.cpp', 'build/pagerank_omp.exe'),
        @('tests/test_openmp.cpp', 'build/test_openmp.exe'),
        @('src/benchmark_cpu.cpp', 'build/benchmark_cpu.exe')
    )) {
        & g++ @common $target[0] -o $target[1]
        if ($LASTEXITCODE -ne 0) { throw "Build failed: $($target[0])" }
    }
    & .\build\test_openmp.exe data/small_graph.txt
    if ($LASTEXITCODE -ne 0) { throw 'OpenMP correctness tests failed' }
} finally {
    Pop-Location
}
