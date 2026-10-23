$ErrorActionPreference = 'Stop'
Push-Location (Split-Path -Parent $PSScriptRoot)
try {
    if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
        $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
        if (-not (Test-Path -LiteralPath $vswhere)) { throw 'Visual Studio C++ Build Tools required' }
        $installation = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
        if (-not $installation) { throw 'No Visual Studio installation with C++ tools found' }
        Import-Module (Join-Path $installation 'Common7\Tools\Microsoft.VisualStudio.DevShell.dll')
        Enter-VsDevShell -VsInstallPath $installation -SkipAutomaticLocation -DevCmdArguments '-arch=x64 -host_arch=x64'
    }
    New-Item -ItemType Directory -Force build | Out-Null
    # Native Ada code avoids relying on newer PTX JIT support in the driver.
    $common = @('-std=c++17', '-O2', '-arch=sm_89', '--fmad=false', '-Xcompiler=/Zc:preprocessor', '-Iinclude', 'src/graph.cpp', 'src/pagerank.cpp', 'src/pagerank_cuda.cu')
    foreach ($target in @(
        @('src/cuda.cu', 'build/pagerank_cuda.exe'),
        @('tests/test_cuda.cu', 'build/test_cuda.exe'),
        @('src/benchmark_cuda.cu', 'build/benchmark_cuda.exe')
    )) {
        & nvcc @common $target[0] -o $target[1]
        if ($LASTEXITCODE -ne 0) { throw "CUDA build failed: $($target[0])" }
    }
    & .\build\test_cuda.exe data/small_graph.txt
    if ($LASTEXITCODE -ne 0) { throw 'CUDA correctness tests failed' }
} finally { Pop-Location }
