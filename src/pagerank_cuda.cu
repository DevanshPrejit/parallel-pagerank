#include "pagerank_cuda.hpp"
#include <cuda_runtime.h>
#include <cub/device/device_reduce.cuh>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

namespace pr {
namespace {
void checked(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}
template<class T> struct Device {
    T* p = nullptr;
    explicit Device(std::size_t count) {
        if (count) checked(cudaMalloc(reinterpret_cast<void**>(&p), count * sizeof(T)), "cudaMalloc");
    }
    ~Device() { if (p) cudaFree(p); }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
};
__global__ void initialize(double* rank, std::size_t n) {
    for (std::size_t v = blockIdx.x * blockDim.x + threadIdx.x; v < n;
         v += static_cast<std::size_t>(gridDim.x) * blockDim.x) rank[v] = 1.0 / n;
}
// Each block writes one partial; CUB then reduces partials without atomics.
__global__ void dangling_partials(const double* rank, const std::size_t* degree,
                                 std::size_t n, double* partial) {
    __shared__ double sums[512];
    double sum = 0;
    for (std::size_t v = blockIdx.x * blockDim.x + threadIdx.x; v < n;
         v += static_cast<std::size_t>(gridDim.x) * blockDim.x)
        if (degree[v] == 0) sum += rank[v];
    sums[threadIdx.x] = sum;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2; stride; stride /= 2) {
        if (threadIdx.x < stride) sums[threadIdx.x] += sums[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) partial[blockIdx.x] = sums[0];
}
__global__ void pull(const std::size_t* offsets, const std::size_t* sources,
                     const std::size_t* degree, const double* rank, double* next,
                     const double* dangling, std::size_t n, double damping, double* partial) {
    __shared__ double sums[512];
    double difference = 0;
    const double base = (1.0 - damping + damping * *dangling) / n;
    for (std::size_t v = blockIdx.x * blockDim.x + threadIdx.x; v < n;
         v += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        double incoming = 0;
        for (auto e = offsets[v]; e < offsets[v + 1]; ++e) {
            const auto u = sources[e];
            incoming += rank[u] / static_cast<double>(degree[u]);
        }
        const double value = base + damping * incoming;
        next[v] = value;
        difference += fabs(value - rank[v]);
    }
    sums[threadIdx.x] = difference;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2; stride; stride /= 2) {
        if (threadIdx.x < stride) sums[threadIdx.x] += sums[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) partial[blockIdx.x] = sums[0];
}
// A full warp cooperates on one incoming adjacency list. All 32 lanes
// execute the shuffle even for vertices with no incoming edges.
__global__ void pull_warp(const std::size_t* offsets, const std::size_t* sources,
                          const std::size_t* degree, const double* rank, double* next,
                          const double* dangling, std::size_t n, double damping, double* partial) {
    __shared__ double sums[512];
    const unsigned lane = threadIdx.x % 32;
    const unsigned warps_per_block = blockDim.x / 32;
    const std::size_t first = static_cast<std::size_t>(blockIdx.x) * warps_per_block + threadIdx.x / 32;
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * warps_per_block;
    const double base = (1.0 - damping + damping * *dangling) / n;
    double difference = 0;
    for (std::size_t v = first; v < n; v += stride) {
        double incoming = 0;
        for (auto e = offsets[v] + lane; e < offsets[v + 1]; e += 32) {
            const auto u = sources[e];
            incoming += rank[u] / static_cast<double>(degree[u]);
        }
        for (int shift = 16; shift; shift /= 2)
            incoming += __shfl_down_sync(0xffffffffu, incoming, shift);
        if (lane == 0) {
            const double value = base + damping * incoming;
            next[v] = value;
            difference += fabs(value - rank[v]);
        }
    }
    // Every thread reaches this barrier, including inactive tail warps.
    sums[threadIdx.x] = difference;
    __syncthreads();
    for (unsigned shift = blockDim.x / 2; shift; shift /= 2) {
        if (threadIdx.x < shift) sums[threadIdx.x] += sums[threadIdx.x + shift];
        __syncthreads();
    }
    if (threadIdx.x == 0) partial[blockIdx.x] = sums[0];
}
using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
}
CudaResult pagerank_cuda(const Graph& g, int block_size, const Options& o, CudaMapping mapping) {
    if (g.size() == 0 || (block_size != 128 && block_size != 256 && block_size != 512) ||
        !std::isfinite(o.damping) || o.damping < 0 || o.damping >= 1 ||
        !std::isfinite(o.tolerance) || o.tolerance <= 0 || o.max_iterations == 0 ||
        (mapping != CudaMapping::thread_per_vertex && mapping != CudaMapping::warp_per_vertex))
        throw std::invalid_argument("Require N>0, block_size in {128,256,512}, 0<=damping<1, finite tolerance>0, max_iterations>0");
    const auto start = Clock::now();
    CudaResult output;
    {
        checked(cudaFree(nullptr), "CUDA context initialization");
        const auto n = g.size();
        const int vertices_per_block = mapping == CudaMapping::warp_per_vertex ? block_size / 32 : block_size;
        const int blocks = static_cast<int>(std::min<std::size_t>((n - 1) / vertices_per_block + 1, 4096));
        Device<std::size_t> offsets(g.offsets.size()), sources(g.sources.size()), degree(n);
        Device<double> a(n), b(n), partial(blocks), dangling(1), difference(1);
        checked(cudaMemcpy(offsets.p, g.offsets.data(), g.offsets.size() * sizeof(std::size_t), cudaMemcpyHostToDevice), "Upload offsets");
        if (!g.sources.empty())
            checked(cudaMemcpy(sources.p, g.sources.data(), g.sources.size() * sizeof(std::size_t), cudaMemcpyHostToDevice), "Upload sources");
        checked(cudaMemcpy(degree.p, g.out_degree.data(), n * sizeof(std::size_t), cudaMemcpyHostToDevice), "Upload degrees");
        std::size_t temporary_bytes = 0;
        checked(cub::DeviceReduce::Sum(nullptr, temporary_bytes, partial.p, dangling.p, blocks), "Reduction workspace query");
        Device<unsigned char> workspace(temporary_bytes);
        double* current = a.p;
        double* next = b.p;
        initialize<<<blocks, block_size>>>(current, n);
        checked(cudaGetLastError(), "Initialization launch");
        checked(cudaDeviceSynchronize(), "Initialization synchronization");
        output.result.ranks.resize(n);
        const auto compute_start = Clock::now();
        for (std::size_t iteration = 0; iteration < o.max_iterations; ++iteration) {
            dangling_partials<<<blocks, block_size>>>(current, degree.p, n, partial.p);
            checked(cudaGetLastError(), "Dangling launch");
            checked(cub::DeviceReduce::Sum(workspace.p, temporary_bytes, partial.p, dangling.p, blocks), "Dangling reduction");
            if (mapping == CudaMapping::warp_per_vertex)
                pull_warp<<<blocks, block_size>>>(offsets.p, sources.p, degree.p, current, next, dangling.p, n, o.damping, partial.p);
            else
                pull<<<blocks, block_size>>>(offsets.p, sources.p, degree.p, current, next, dangling.p, n, o.damping, partial.p);
            checked(cudaGetLastError(), "Pull launch");
            checked(cub::DeviceReduce::Sum(workspace.p, temporary_bytes, partial.p, difference.p, blocks), "L1 reduction");
            // Blocking download waits for all preceding default-stream work.
            checked(cudaMemcpy(&output.result.l1_difference, difference.p, sizeof(double), cudaMemcpyDeviceToHost), "Download L1");
            std::swap(current, next);
            output.result.iterations = iteration + 1;
            if (output.result.l1_difference < o.tolerance) {
                output.result.converged = true; break;
            }
        }
        output.compute_ms = elapsed(compute_start);
        checked(cudaMemcpy(output.result.ranks.data(), current, n * sizeof(double), cudaMemcpyDeviceToHost), "Download ranks");
    }
    output.solver_end_to_end_ms = elapsed(start);
    return output;
}
}
