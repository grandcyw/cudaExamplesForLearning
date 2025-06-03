#include <random>
#include <chrono>


#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <iostream>

#define THREADS_PER_BLOCK 256




__global__ void bfs_kernel(
    const int* __restrict__ edges, 
    const int* __restrict__ offsets,
    int* __restrict__ distances,
    int* __restrict__ frontier,
    int* __restrict__ frontier_size,
    int current_distance,
    int num_nodes) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= *frontier_size) return;

    int node = frontier[idx];
    int start = offsets[node];
    int end = offsets[node + 1];

    for (int i = start; i < end; ++i) {
        int neighbor = edges[i];
        if (atomicMin(&distances[neighbor], current_distance + 1) == INT_MAX) {
            int pos = atomicAdd(frontier_size, 1);
            frontier[pos] = neighbor;
        }
    }
}


__global__ void bfs_expand_kernel(
    const int* edges, const int* offsets,
    const int* frontier, int frontier_size,
    int* distances, int current_distance,
    int* new_frontier, int* new_size) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= frontier_size) return;

    int node = frontier[idx];
    int start = offsets[node];
    int end = offsets[node + 1];

    for (int i = start; i < end; ++i) {
        int neighbor = edges[i];
        if (atomicMin(&distances[neighbor], current_distance) == INT_MAX) {
            int pos = atomicAdd(new_size, 1);
            new_frontier[pos] = neighbor;
        }
    }
}

void bfs_cuda_optimized(...) {
    // 初始化同上...
    
    thrust::device_vector<int> frontier[2];
    frontier[0].resize(offsets.size() - 1);
    frontier[1].resize(offsets.size() - 1);
    int current_frontier = 0;
    frontier[current_frontier][0] = start_node;
    int frontier_size = 1;

    while (frontier_size > 0) {
        int new_size = 0;
        int blocks = (frontier_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        
        bfs_expand_kernel<<<blocks, THREADS_PER_BLOCK>>>(
            /* 参数传递 */);
        
        cudaDeviceSynchronize();
        frontier_size = new_size;
        current_frontier ^= 1; // 切换frontier缓冲区
        current_distance++;
    }
}


void bfs_cuda(const thrust::device_vector<int>& edges,
             const thrust::device_vector<int>& offsets,
             thrust::device_vector<int>& distances,
             int start_node) {
    
    thrust::fill(distances.begin(), distances.end(), INT_MAX);
    distances[start_node] = 0;

    thrust::device_vector<int> frontier(offsets.size() - 1);
    thrust::device_vector<int> frontier_size(1, 1);
    frontier[0] = start_node;

    int current_distance = 0;
    while (frontier_size[0] > 0) {
        int blocks = (frontier_size[0] + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        bfs_kernel<<<blocks, THREADS_PER_BLOCK>>>(
            thrust::raw_pointer_cast(edges.data()),
            thrust::raw_pointer_cast(offsets.data()),
            thrust::raw_pointer_cast(distances.data()),
            thrust::raw_pointer_cast(frontier.data()),
            thrust::raw_pointer_cast(frontier_size.data()),
            current_distance,
            offsets.size() - 1);
        
        cudaDeviceSynchronize();
        frontier_size[0] = 0; // Reset for next level
        current_distance++;
    }
}

void generate_random_graph(thrust::host_vector<int>& edges,
                          thrust::host_vector<int>& offsets,
                          int num_nodes, int avg_degree) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, num_nodes - 1);

    offsets.resize(num_nodes + 1);
    int total_edges = num_nodes * avg_degree;
    edges.resize(total_edges);

    offsets[0] = 0;
    for (int i = 0; i < num_nodes; ++i) {
        int degree = avg_degree;
        offsets[i + 1] = offsets[i] + degree;
        for (int j = 0; j < degree; ++j) {
            edges[offsets[i] + j] = dis(gen);
        }
    }
}

void test_bfs_performance() {
    const int num_nodes = 1 << 20; // 1M节点
    const int avg_degree = 8;
    
    // 生成随机图
    thrust::host_vector<int> h_edges, h_offsets;
    generate_random_graph(h_edges, h_offsets, num_nodes, avg_degree);
    
    // 拷贝到设备
    thrust::device_vector<int> d_edges = h_edges;
    thrust::device_vector<int> d_offsets = h_offsets;
    thrust::device_vector<int> d_distances(num_nodes);
    
    // 测试原子操作版本
    auto start = std::chrono::high_resolution_clock::now();
    bfs_cuda(d_edges, d_offsets, d_distances, 0);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Atomic BFS time: " 
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() 
              << " ms\n";
    
    // 测试优化版本
    start = std::chrono::high_resolution_clock::now();
    bfs_cuda_optimized(d_edges, d_offsets, d_distances, 0);
    end = std::chrono::high_resolution_clock::now();
    std::cout << "Optimized BFS time: " 
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() 
              << " ms\n";
}