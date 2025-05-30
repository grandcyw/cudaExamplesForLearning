//nvcc -rdc=true  merge_sort.cu -o merge_sort -lcudadevrt启用动态并行在核函数中发射子kernel

#include <iostream>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include"printGpuInfo.cu"

// #define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
// template <typename T>
// void check(T err, const char* const func, const char* const file, const int line) {
//     if (err != cudaSuccess) {
//         std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
//         std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
//         exit(1);
//     }
// }

// // 归并两个有序序列的核函数
// __global__ void merge_kernel(int* arr, int* temp, int n, int size) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx >= n) return;

//     int start = idx * size * 2;
//     int mid = start + size;
//     int end = min(start + size * 2, n);

//     int i = start;
//     int j = mid;
//     int k = start;

//     while (i < mid && j < end) {
//         if (arr[i] <= arr[j]) {
//             temp[k++] = arr[i++];
//         } else {
//             temp[k++] = arr[j++];
//         }
//     }

//     while (i < mid) {
//         temp[k++] = arr[i++];
//     }

//     while (j < end) {
//         temp[k++] = arr[j++];
//     }

//     // 将结果拷贝回原数组
//     for (int m = start; m < end; m++) {
//         arr[m] = temp[m];
//     }
// }

// 归并排序的核函数
template<typename TType>
__device__ void swap_device_func(TType* element1, TType* element2) {
    TType temp = *element1;
    *element1 = *element2;
    *element2 = temp;
}

template<typename TType>
__global__ void print_subkernel() {
    extern __shared__ TType shared_data[];
    // printf("swap_device_func called with shared memory size %d\n", sizeof(shared_data));
    printf("using thread %d\n",blockIdx.x*blockDim.x+threadIdx.x);
    printf("swap_device_func called\n");
}
// 归并排序的核函数
template<typename TType>
__global__ void merge_sort_kernel(TType* arr,int start, int end, int n,int load) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx*2+1 >= n) return;
    TType left=__ldg(arr+idx*2);
    TType right=__ldg(arr+idx*2+1);
    // printf("Thread %d: swapping %d and %d\n", idx, left, right);
    if (left < right) {
        // 交换元素
        // printf("Thread %d: swapping %d and %d\n", idx, left, right);
        // arr[idx*2] = right;
        // arr[idx*2 + 1] = left;
        swap_device_func<TType>(arr + idx * 2, arr + idx * 2 + 1);
    }
    print_subkernel<TType><<<2,256,12>>>();
    // __syncthreads();

}

// CUDA归并排序
template<typename TType>
void cuda_merge_sort(TType* h_arr, int n) {
    TType* d_arr;
    
    // 分配设备内存
    CHECK_CUDA_ERROR(cudaMalloc(&d_arr, n * sizeof(int)));

    // 拷贝数据到设备
    CHECK_CUDA_ERROR(cudaMemcpy(d_arr, h_arr, n * sizeof(TType), cudaMemcpyHostToDevice));

    // 配置线程块和网格
    int block_size = 256;
    int grid_size=1;

    printInfo();
    printf("Launch config: grid_size=%d, block_size=%d, n=%d\n", grid_size, block_size, n);
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    merge_sort_kernel<<<grid_size, block_size>>>(d_arr,0, n-1, n,2);
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    float milliseconds = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));
    std::cout << "Kernel execution time: " << milliseconds << " ms" << std::endl;
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));
    // cudaDeviceSynchronize();
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    // 首先对每个元素进行排序（每个元素视为已排序的序列）
    // for (int size = 1; size < n; size *= 2) {
    //     grid_size = n / (size * 2) + 1;
        
    //     if (size == 1) {
    //         // 初始阶段，每个元素视为已排序
    //         // 不需要做任何操作，直接进入下一阶段
    //         continue;
    //     } else {
    //         // 执行归并排序
    //         printf("Launch config: grid_size=%d, block_size=%d, size=%d\n", grid_size, block_size, size);
    //         merge_sort_kernel<<<grid_size, block_size>>>(d_arr, d_temp, n, size);
    //         CHECK_CUDA_ERROR(cudaGetLastError());
    //         CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    //     }
    // }

    // 拷贝结果回主机
    CHECK_CUDA_ERROR(cudaMemcpy(h_arr, d_arr, n * sizeof(int), cudaMemcpyDeviceToHost));

    // 释放设备内存
    CHECK_CUDA_ERROR(cudaFree(d_arr));
}

// 验证排序结果
template<typename TType>
bool verify_sorted(const std::vector<TType>& arr) {
    for (size_t i = 1; i < arr.size(); ++i) {
        if (arr[i - 1] > arr[i]) {
            std::cerr << "Sorting verification failed at index " << i - 1 
                      << ": " << arr[i - 1] << " > " << arr[i] << std::endl;
            return false;
        }
    }
    return true;
}

// 生成随机数
std::vector<int> generate_random_array(size_t size) {
    std::vector<int> arr(size);
    for (size_t i = 0; i < size; ++i) {
        arr[i] = rand() % 1000; // 0-999的随机数
    }
    return arr;
}

// 生成递增数
template<typename TType>
void generate_fixed_array_ascend(std::vector<TType>& arr) {
    size_t size=arr.size();
    for (size_t i = 0; i < size; ++i) {
        arr[i] = static_cast<TType>(i) / static_cast<TType>(1000); // 0-999的随机数
    }
    return;
}

// 打印数组
template<typename TType>
void print_array(const std::vector<TType>& arr, size_t max_print = 21) {
    std::cout << "Array (first " << std::min(max_print, arr.size()) << " elements): ";
    for (size_t i = 0; i < std::min(max_print, arr.size()); ++i) {
        std::cout << arr[i] << " ";
    }
    std::cout << std::endl;
}

int main() {
    const size_t n = 1 << 10; // 1M元素
    using TType=float;

    std::cout << "Sorting " << n << " elements..." << std::endl;

    // 生成随机数组
    std::vector<TType> arr(n);
    generate_fixed_array_ascend<TType>(arr);
    std::vector<TType> arr_cpu = arr; // 用于CPU排序验证

    // 打印原始数组（前20个元素）
    std::cout << "\nOriginal array:" << std::endl;
    print_array<TType>(arr);

    // 执行CUDA归并排序
    cuda_merge_sort<TType>(arr.data(), n);

    // 打印排序后的数组（前20个元素）
    std::cout << "\nAfter CUDA merge sort:" << std::endl;
    print_array<TType>(arr);

    // 验证排序结果
    std::cout << "\nVerifying CUDA sort result..." << std::endl;
    bool cuda_sorted = verify_sorted<TType>(arr);
    std::cout << "CUDA merge sort " << (cuda_sorted ? "succeeded" : "failed") << std::endl;

    // 使用CPU排序进行验证
    std::cout << "\nRunning CPU sort for verification..." << std::endl;
    std::sort(arr_cpu.begin(), arr_cpu.end());

    // 比较结果
    bool match = (arr == arr_cpu);
    std::cout << "CUDA result " << (match ? "matches" : "does not match") << " CPU result" << std::endl;

    if (!match) {
        // 找出第一个不匹配的位置
        for (size_t i = 0; i < n; ++i) {
            if (arr[i] != arr_cpu[i]) {
                std::cerr << "First mismatch at index " << i 
                          << ": CUDA=" << arr[i] << ", CPU=" << arr_cpu[i] << std::endl;
                break;
            }
        }
    }

    return cuda_sorted && match ? 0 : 1;
}