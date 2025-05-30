#include <iostream>
#include <cuda_runtime.h>

// 错误检查宏
#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        exit(1);
    }
}

void printGPUDeviceParameters(int device_id = 0) {
    cudaDeviceProp device_prop;
    CHECK_CUDA_ERROR(cudaGetDeviceProperties(&device_prop, device_id));

    std::cout << "\n========== GPU Device " << device_id << " Parameters ==========\n";
    
    // 基础信息
    std::cout << "Device Name: " << device_prop.name << "\n";
    std::cout << "Compute Capability: " << device_prop.major << "." << device_prop.minor << "\n";
    std::cout << "Clock Rate: " << device_prop.clockRate / 1000 << " MHz\n";
    std::cout << "Device Overlap: " << (device_prop.deviceOverlap ? "Yes" : "No") << "\n";
    std::cout << "Kernel Execution Timeout: " << (device_prop.kernelExecTimeoutEnabled ? "Yes" : "No") << "\n";
    
    // 内存信息
    std::cout << "\n------ Memory Information ------\n";
    std::cout << "Total Global Memory: " << device_prop.totalGlobalMem / (1024 * 1024) << " MB\n";
    std::cout << "Total Constant Memory: " << device_prop.totalConstMem / 1024 << " KB\n";
    std::cout << "Max Shared Memory per Block: " << device_prop.sharedMemPerBlock / 1024 << " KB\n";
    std::cout << "Memory Pitch: " << device_prop.memPitch / (1024 * 1024) << " MB\n";
    std::cout << "Texture Alignment: " << device_prop.textureAlignment / 1024 << " KB\n";
    
    // 线程和块信息
    std::cout << "\n------ Thread/Block Information ------\n";
    std::cout << "Max Threads per Block: " << device_prop.maxThreadsPerBlock << "\n";
    std::cout << "Max Threads Dim: (" << device_prop.maxThreadsDim[0] << ", " 
                                     << device_prop.maxThreadsDim[1] << ", " 
                                     << device_prop.maxThreadsDim[2] << ")\n";
    std::cout << "Max Grid Size: (" << device_prop.maxGridSize[0] << ", " 
                                  << device_prop.maxGridSize[1] << ", " 
                                  << device_prop.maxGridSize[2] << ")\n";
    std::cout << "Warp Size: " << device_prop.warpSize << "\n";
    
    // 多处理器信息
    std::cout << "\n------ Multiprocessor Information ------\n";
    std::cout << "Multiprocessor Count: " << device_prop.multiProcessorCount << "\n";
    std::cout << "Max Threads per Multiprocessor: " << device_prop.maxThreadsPerMultiProcessor << "\n";
    std::cout << "Max Registers per Block: " << device_prop.regsPerBlock << "\n";
    std::cout << "Registers per Multiprocessor: " << device_prop.regsPerMultiprocessor << "\n";
    
    // 其他特性
    std::cout << "\n------ Features ------\n";
    std::cout << "Concurrent Kernels: " << (device_prop.concurrentKernels ? "Yes" : "No") << "\n";
    std::cout << "Integrated GPU: " << (device_prop.integrated ? "Yes" : "No") << "\n";
    std::cout << "Can Map Host Memory: " << (device_prop.canMapHostMemory ? "Yes" : "No") << "\n";
    std::cout << "ECC Enabled: " << (device_prop.ECCEnabled ? "Yes" : "No") << "\n";
    std::cout << "Unified Addressing: " << (device_prop.unifiedAddressing ? "Yes" : "No") << "\n";
    
    std::cout << "=======================================\n\n";
}

int printInfo() {
    int device_count;
    CHECK_CUDA_ERROR(cudaGetDeviceCount(&device_count));
    
    if (device_count == 0) {
        std::cerr << "No CUDA-capable devices found!" << std::endl;
        return 1;
    }
    
    std::cout << "Found " << device_count << " CUDA-capable device(s)\n";
    
    for (int i = 0; i < device_count; ++i) {
        printGPUDeviceParameters(i);
    }
    
    return 0;
}