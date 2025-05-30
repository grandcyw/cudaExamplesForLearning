#include<iostream>
using namespace std;

__global__ void add_atomic(int *a)
{
    int local=__ldg(a);
    printf("Thread %d read value: %d\n", threadIdx.x, local);
    atomicAdd(a,1);
    // __syncthreads();
}

int main()
{
    int *a;
    cudaMallocManaged((void**)&a,sizeof(int));
    *a=4;
    add_atomic<<<1,2>>>(a);
    cudaError_t error=cudaGetLastError();
    if(error)
    {
        printf("Error: %s\n", cudaGetErrorString(error));
    }
    cudaDeviceSynchronize();
    printf("Result: %d\n", *a);
    cudaFree(a);
    return 0;
}