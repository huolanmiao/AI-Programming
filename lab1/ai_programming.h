// Use 512 or 256 threads per block
#include<device_launch_parameters.h>
#include <cuda_runtime.h>
const int kCudaThreadsNum = 512;
inline int CudaGetBlocks (const int N ){
    return (N + kCudaThreadsNum - 1 ) / kCudaThreadsNum;
}
// Define the grid stride looping
#define CUDA_KERNEL_LOOP( i , n ) \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
        i < (n); \
        i += blockDim.x * gridDim.x)

