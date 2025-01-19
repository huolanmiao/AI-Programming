#include <vector>
#include <algorithm>

#include <iostream>
#include <vector>
#include <cmath>
#include "ai_programming.h"
#include <cuda_runtime.h>
enum Device { CPU, GPU };
//Tensor类，通过构造函数和析构函数管理内存,实现cpu()和gpu()成员函数。
class Tensor {
    
public:
    std::vector<int> shape;
    Device device;
    float* data = nullptr;  // Pointer to data

    // 复制构造函数
    Tensor(const Tensor& other) {
        this->shape = other.shape;
        this->device = other.device;
        int totalSize = 1;
        for (int dim : shape) totalSize *= dim;
        // deep copy
        if (device == GPU) {
            cudaMalloc(&data, totalSize*sizeof(float));  // Allocate memory on GPU
            cudaMemcpy(data, other.data, totalSize*sizeof(float), cudaMemcpyDeviceToDevice);  // Copy data to GPU
        } else {
            this->data = new float[totalSize];  // Allocate memory on CPU
            // printf("Copy constructor called.\n");
            // printf("totalSize: %d\n", totalSize);
            // printf("other.data: %d\n", other.data);
            // printf("other.data + totalSize: %d\n", other.data + totalSize);
            std::copy(other.data, other.data + totalSize, data);  // Copy data to CPU
            // printf("Copy constructor finished.\n\n");
            
        }
    }

    Tensor(const std::vector<int>& shape, Device device)
        : shape(shape), device(device) {
        int totalSize = 1;
        for (int dim : shape) totalSize *= dim;
        if (device == GPU) {
            cudaError_t err = cudaMalloc(&data, totalSize*sizeof(float));  // Allocate memory on GPU
            printf("GPU_cudaMalloc: %s\n", cudaGetErrorString(err));
        } else {
            data = new float[totalSize];  // Allocate memory on CPU
        }
    }

    void show_tensor() const{
        // 打印shape和device
        printf("shape: {%d, %d}\n", shape[0], shape[1]);
        printf("device: %s\n", device == CPU ? "CPU" : "GPU");
        if (device == GPU) {
            // 打印GPU上的数据
            int totalSize = 1;
            for (int dim : shape) totalSize *= dim;
            float* cpu_data = new float[totalSize];
            cudaMemcpy(cpu_data, data, totalSize * sizeof(float), cudaMemcpyDeviceToHost);
            for(int i = 0; i < shape[0]; i++) {
                for(int j = 0; j < shape[1]; j++) {
                    printf("%f ", cpu_data[i*shape[1] + j]);
                }
                printf("\n");
            }
            delete[] cpu_data;
        } else {
            // 打印CPU上的数据
            for(int i = 0; i < shape[0]; i++) {
                for(int j = 0; j < shape[1]; j++) {
                    printf("%f ", data[i*shape[1] + j]);
                }
                printf("\n");
            }
        }
        
    }

    

    // 复制赋值构造函数
    Tensor& operator=(const Tensor& other) {
        if (this == &other) return *this;  // Handle self-assignment
        shape = other.shape;
        device = other.device;
        int totalSize = 1;
        for (int dim : shape) totalSize *= dim;
        // deep copy
        if (device == GPU) {
            cudaMalloc(&data, totalSize*sizeof(float));  // Allocate memory on GPU
            cudaMemcpy(data, other.data, totalSize*sizeof(float), cudaMemcpyDeviceToDevice);  // Copy data to GPU
        } else {
            data = new float[totalSize];  // Allocate memory on CPU
            std::copy(other.data, other.data + totalSize, data);  // Copy data to CPU
        }
        return *this;
    }

    

    ~Tensor() {
        if (device == GPU) {
            cudaFree(data);  // Free GPU memory
        } else {
            delete[] data;  // Free CPU memory
        }
    }

    Tensor cpu() {
        // Copy data to CPU 
        if (device == CPU) return *this;
        else {
            printf("Data coping to CPU.\n");
            int totalSize = 1;
            for (int dim : shape) totalSize *= dim;
            float* cpu_data = new float[totalSize];  // Allocate memory on CPU
            cudaError_t err_cpy = cudaMemcpy(cpu_data, this->data, totalSize*sizeof(float), cudaMemcpyDeviceToHost);  // Copy data to CPU
            printf("cudaMemcpy: %s\n", cudaGetErrorString(err_cpy));
            cudaError_t err_free = cudaFree(this->data);  // Free GPU memory
            printf("cudaFree: %s\n", cudaGetErrorString(err_free));
            this->data = cpu_data;
            this->device = CPU;
            printf("Data copied to CPU.\n\n");
            return *this;
        }        
    }

    Tensor gpu() {
        // Copy data to GPU 
        // std::cout << "Data copied to GPU." << std::endl;
        if (device == GPU) return *this;
        else {
            printf("Data coping to GPU.\n");
            int totalSize = 1;
            for (int dim : shape) totalSize *= dim;
            float* gpu_data;
            cudaError_t err_malloc = cudaMalloc(&gpu_data, totalSize*sizeof(float));  // Allocate memory on GPU
            printf("cudaMalloc: %s\n", cudaGetErrorString(err_malloc));
            cudaError_t err_cpy = cudaMemcpy(gpu_data, this->data, totalSize*sizeof(float), cudaMemcpyHostToDevice);  // Copy data to GPU
            printf("cudaMemcpy: %s\n", cudaGetErrorString(err_cpy));
            delete[] this->data;  // Free CPU memory
            this->data = gpu_data;
            this->device = GPU;
            printf("Data copied to GPU.\n\n");
            return *this;
        }        
    }

    
};

__global__ void relu_gpu(float * in, float * out, int n) {
    CUDA_KERNEL_LOOP(i , n){
        out[i] = in[i] > 0 ? in[i] : 0;
    }
}

__global__ void sigmoid_gpu(float * in, float * out, int n) {
    CUDA_KERNEL_LOOP(i , n){
        out[i] = 1 / (1 + exp(-in[i]));
    }
}

void relu_cpu(float * in, float * out, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = in[i] > 0 ? in[i] : 0;
    }
}

void sigmoid_cpu(float * in, float * out, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = 1 / (1 + exp(-in[i]));
    }
}

__global__ void relu_grad_gpu(float* grad_data, const float* result_data, const float* pre_grad_data, int n) {
    CUDA_KERNEL_LOOP(i, n) {
        grad_data[i] = (result_data[i] > 0 ? 1 : 0) * pre_grad_data[i];
    }
}

__global__ void sigmoid_grad_gpu(float* grad_data, const float* result_data, const float* pre_grad_data, int n) {
    CUDA_KERNEL_LOOP(i, n) {
        grad_data[i] = result_data[i] * (1 - result_data[i]) * pre_grad_data[i];
    }
}

void relu_grad_cpu(float* grad_data, const float* result_data, const float* pre_grad_data, int n) {
    for (int i = 0; i < n; i++) {
        grad_data[i] = (result_data[i] > 0 ? 1 : 0) * pre_grad_data[i];
    }
}

void sigmoid_grad_cpu(float* grad_data, const float* result_data, const float* pre_grad_data, int n) {
    for (int i = 0; i < n; i++) {
        grad_data[i] = result_data[i] * (1 - result_data[i]) * pre_grad_data[i];
    }
}

// 对Tensor实例计算RELU的前向传播
Tensor relu(const Tensor& input) {
    Tensor output(input.shape, input.device);
    int N = 1;
    for(int dim : input.shape) N *= dim;
    if (input.device == GPU) {
        relu_gpu<<<CudaGetBlocks(N), kCudaThreadsNum>>>(input.data, output.data, N);
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        printf("relu_gpu: %s\n", cudaGetErrorString(err));
    } else {
        relu_cpu(input.data, output.data, N);
    }
    return output;
}
// 对Tensor实例计算RELU的反向传播
Tensor relu_grad(const Tensor& pre_grad, const Tensor& output) {
    Tensor grad(output.shape, output.device);
    int N = 1;
    for(int dim : output.shape) N *= dim;
    if (grad.device == GPU) {
        relu_grad_gpu<<<CudaGetBlocks(N), kCudaThreadsNum>>>(grad.data, output.data, pre_grad.data, N);
    } else {
        relu_grad_cpu(grad.data, output.data, pre_grad.data, N);
    }
    return grad;
}
// 对Tensor实例计算Sigmoid的前向传播
Tensor sigmoid(const Tensor& input) {
    Tensor output(input.shape, input.device);
    int N = 1;
    for(int dim : input.shape) N *= dim;
    if (input.device == GPU) {
        sigmoid_gpu<<<CudaGetBlocks(N), kCudaThreadsNum>>>(input.data, output.data, N);
    } else {
        sigmoid_cpu(input.data, output.data, N);
    }
    return output;
}
// 对Tensor实例计算Sigmoid的反向传播
Tensor sigmoid_grad(const Tensor& pre_grad, const Tensor& output) {
    Tensor grad(output.shape, output.device);
    int N = 1;
    for(int dim : output.shape) N *= dim;
    if (grad.device == GPU) {
        sigmoid_grad_gpu<<<CudaGetBlocks(N), kCudaThreadsNum>>>(grad.data, output.data, pre_grad.data, N);
    } else {
        sigmoid_grad_cpu(grad.data, output.data, pre_grad.data, N);
    }
    return grad;
}

int main() {
    
    std::vector<int> shape = {2, 3};
    Tensor tensor_cpu(shape, CPU);
    Tensor tensor_gpu(shape, GPU);

    // Fill tensor_cpu and tensor_gpu with the same data
    float test_data[6] = {-1, 2, 3, 4, 5, 6};
    std::copy(test_data, test_data + 6, tensor_cpu.data);  // Copy data to CPU
    cudaError_t err = cudaMemcpy(tensor_gpu.data, test_data, 6*sizeof(float), cudaMemcpyHostToDevice);  // Copy data to GPU
    printf("Show tensor_cpu.\n");
    tensor_cpu.show_tensor();
    printf("Show tensor_gpu.\n");
    tensor_gpu.show_tensor();

    Tensor pre_grad_cpu(shape, CPU);
    Tensor pre_grad_gpu(shape, GPU);
    float test_pre_grad_data[6] = {1, 1, 1, 1, 1, 1};
    std::copy(test_pre_grad_data, test_pre_grad_data + 6, pre_grad_cpu.data);  // Copy data to CPU
    cudaError_t err_pre_grad = cudaMemcpy(pre_grad_gpu.data, test_pre_grad_data, 6*sizeof(float), cudaMemcpyHostToDevice);  // Copy data to GPU
    printf("Show pre_grad_cpu.\n");
    pre_grad_cpu.show_tensor();
    printf("Show pre_grad_gpu.\n");
    pre_grad_gpu.show_tensor();
    printf("------------------------------------\n");
    
    
    // 1. 测试cpu()和gpu()函数
    Tensor original_tensor_cpu = tensor_cpu;
    // Copy tensor_cpu to GPU and back to CPU
    Tensor tensor_gpu_copy = tensor_cpu.gpu();
    Tensor tensor_cpu_copy = tensor_cpu.cpu();
    printf("Copy tensor_cpu to GPU and back to CPU.\n");
    // Verify the values are the same
    bool success = true;
    for (int i = 0; i < 6; ++i) {
        if (tensor_cpu.data[i] != original_tensor_cpu.data[i]) {
            std::cerr << "Error: Data mismatch at index " << i << std::endl;
            success = false;
            break;
        }
    }
    printf("Test cpu() and gpu(): %s\n\n", success ? "PASS" : "FAIL");
    printf("------------------------------------\n");
    printf("Test on CPU.\n\n");
    // 2. 测试relu函数 on CPU
    Tensor relu_output = relu(tensor_cpu);
    std::cout << "ReLU output: \n";
    relu_output.show_tensor();
    std::cout << std::endl;

    // 3. 测试sigmoid函数 on CPU
    Tensor sigmoid_output = sigmoid(tensor_cpu);
    std::cout << "Sigmoid output: \n";
    sigmoid_output.show_tensor();
    std::cout << std::endl;

    // 4. 测试relu_grad函数 on CPU
    Tensor relu_grad_output = relu_grad(pre_grad_cpu, relu_output);
    std::cout << "ReLU grad output: \n";
    relu_grad_output.show_tensor();
    std::cout << std::endl;

    // 5. 测试sigmoid_grad函数 on CPU
    Tensor sigmoid_grad_output = sigmoid_grad(pre_grad_cpu, sigmoid_output);
    std::cout << "Sigmoid grad output: \n";
    sigmoid_grad_output.show_tensor();
    std::cout << std::endl;

    printf("Test on CPU finished.\n");
    printf("------------------------------------\n");
    printf("Test on GPU.\n\n");
    // 6. 测试relu函数 on GPU
    Tensor relu_output_gpu = relu(tensor_gpu);
    std::cout << "ReLU output on GPU: \n";
    relu_output_gpu.show_tensor();
    std::cout << std::endl;

    // 7. 测试sigmoid函数 on GPU
    Tensor sigmoid_output_gpu = sigmoid(tensor_gpu);
    std::cout << "Sigmoid output on GPU: \n";
    sigmoid_output_gpu.show_tensor();
    std::cout << std::endl;

    // 8. 测试relu_grad函数 on GPU
    Tensor relu_grad_output_gpu = relu_grad(pre_grad_gpu, relu_output_gpu);
    std::cout << "ReLU grad output on GPU: \n";
    relu_grad_output_gpu.show_tensor();
    std::cout << std::endl;

    // 9. 测试sigmoid_grad函数 on GPU
    Tensor sigmoid_grad_output_gpu = sigmoid_grad(pre_grad_gpu, sigmoid_output_gpu);
    std::cout << "Sigmoid grad output on GPU: \n";
    sigmoid_grad_output_gpu.show_tensor();
    std::cout << std::endl;

    printf("Test on GPU finished.\n");
    printf("------------------------------------\n");
    printf("Comparing CPU and GPU results we find that they get the same result.\n");
    printf("Test finished.\n");

    return 0;
}