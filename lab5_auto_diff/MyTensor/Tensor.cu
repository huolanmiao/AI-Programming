#include "Tensor.h"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <cstddef>

Tensor::Tensor(const std::vector<int>& shape, const std::string & device):
        shape(shape), device(device) {
        int totalSize = 1;
        for (int dim : shape) totalSize *= dim;
        if (device == "GPU") {
            cudaError_t err = cudaMalloc(&data, totalSize*sizeof(float));  // Allocate memory on GPU
            // printf("GPU_cudaMalloc: %s\n", cudaGetErrorString(err));
        } else {
            data = new float[totalSize];  // Allocate memory on CPU
        }
}

Tensor::Tensor(const Tensor& other) {
    if (data) {
        if (device == "GPU") cudaFree(data);
        else delete[] data;
    }
    this->shape = other.shape;
    this->device = other.device;
    int totalSize = 1;
    for (int dim : shape) totalSize *= dim;

    if (device == "GPU") {
        cudaMalloc(&data, totalSize * sizeof(float));  // Allocate memory on GPU
        cudaMemcpy(data, other.data, totalSize * sizeof(float), cudaMemcpyDeviceToDevice);  // Copy data to GPU
    } else {
        this->data = new float[totalSize];  // Allocate memory on CPU
        std::copy(other.data, other.data + totalSize, data);  // Copy data to CPU
    }
}

Tensor::Tensor(py::array_t<float> new_data, const std::string & device): device(device){
    shape = std::vector<int>(new_data.ndim());
    for (int i = 0; i<shape.size(); ++i){
        shape[i] = new_data.shape(i);
    }
    // Allocate memory and Store data
    int totalSize = 1;
    for (int dim : shape) totalSize *= dim;
    if (device == "GPU") {
        cudaError_t err = cudaMalloc(&data, totalSize*sizeof(float));  // Allocate memory on GPU
        // printf("GPU_cudaMalloc: %s\n", cudaGetErrorString(err));
        cudaMemcpy(data, new_data.data(), totalSize * sizeof(float), cudaMemcpyHostToDevice);
    } else {
        data = new float[totalSize];  // Allocate memory on CPU
        std::copy(new_data.data(), new_data.data() + totalSize, data); 
    }
}

py::array_t<float> Tensor::to_numpy(){
    int totalSize = 1;
    for (int dim : shape) totalSize *= dim;
    auto a = py::array_t<float>(totalSize);
    py::buffer_info buf = a.request();
    float* ptr = (float*)buf.ptr; 
    if(device == "GPU"){
        cudaMemcpy(ptr, data, totalSize * sizeof(float), cudaMemcpyDeviceToHost);
    } else{
        std::copy(data, data + totalSize, ptr);
    }
    a.resize(shape);
    return a;
}


void Tensor::set_data(py::array_t<float> new_data){
    // Correct the shape
    int original_size = 1;
    for(int dim : shape) original_size *= dim;

    shape = std::vector<int>(new_data.ndim());
    for (int i = 0; i<shape.size(); ++i){
        shape[i] = new_data.shape(i);
    }
    // Allocate memory and Store data
    int totalSize = 1;
    for (int dim : shape) totalSize *= dim;
    if (totalSize != original_size){
        printf("Reallocate memory!");
        if (device == "GPU") {
            cudaFree(data);  // Free GPU memory
            cudaError_t err = cudaMalloc(&data, totalSize*sizeof(float));  // Allocate memory on GPU
            printf("GPU_cudaMalloc: %s\n", cudaGetErrorString(err));
        } else {
            delete[] data;  // Free CPU memory
            data = new float[totalSize];  // Allocate memory on CPU
        }
    }
    if (device == "GPU") {
        cudaMemcpy(data, new_data.data(), totalSize * sizeof(float), cudaMemcpyHostToDevice);
    } else {
        for (int i = 0; i<totalSize; ++i){
            data[i] = new_data.data()[i];
        }
    }
    
}

void Tensor::show_tensor() const{
    // 打印shape和device
    printf("shape: [");
    for (int i = 0; i < shape.size(); i++) {
        printf("%d", shape[i]);
        if (i < shape.size() - 1) printf(", ");
    }
    printf("]\n");
    printf("device: %s\n", device);
    int totalSize = 1;
    for (int dim : shape) totalSize *= dim;
    if (device == "GPU") {
        // 打印GPU上的数据
        float* cpu_data = new float[totalSize];
        cudaMemcpy(cpu_data, data, totalSize * sizeof(float), cudaMemcpyDeviceToHost);
        if (shape.size() == 2){
            for(int i = 0; i < shape[0]; i++) {
                for(int j = 0; j < shape[1]; j++) {
                    printf("%f ", cpu_data[i*shape[1] + j]);
                }
                printf("\n");
            }
        } else{
            for(int i = 0; i<totalSize; i++){
                printf("%f ", cpu_data[i]);
            }
            printf("\n");
        }
        delete[] cpu_data;
    } else {
        // 打印CPU上的数据
        if(shape.size() == 2){
            for(int i = 0; i < shape[0]; i++) {
                for(int j = 0; j < shape[1]; j++) {
                    printf("%f ", data[i*shape[1] + j]);
                }
                printf("\n");
            }
        } else{
            for(int i = 0; i<totalSize; i++){
                printf("%f ", data[i]);
            }
            printf("\n");
        }
        
    }
        
}


// 析构函数
Tensor::~Tensor() {
    if (device == "GPU") {
        cudaFree(data);  // Free GPU memory
    } else {
        delete[] data;  // Free CPU memory
    }
}

// 将数据复制到 CPU
Tensor Tensor::cpu() {
    // Copy data to CPU 
    if (device == "CPU") return *this;
    else {
        // printf("Data coping to CPU.\n");
        int totalSize = 1;
        for (int dim : shape) totalSize *= dim;
        float* cpu_data = new float[totalSize];  // Allocate memory on CPU
        cudaError_t err_cpy = cudaMemcpy(cpu_data, this->data, totalSize*sizeof(float), cudaMemcpyDeviceToHost);  // Copy data to CPU
        // printf("cudaMemcpy: %s\n", cudaGetErrorString(err_cpy));
        cudaError_t err_free = cudaFree(this->data);  // Free GPU memory
        // printf("cudaFree: %s\n", cudaGetErrorString(err_free));
        this->data = cpu_data;
        this->device = "CPU";
        // printf("Data copied to CPU.\n\n");
        return *this;
    }        
}
// 将数据复制到 GPU
Tensor Tensor::gpu() {
    // Copy data to GPU 
    // std::cout << "Data copied to GPU." << std::endl;
    if (device == "GPU") return *this;
    else {
        // printf("Data coping to GPU.\n");
        int totalSize = 1;
        for (int dim : shape) totalSize *= dim;
        float* gpu_data;
        cudaError_t err_malloc = cudaMalloc(&gpu_data, totalSize*sizeof(float));  // Allocate memory on GPU
        // printf("cudaMalloc: %s\n", cudaGetErrorString(err_malloc));
        cudaError_t err_cpy = cudaMemcpy(gpu_data, this->data, totalSize*sizeof(float), cudaMemcpyHostToDevice);  // Copy data to GPU
        // printf("cudaMemcpy: %s\n", cudaGetErrorString(err_cpy));
        delete[] this->data;  // Free CPU memory
        this->data = gpu_data;
        this->device = "GPU";
        // printf("Data copied to GPU.\n\n");
        return *this;
    }        
}
