// #ifndef TENSOR_H
// #define TENSOR_H

// #include <vector>
// #include <iostream>
// #include <cuda_runtime.h>

// // 枚举类型，表示设备
// enum Device { CPU, GPU };

// class Tensor {
// public:
//     // 数据成员
//     std::vector<int> shape;
//     Device device;
//     float* data = nullptr;  // 指向数据的指针

//     // 构造函数和析构函数
//     Tensor(const std::vector<int>& shape, Device device);
//     Tensor(const Tensor& other);  // 复制构造函数
//     Tensor& operator=(const Tensor& other);  // 复制赋值构造函数
//     ~Tensor();  // 析构函数

//     // 显示张量信息
//     void show_tensor() const;

//     // 数据迁移函数
//     Tensor cpu();
//     Tensor gpu();
// };

// // CUDA内核函数声明
// __global__ void relu_gpu(float *in, float *out, int n);
// __global__ void sigmoid_gpu(float *in, float *out, int n);
// __global__ void relu_grad_gpu(float* grad_data, const float* result_data, const float* pre_grad_data, int n);
// __global__ void sigmoid_grad_gpu(float* grad_data, const float* result_data, const float* pre_grad_data, int n);

// // CPU实现的函数声明
// void relu_cpu(float *in, float *out, int n);
// void sigmoid_cpu(float *in, float *out, int n);
// void relu_grad_cpu(float* grad_data, const float* result_data, const float* pre_grad_data, int n);
// void sigmoid_grad_cpu(float* grad_data, const float* result_data, const float* pre_grad_data, int n);

// // 操作函数声明
// Tensor relu(const Tensor& input);
// Tensor relu_grad(const Tensor& pre_grad, const Tensor& output);
// Tensor sigmoid(const Tensor& input);
// Tensor sigmoid_grad(const Tensor& pre_grad, const Tensor& output);

// #endif // TENSOR_H


#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <iostream>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // To support std::vector
#include <pybind11/numpy.h>
#include <memory>
#include <cstddef>
#include "ai_programming.h"

namespace py = pybind11;

// void printArray(const float *arr, const int *shape, int dim, int index);
class Tensor {
    
public:
    // 数据成员
    std::vector<int> shape;
    std::string device;
    float* data = nullptr;  // 指向数据的指针

    Tensor(const std::vector<int>& shape, const std::string & device);
    Tensor(const Tensor& other);
    Tensor(py::array_t<float> data, const std::string & device);
    // Tensor& operator=(const Tensor& other);
    // static Tensor create(const std::vector<int>& shape, const std::string & device){return Tensor(shape, device);}
    // 构造函数和析构函数
    // Tensor(const std::vector<int>& shape, std::string device);
    // Tensor(const Tensor& other);  // 复制构造函数
    // Tensor& operator=(const Tensor& other);  // 复制赋值构造函数
    
    py::array_t<float> to_numpy();
    void set_data(py::array_t<float> data);
    
    ~Tensor();  // 析构函数

    // 显示张量信息
    void show_tensor() const;
    // std::vector<size_t> get_strides();
    // 数据迁移函数
    Tensor cpu();
    Tensor gpu();
};

// py::array_t<float> Tensor2Numpy(const Tensor & t);

#endif // TENSOR_H

