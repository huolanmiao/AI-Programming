#ifndef FULLY_CONNECTED_H
#define FULLY_CONNECTED_H

#include <cublas_v2.h>
#include <curand.h>
#include <iostream>
#include <cuda_runtime.h>
#include "Tensor.h"



// 函数声明
void gemm_gpu(cublasOperation_t transA, cublasOperation_t transB,
              int m, int n, int k,
              float a, const float* A, int lda,
              const float* B, int ldb,
              float b, float* C, int ldc);

void forward_fc(const Tensor& input, const Tensor& weights,
                const Tensor& bias, Tensor& output);

void backward_fc(const Tensor& input, const Tensor& weights,
                 const Tensor& bias, const Tensor& grad_output,
                 Tensor& grad_input, Tensor& grad_weights, Tensor& grad_bias);

void matrix_init(float* matrix, const std::vector<int> &shape);




#endif // FULLY_CONNECTED_H
