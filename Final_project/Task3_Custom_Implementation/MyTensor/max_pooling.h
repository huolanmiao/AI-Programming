#ifndef MAX_POOLING_H
#define MAX_POOLING_H



#include <float.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <iostream>
#include "ai_programming.h"
#include "Tensor.h"

void max_pooling(const Tensor& input, Tensor& output, Tensor& output_mask, int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h, int pad_w);
void max_pooling_backward(const Tensor& out_grad, Tensor& out_mask, Tensor& in_grad);
#endif  // POOLING_H