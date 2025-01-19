#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include <vector>
#include <cuda_runtime.h>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include "ai_programming.h"
#include "Tensor.h"
using namespace thrust::placeholders; // 使用占位符 _1

void softmax(const Tensor& input, Tensor& output);
Tensor relu(const Tensor& input);
Tensor relu_grad(const Tensor& pre_grad, const Tensor& output);
Tensor sigmoid(const Tensor& input);
Tensor sigmoid_grad(const Tensor& pre_grad, const Tensor& output);

#endif // ACTIVATIONS_H
