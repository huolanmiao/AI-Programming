#ifndef CONV_H
#define CONV_H

#include <cublas_v2.h>
#include <curand.h>
#include <iostream>
#include <cuda_runtime.h>
#include <curand.h>
#include <cuda_runtime.h>
#include "Tensor.h"
#include "ai_programming.h"
#include "fully_connected.h"

void forward_conv(const Tensor& input, const Tensor& weights, Tensor& output, int pad=1, int stride=1);

void backward_conv(const Tensor& input, const Tensor& weights, const Tensor& grad_output,
                 Tensor& grad_input, Tensor& grad_weights, int pad=1, int stride=1);



#endif  // CONVOLUTION_H
