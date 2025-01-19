#ifndef CELOSS
#define CELOSS

#include "Tensor.h"
#include "ai_programming.h"
#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>

float cross_entropy_loss(const Tensor& p, const Tensor& labels) ;
void cross_entropy_loss_grad(const Tensor& p, const Tensor& labels, Tensor& grad);

#endif // CELOSS
