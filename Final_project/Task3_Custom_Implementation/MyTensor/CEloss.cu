#include "CEloss.h"
// 此文件实现cross entropy loss的前向运算和cross entropy loss with softmax的反向传播

__global__ void cross_entropy_loss(const float* p, const float* labels, float* loss, int N, int C) {
    CUDA_KERNEL_LOOP(i, N) {
        int label = labels[i];
        loss[i] = -logf(p[i * C + label]);
    }
}

float cross_entropy_loss(const Tensor& p, const Tensor& labels) {
    int N = p.shape[0];
    int C = p.shape[1];

    // Check if the dimensions are correct
    if (labels.shape[1] != N) {
        throw std::invalid_argument("The dimensions of the labels tensor are incorrect.");
    }
    
    
    Tensor CE_loss({N,1}, "GPU"); 
    // Call the cross entropy loss kernel
    cross_entropy_loss<<<CudaGetBlocks(N), kCudaThreadsNum>>>(p.data, labels.data, CE_loss.data, N, C);

    thrust::device_vector<float> d_vec(CE_loss.data, CE_loss.data + N);
    
    // CE_loss.show_tensor();
    return thrust::reduce(d_vec.begin(), d_vec.end(), 0.0f, thrust::plus<float>()) / N;
}

__global__ void cross_entropy_loss_grad(const float* p, const float* labels, float* grad, int N, int C) {
    CUDA_KERNEL_LOOP(i, N) {
        int label = labels[i];
        for (int j = 0; j < C; ++j) {
            grad[i * C + j] = p[i * C + j];
        }
        grad[i * C + label] -= 1.0f;
    }
}

void cross_entropy_loss_grad(const Tensor& p, const Tensor& labels, Tensor& grad) {
    int N = p.shape[0];
    int C = p.shape[1];

    // Check if the dimensions are correct
    if (labels.shape[0] != N) {
        throw std::invalid_argument("The dimensions of the labels tensor are incorrect.");
    }
    if (grad.shape[0] != N || grad.shape[1] != C) {
        throw std::invalid_argument("The dimensions of the gradient tensor are incorrect.");
    }

    // Call the cross entropy loss gradient kernel
    cross_entropy_loss_grad<<<CudaGetBlocks(N), kCudaThreadsNum>>>(p.data, labels.data, grad.data, N, C);
}