#include "activations.h"

// CUDA kernel for ReLU activation
__global__ void relu_gpu(float * in, float * out, int n) {
    CUDA_KERNEL_LOOP(i , n){
        out[i] = in[i] > 0 ? in[i] : 0;
    }
}

// CUDA kernel for Sigmoid activation
__global__ void sigmoid_gpu(float* in, float* out, int n) {
    CUDA_KERNEL_LOOP(i , n){
        out[i] = 1 / (1 + exp(-in[i]));
    }
}

// CPU implementation of ReLU
void relu_cpu(float* in, float* out, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = in[i] > 0 ? in[i] : 0;
    }
}

// CPU implementation of Sigmoid
void sigmoid_cpu(float* in, float* out, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = 1 / (1 + exp(-in[i]));
    }
}

// ReLU forward function
Tensor relu(const Tensor& input) {
    Tensor output(input.shape, input.device);
    int N = 1;
    for(int dim : input.shape) N *= dim;
    if (input.device == "GPU") {
        relu_gpu<<<CudaGetBlocks(N), kCudaThreadsNum>>>(input.data, output.data, N);
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        printf("relu_gpu: %s\n", cudaGetErrorString(err));
    } else {
        relu_cpu(input.data, output.data, N);
    }
    return output;
}

// ReLU gradient calculation
__global__ void relu_grad_gpu(float* grad_data, const float* result_data, const float* pre_grad_data, int n) {
    CUDA_KERNEL_LOOP(i, n) {
        grad_data[i] = (result_data[i] > 0 ? 1 : 0) * pre_grad_data[i];
    }
}

// Sigmoid gradient calculation
__global__ void sigmoid_grad_gpu(float* grad_data, const float* result_data, const float* pre_grad_data, int n) {
    CUDA_KERNEL_LOOP(i, n) {
        grad_data[i] = result_data[i] * (1 - result_data[i]) * pre_grad_data[i];
    }
}

// CPU implementation of ReLU gradient
void relu_grad_cpu(float* grad_data, const float* result_data, const float* pre_grad_data, int n) {
    for (int i = 0; i < n; i++) {
        grad_data[i] = (result_data[i] > 0 ? 1 : 0) * pre_grad_data[i];
    }
}

// CPU implementation of Sigmoid gradient
void sigmoid_grad_cpu(float* grad_data, const float* result_data, const float* pre_grad_data, int n) {
    for (int i = 0; i < n; i++) {
        grad_data[i] = result_data[i] * (1 - result_data[i]) * pre_grad_data[i];
    }
}

// ReLU gradient function
Tensor relu_grad(const Tensor& pre_grad, const Tensor& output) {
    Tensor grad(output.shape, output.device);
    int N = 1;
    for(int dim : output.shape) N *= dim;
    if (grad.device == "GPU") {
        relu_grad_gpu<<<CudaGetBlocks(N), kCudaThreadsNum>>>(grad.data, output.data, pre_grad.data, N);
    } else {
        relu_grad_cpu(grad.data, output.data, pre_grad.data, N);
    }
    return grad;
}

// Sigmoid gradient function
Tensor sigmoid_grad(const Tensor& pre_grad, const Tensor& output) {
    Tensor grad(output.shape, output.device);
    int N = 1;
    for(int dim : output.shape) N *= dim;
    if (grad.device == "GPU") {
        sigmoid_grad_gpu<<<CudaGetBlocks(N), kCudaThreadsNum>>>(grad.data, output.data, pre_grad.data, N);
    } else {
        sigmoid_grad_cpu(grad.data, output.data, pre_grad.data, N);
    }
    return grad;
}


// Sigmoid forward function
Tensor sigmoid(const Tensor& input) {
    Tensor output(input.shape, input.device);
    int N = 1;
    for(int dim : input.shape) N *= dim;
    if (input.device == "GPU") {
        sigmoid_gpu<<<CudaGetBlocks(N), kCudaThreadsNum>>>(input.data, output.data, N);
    } else {
        sigmoid_cpu(input.data, output.data, N);
    }
    return output;
}

__global__ void softmax_forward(const float* input, float* output, int N, int C) {
    CUDA_KERNEL_LOOP(idx, N) {
        float max_val = -INFINITY;
        // 计算该样本的最大值
        for (int i = 0; i < C; ++i) {
            max_val = fmaxf(max_val, input[idx * C + i]);
        }

        float sum = 0.0f;
        
        // 计算每个类别的指数和（减去最大值以保持数值稳定性）
        for (int i = 0; i < C; ++i) {
            sum += expf(input[idx * C + i] - max_val);
        }
        
        // 归一化每个类别的概率
        for (int i = 0; i < C; ++i) {
            output[idx * C + i] = expf(input[idx * C + i] - max_val) / sum;
        }
    }
}

struct exponential_functor {
    __host__ __device__
    float operator()(const float& x) const {
        return expf(x);  // 使用expf，确保在CUDA设备上兼容
    }
};

void softmax(const Tensor& input, Tensor& output) {
    int N = input.shape[0];
    int C = input.shape[1];

    // Check if the dimensions are correct
    if (output.shape[0] != N || output.shape[1] != C) {
        throw std::invalid_argument("The dimensions of the output tensor are incorrect.");
    }

    // Use thrust to perform the softmax operation
    thrust::device_vector<float> d_in(input.data, input.data + N*C);
    float max_value = thrust::reduce(d_in.begin(), d_in.end(), INT_MIN, thrust::maximum<float>());
    thrust::transform(d_in.begin(), d_in.end(), d_in.begin(), _1 - max_value);
    thrust::transform(d_in.begin(), d_in.end(), d_in.begin(), exponential_functor());
    for (int i = 0; i < N; i++) {
        float sum = thrust::reduce(d_in.begin() + i * C, d_in.begin() + (i + 1) * C, 0.0f, thrust::plus<float>());
        thrust::transform(d_in.begin() + i * C, d_in.begin() + (i + 1) * C, output.data+ i * C, _1 / sum);
    }
    
    // Call the softmax forward kernel
    // softmax_forward<<<CudaGetBlocks(N), kCudaThreadsNum>>>(input.data, output.data, N, C);
}
