#include "fully_connected.h"
/*
    此文件实现全连接层的前向与反向传播，同时也实现了随机初始化矩阵参数的matrix_init函数
    前向传播: 将输入矩阵(N, Cin)与权重矩阵(Cin, Cout)相乘，再加上偏置矩阵(Cout),得到输出矩阵(N, Cout)
    反向传播: 传入输出的梯度(N, Cout)，按照链式法则得到的公式，计算输入梯度(N, Cin)，权重梯度(Cin, Cout)，偏置梯度(Cout)
*/

// Fill the matrix with random numbers on GPU
void matrix_init(float* A, const std::vector<int> &shape) {

    int totalSize = 1;
    for (int dim : shape) totalSize *= dim;

    // Create a pseudo random number generator
    curandGenerator_t prng;
    curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
    
    curandSetPseudoRandomGeneratorSeed(prng, 42);
    
    // Fill the array with random numbers on the device
    curandGenerateUniform(prng, A, totalSize);
    
    // Clean up the generator
    curandDestroyGenerator(prng);
}

// GEMM wrapper function, handle the cublasHandle
void gemm_gpu(cublasOperation_t transA, cublasOperation_t transB,
              int m, int n, int k,
              float a, const float* A, int lda,
              const float* B, int ldb,
              float b, float* C, int ldc) {

    const float alf = a , bet = b;
    const float *alpha = &alf;
    const float *beta = &bet;
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    cublasSgemm(handle, transA, transB, m, n, k,
                alpha, A, lda, B, ldb,
                beta, C, ldc);
    
    cublasDestroy(handle);
}

// Forward pass function, the output should be a Tensor
void forward_fc(const Tensor& input, const Tensor& weights, const Tensor& bias, Tensor& output) {
    
    // Get the dimensions of the input, weights, and bias
    const int batch_size = input.shape[0];
    int in_channels = input.shape[1];
    int out_channels = weights.shape[1];
    
    // Check if the dimensions are correct
    if (input.shape[1] != weights.shape[0]) {
        throw std::invalid_argument("Input channels do not match the number of rows in weights.");
    }
    if (bias.shape[1] != weights.shape[1]) {
        throw std::invalid_argument("Bias size does not match the number of columns in weights.");
    }
    if (output.shape[0] != batch_size || output.shape[1] != out_channels) {
        throw std::invalid_argument("Output size does not match the expected output size.");
    }
    
    // Perform the matrix multiplication
    gemm_gpu(CUBLAS_OP_N, CUBLAS_OP_N, out_channels, batch_size, in_channels,
             1.0f, weights.data, out_channels, input.data, in_channels, 
             0.0f, output.data, out_channels);
    
    // Allocate and fill the device memory for bias addition
    float* ones = new float[batch_size];
    for (int i = 0; i < batch_size; ++i) {
        ones[i] = 1.0;
    }
    float *ones_;
    cudaMalloc(&ones_, batch_size * sizeof(float));
    cudaMemcpy(ones_, ones, batch_size * sizeof(float), cudaMemcpyHostToDevice);
    
    // Add the bias to the output
    gemm_gpu(CUBLAS_OP_N, CUBLAS_OP_N, out_channels, batch_size, 1,
             1.0f, bias.data, out_channels, ones_, 1, 
             1.0f, output.data, out_channels);

    cudaFree(ones_);
    delete [] ones;
}

// Backward pass function
void backward_fc(const Tensor& input, const Tensor& weights, const Tensor& bias,
                 const Tensor& output_grad, Tensor& input_grad,
                 Tensor& weights_grad, Tensor& bias_grad) {
    
    // Get the dimensions of the input, weights, and bias
    const int batch_size = input.shape[0];
    int in_channels = input.shape[1];
    int out_channels = weights.shape[1];

    // Check if the dimensions are correct
    if (input.shape[1] != weights.shape[0]) {
        throw std::invalid_argument("Input channels do not match the number of rows in weights.");
    }
    if (bias.shape[1] != weights.shape[1]) {
        throw std::invalid_argument("Bias size does not match the number of columns in weights.");
    }
    if (output_grad.shape[0] != batch_size || output_grad.shape[1] != out_channels) {
        throw std::invalid_argument("Output gradient size does not match the expected output size.");
    }
    if (input_grad.shape[0] != batch_size || input_grad.shape[1] != in_channels) {
        throw std::invalid_argument("Input gradient size does not match the expected input size.");
    }
    if (weights_grad.shape[0] != in_channels || weights_grad.shape[1] != out_channels) {
        throw std::invalid_argument("Weights gradient size does not match the expected weights size.");
    }
    if (bias_grad.shape[1] != out_channels) {
        throw std::invalid_argument("Bias gradient size does not match the expected bias size.");
    }

    // Calculate weights gradient: input^T * output_grad
    gemm_gpu(CUBLAS_OP_N, CUBLAS_OP_T, out_channels, in_channels, batch_size,
             1.0f, output_grad.data, out_channels, input.data, in_channels, 
             0.0f, weights_grad.data, out_channels);
    
    // Calculate input gradient: output_grad * weights^T
    gemm_gpu(CUBLAS_OP_T, CUBLAS_OP_N, in_channels, batch_size, out_channels,
             1.0f, weights.data, out_channels, output_grad.data, out_channels,
             0.0f, input_grad.data, in_channels);

    // Calculate bias gradient: sum(output_grad) across batch
    float* ones = new float[batch_size];
    for (int i = 0; i < batch_size; ++i) {
        ones[i] = 1.0f;
    }
    float *ones_;
    cudaMalloc(&ones_, batch_size * sizeof(float));
    cudaMemcpy(ones_, ones, batch_size * sizeof(float), cudaMemcpyHostToDevice);
    
    // Sum gradients for the bias
    gemm_gpu(CUBLAS_OP_N, CUBLAS_OP_T, 1, out_channels, batch_size,
             1.0f, ones_, 1, output_grad.data, out_channels,
             0.0f, bias_grad.data, 1);

    cudaFree(ones_); // Free device memory
    delete [] ones;
}
