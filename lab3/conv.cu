#include "conv.h"
/*
    此文件实现卷积层的前向传播和反向传播, 包括im2col和col2im的实现
    im2col: 将输入的维度为(N, Cin, H, W)的图像转换为维度为(N, H_out, W_out, Cin, kH, kW)的col矩阵
    前向传播: 将im2col的输出(N, H_out, W_out, Cin, kH, kW)与权重矩阵(Cout, Cin, kH, kW)相乘, 得到输出矩阵
    col2im: 将im2col的输出转换回原始输入
    反向传播得到输入矩阵梯度: 将输出的梯度(N, Cout, Hout, Wout)转变为(Cout,N, Hout, Wout)，与权重矩阵(Cout, Cin, kH, kW)相乘，
                            得到col_grad矩阵(N, H_out, W_out, Cin, kH, kW), 再将col_grad矩阵转换回原始输入(N, Cin, H, W)(此时需要用atomicadd解决多线程的访问统一内存地址的竞争问题)
    反向传播得到权重矩阵梯度: 将输出的梯度(N, Cout, Hout, Wout)转变为(Cout, N, Hout, Wout), 与col矩阵(N, H_out, W_out, Cin, kH, kW)相乘，得到权重矩阵梯度
*/
__global__ void im2col_kernel(const float* input, float* col, int N, int Cin, int H, int W, int kH, int kW, int pad, int stride) {
    int H_out = (H + 2 * pad - kH) / stride + 1;
    int W_out = (W + 2 * pad - kW) / stride + 1;
    int total_elements = N * H_out * W_out;
    
    CUDA_KERNEL_LOOP(index, total_elements) {
        int w_out = index % W_out;
        int h_out = (index / W_out) % H_out;
        int n = index / (H_out * W_out);

        int h_in = h_out * stride - pad;
        int w_in = w_out * stride - pad;
        int col_start = (n * H_out * W_out + h_out * W_out + w_out) * Cin * kH * kW;

        for (int c = 0; c < Cin; c++) {
            for (int i = 0; i < kH; i++) {
                for (int j = 0; j < kW; j++) {
                    int h = h_in + i;
                    int w = w_in + j;
                    int col_idx = col_start + c * kH * kW + i * kW + j;
                    if (h >= 0 && h < H && w >= 0 && w < W) {
                        col[col_idx] = input[((n * Cin + c) * H + h) * W + w];
                    } else {
                        col[col_idx] = 0;
                    }
                }
            }
        }
    }
}


__global__ void col2im_kernel(const float* col, float* input_grad, int N, int Cin, int H, int W, int kH, int kW, int pad, int stride) {
    int H_out = (H + 2 * pad - kH) / stride + 1;
    int W_out = (W + 2 * pad - kW) / stride + 1;
    int total_elements = N * H_out * W_out;
    
    CUDA_KERNEL_LOOP(index, total_elements) {
        int w_out = index % W_out;
        int h_out = (index / W_out) % H_out;
        int n = index / (H_out * W_out);

        int h_in = h_out * stride - pad;
        int w_in = w_out * stride - pad;
        int col_start = (n * H_out * W_out + h_out * W_out + w_out) * Cin * kH * kW;

        for (int c = 0; c < Cin; c++) {
            for (int i = 0; i < kH; i++) {
                for (int j = 0; j < kW; j++) {
                    int h = h_in + i;
                    int w = w_in + j;
                    int col_idx = col_start + c * kH * kW + i * kW + j;
                    int input_idx = ((n * Cin + c) * H + h) * W + w;
                    if (h >= 0 && h < H && w >= 0 && w < W) {
                        atomicAdd(&input_grad[input_idx], col[col_idx]);
                    } 
                }
            }
        }
    }
}

__global__ void transpose_tensor(const float* input, float* output, int N, int Cout, int Hout, int Wout) {
    CUDA_KERNEL_LOOP(i, N * Cout * Hout * Wout) {
        // Compute indices in the output and input tensors
        int n = i / (Cout * Hout * Wout); // batch index
        int cout = (i % (Cout * Hout * Wout)) / (Hout * Wout); // output channel index
        int h = (i % (Hout * Wout)) / Wout; // height index
        int w = i % Wout; // width index

        // Calculate the corresponding input index
        int input_idx = cout * N * Hout * Wout + n * Hout * Wout + h * Wout + w;

        // Calculate the corresponding output index
        int output_idx = n * Cout * Hout * Wout + cout * Hout * Wout + h * Wout + w;

        // Perform the assignment
        output[output_idx] = input[input_idx];
    }
}


void forward_conv(const Tensor& input, const Tensor& weights, Tensor& output) {
    // Get dimentions of the input and weights
    int N = input.shape[0];
    int Cin = input.shape[1];
    int H = input.shape[2];
    int W = input.shape[3];
    int Cout = weights.shape[0];
    int kH = weights.shape[2];
    int kW = weights.shape[3];
    // kernel size 3*3, stride 1, so pad = 1
    int pad = 1;
    int stride = 1;
    // Calculate the output dimensions
    int H_out = (H + 2 * pad - kH) / stride + 1;
    int W_out = (W + 2 * pad - kW) / stride + 1;
    
    // Check if the dimensions are correct
    if (output.shape[0] != N || output.shape[1] != Cout || output.shape[2] != H_out || output.shape[3] != W_out) {
        throw std::invalid_argument("Output size does not match the expected output size.");
    }

    // Allocate memory for im2col matrix
    Tensor col({N, Cin, H_out, W_out, kH, kW}, "GPU");
    Tensor output_grad_trans({Cout, N, H_out, W_out}, "GPU");

    // Launch im2col kernel
    int num_blocks = CudaGetBlocks(N * Cin * H_out * W_out);
    im2col_kernel<<<num_blocks, kCudaThreadsNum>>>(input.data, col.data, N, Cin, H, W, kH, kW, pad, stride);

    // Perform matrix multiplication
    int M = Cout;
    int K = Cin * kH * kW;
    int N_out = H_out * W_out * N;
    gemm_gpu(CUBLAS_OP_T, CUBLAS_OP_N, N_out, M, K, 1.0f, col.data, K, weights.data, K, 0.0f, output_grad_trans.data, N_out);

    // Transpose the output tensor. From (Cout, N, H_out, W_out) to (N, Cout, H_out, W_out)
    num_blocks = CudaGetBlocks(N * Cin * H_out * W_out);
    transpose_tensor<<<num_blocks, kCudaThreadsNum>>>(output_grad_trans.data, output.data, N, Cout, H_out, W_out);

}

void backward_conv(const Tensor& input, const Tensor& weights, const Tensor& grad_output,
                 Tensor& grad_input, Tensor& grad_weights){
    // Get dimentions of the input and weights
    int N = input.shape[0];
    int Cin = input.shape[1];
    int H = input.shape[2];
    int W = input.shape[3];
    int Cout = weights.shape[0];
    int kH = weights.shape[2];
    int kW = weights.shape[3];
    // kernel size 3*3, stride 1, so pad = 1
    int pad = 1;
    int stride = 1;
    // Calculate the output dimensions
    int H_out = (H + 2 * pad - kH) / stride + 1;
    int W_out = (W + 2 * pad - kW) / stride + 1;

    // Check if the dimensions are correct
    if (weights.shape[0] != Cout || weights.shape[1] != Cin || weights.shape[2] != kH || weights.shape[3] != kW) {
        throw std::invalid_argument("Weights size does not match the expected weights size.");
    }
    if (grad_output.shape[0] != N || grad_output.shape[1] != Cout || grad_output.shape[2] != H_out || grad_output.shape[3] != W_out) {
        throw std::invalid_argument("Output size does not match the expected output size.");
    }
    if (grad_output.shape[0] != N || grad_output.shape[1] != Cout || grad_output.shape[2] != H_out || grad_output.shape[3] != W_out) {
        throw std::invalid_argument("Output gradient size does not match the expected output size.");
    }
    if (grad_input.shape[0] != N || grad_input.shape[1] != Cin || grad_input.shape[2] != H || grad_input.shape[3] != W) {
        throw std::invalid_argument("Input gradient size does not match the expected input size.");
    }
    if (grad_weights.shape[0] != Cout || grad_weights.shape[1] != Cin || grad_weights.shape[2] != kH || grad_weights.shape[3] != kW) {
        throw std::invalid_argument("Weights gradient size does not match the expected weights size.");
    }

    // Allocate memory for intermediate tensors
    Tensor col({N, Cin, H_out, W_out, kH, kW}, "GPU");
    Tensor col_grad({N, Cin, H, W, kH, kW}, "GPU");
    Tensor output_grad_trans({Cout, N, H_out, W_out}, "GPU");

    // Launch im2col kernel
    int num_blocks = CudaGetBlocks(N * Cin * H_out * W_out);
    im2col_kernel<<<num_blocks, kCudaThreadsNum>>>(input.data, col.data, N, Cin, H, W, kH, kW, pad, stride);

    // Transpose the grad_output tensor. From (N, Cout, H_out, W_out) to (Cout, N, H_out, W_out) 
    num_blocks = CudaGetBlocks(N * Cout * H_out * W_out);
    transpose_tensor<<<num_blocks, kCudaThreadsNum>>>(grad_output.data, output_grad_trans.data, Cout, N, H_out, W_out);

    // Perform matrix multiplication for grad_weights
    int M = Cout;
    int K = N * H_out * W_out;
    int N_out = Cin * kH * kW;
    gemm_gpu(CUBLAS_OP_N, CUBLAS_OP_N, N_out, M, K, 1.0f, col.data, N_out, output_grad_trans.data, K, 0.0f, grad_weights.data, N_out);

    // Perform matrix multiplication for grad_input
    M = Cin * kH * kW;
    K = Cout;
    N_out = N * H_out * W_out;
    gemm_gpu(CUBLAS_OP_N, CUBLAS_OP_T, M, N_out, K, 1.0f, weights.data, M, output_grad_trans.data, N_out, 0.0f, col_grad.data, M);
    
    // Launch col2im kernel
    num_blocks = CudaGetBlocks(N * Cin * H_out * W_out);
    col2im_kernel<<<num_blocks, kCudaThreadsNum>>>(col_grad.data, grad_input.data, N, Cin, H, W, kH, kW, pad, stride);

}