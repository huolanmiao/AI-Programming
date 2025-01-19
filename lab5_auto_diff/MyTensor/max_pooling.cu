#include "max_pooling.h"
// 此文件实现max pooling的前向传播

__global__ void max_pool_forward(
    float* in_data, int nthreads, int num, int channels, int in_h, int in_w, int out_h, int out_w, 
    int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h, int pad_w, float* out_data, float* out_mask) {
    
    CUDA_KERNEL_LOOP(index, nthreads) {
        int n = index / (out_h * out_w * channels); // Batch index
        int c = (index / (out_h * out_w)) % channels; // Channel index
        int ph = (index / out_w) % out_h; // Output height index
        int pw = index % out_w; // Output width index

        // Define the starting point for the pooling window
        int h_start = ph * stride_h - pad_h;
        int w_start = pw * stride_w - pad_w;
        int h_end = min(h_start + kernel_h, in_h);
        int w_end = min(w_start + kernel_w, in_w);
        h_start = max(h_start, 0);
        w_start = max(w_start, 0);

        // Apply max pooling and store the max value and its position
        float max_val = -FLT_MAX;
        int max_idx = -1;

        for (int h = h_start; h < h_end; ++h) {
            for (int w = w_start; w < w_end; ++w) {
                int input_idx = n * channels * in_h * in_w + c * in_h * in_w + h * in_w + w;
                if (in_data[input_idx] > max_val) {
                    max_val = in_data[input_idx];
                    max_idx = h * in_w + w;
                }
            }
        }

        // Store the max value and the index of the max value (mask)
        int output_idx = n * channels * out_h * out_w + c * out_h * out_w + ph * out_w + pw;
        out_data[output_idx] = max_val;
        out_mask[output_idx] = max_idx;
    }
}


void max_pooling(const Tensor& input, Tensor& output, Tensor& output_mask, int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h, int pad_w) {
    // Get the dimensions of the input tensor
    int num = input.shape[0];
    int channels = input.shape[1];
    int in_h = input.shape[2];
    int in_w = input.shape[3];

    // Calculate the dimensions of the output tensor
    int out_h = (in_h + 2 * pad_h - kernel_h) / stride_h + 1;
    int out_w = (in_w + 2 * pad_w - kernel_w) / stride_w + 1;

    // Check if the dimensions are correct
    if (output.shape[0] != num || output.shape[1] != channels || output.shape[2] != out_h || output.shape[3] != out_w) {
        throw std::invalid_argument("The dimensions of the output tensor are incorrect.");
    }

    // Calculate the number of threads
    int nthreads = num * channels * out_h * out_w;

    // Call the CUDA kernel
    int num_blocks = CudaGetBlocks(nthreads);
    max_pool_forward<<<num_blocks, kCudaThreadsNum>>>(
        input.data, nthreads, num, channels, in_h, in_w, out_h, out_w, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, output.data, output_mask.data);
}

