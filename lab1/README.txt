# AI Programming Project
在本项目中我实现了作业文档中要求的功能，包括Tensor类、relu和sigmoid函数的正向反向传播、测试代码。

运行此项目：
nvcc -o test Tensor.cu
./test

详细介绍：
ai_programming.h 包含ppt中建议的宏和内联函数
Tensor.cu 为本项目的主体部分：
Tensor类 1、手动实现复制构造函数、复制赋值构造函数、析构函数
         2、实现成员函数cpu(),gpu()，符合作业文档要求
relu和sigmoid激活函数的正向传播和反向传播都分别在cpu和gpu上实现
在main函数中，我设计了一些测试代码，包括cpu(),gpu()转换、relu和sigmoid在cpu和gpu上的计算、relu和sigmoid反向传播在cpu和gpu上的计算。

以下为测试结果：
GPU_cudaMalloc: no error
Show tensor_cpu.                    #初始化tensor
shape: {2, 3}
device: CPU
-1.000000 2.000000 3.000000
4.000000 5.000000 6.000000
Show tensor_gpu.
shape: {2, 3}
device: GPU
-1.000000 2.000000 3.000000
4.000000 5.000000 6.000000
GPU_cudaMalloc: no error
Show pre_grad_cpu.                  #初始化反向传播到激活函数的grad
shape: {2, 3}
device: CPU
1.000000 1.000000 1.000000
1.000000 1.000000 1.000000
Show pre_grad_gpu.
shape: {2, 3}
device: GPU
1.000000 1.000000 1.000000
1.000000 1.000000 1.000000
------------------------------------
Data coping to GPU.                 #测试cpu()和gpu()
cudaMalloc: no error
cudaMemcpy: no error
Data copied to GPU.

Data coping to CPU.
cudaMemcpy: no error
cudaFree: no error
Data copied to CPU.

Copy tensor_cpu to GPU and back to CPU.
Test cpu() and gpu(): PASS

------------------------------------
Test on CPU.                        #在cpu上测试relu和sigmoid的正向和反向传播

ReLU output:
shape: {2, 3}
device: CPU
0.000000 2.000000 3.000000
4.000000 5.000000 6.000000

Sigmoid output:
shape: {2, 3}
device: CPU
0.268941 0.880797 0.952574
0.982014 0.993307 0.997527

ReLU grad output:
shape: {2, 3}
device: CPU
0.000000 1.000000 1.000000
1.000000 1.000000 1.000000

Sigmoid grad output:
shape: {2, 3}
device: CPU
0.196612 0.104994 0.045177
0.017663 0.006648 0.002466

Test on CPU finished.
------------------------------------
Test on GPU.                        #在gpu上测试relu和sigmoid的正向和反向传播

GPU_cudaMalloc: no error
relu_gpu: no error
ReLU output on GPU:
shape: {2, 3}
device: GPU
0.000000 2.000000 3.000000
4.000000 5.000000 6.000000

GPU_cudaMalloc: no error
Sigmoid output on GPU:
shape: {2, 3}
device: GPU
0.268941 0.880797 0.952574
0.982014 0.993307 0.997527

GPU_cudaMalloc: no error
ReLU grad output on GPU:
shape: {2, 3}
device: GPU
0.000000 1.000000 1.000000
1.000000 1.000000 1.000000

GPU_cudaMalloc: no error
Sigmoid grad output on GPU:
shape: {2, 3}
device: GPU
0.196612 0.104994 0.045177
0.017663 0.006648 0.002466

Test on GPU finished.
------------------------------------
Comparing CPU and GPU results we find that they get the same result.
Test finished.