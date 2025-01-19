"""
利用作业三中编译好的卷积函数，实现Conv2D和MaxPooling算子，以适配作业四中的自动微分框架，使得模型在训练过程中能够调用我用cuda编写、pybind编译的卷积层实现。测试发现，我写的for-loop版本池化层反向传播耗时较多，所以我实现了CUDA并行的池化层反向传播，实验发现能显著提高运行速度。
"""

import numpy as np
from typing import List, Optional, Tuple, Union
from device import cpu, Device
from basic_operator import Op, Value
from task0_autodiff import compute_gradient_of_variables
import time

import MyTensor.tensor as MyCuda

class Tensor(Value):
    grad: "Tensor"

    def __init__(
        self,
        array,
        *,
        device: Optional[Device] = None,
        dtype=None,
        requires_grad=True,
        **kwargs
    ):
        if isinstance(array, Tensor):
            if device is None:
                device = array.device
            if dtype is None:
                dtype = array.dtype
            if device == array.device and dtype == array.dtype:
                cached_data = array.realize_cached_data()
            else:
                cached_data = Tensor._array_from_numpy(
                    array.numpy(), device=device, dtype=dtype
                )
        else:
            device = device if device else cpu()
            cached_data = Tensor._array_from_numpy(array, device=device, dtype=dtype)

        self._init(
            None,
            [],
            cached_data=cached_data,
            requires_grad=requires_grad,
        )

    @staticmethod
    def _array_from_numpy(numpy_array, device, dtype):
        return np.array(numpy_array, dtype=dtype)

    @staticmethod
    def make_from_op(op: Op, inputs: List["Value"]):  # 算子返回的还是Tensor，所以先前out.backward()报错
        tensor = Tensor.__new__(Tensor)
        tensor._init(op, inputs)
        if not tensor.requires_grad:
            return tensor.detach()
        tensor.realize_cached_data()
        return tensor

    @staticmethod
    def make_const(data, requires_grad=False):
        tensor = Tensor.__new__(Tensor)
        tensor._init(
            None,
            [],
            cached_data=data
            if not isinstance(data, Tensor)
            else data.realize_cached_data(),
            requires_grad=requires_grad,
        )
        return tensor

    @property
    def data(self):
        return self.detach()

    @data.setter
    def data(self, value):
        assert isinstance(value, Tensor)
        assert value.dtype == self.dtype, "%s %s" % (
            value.dtype,
            self.dtype,
        )
        self.cached_data = value.realize_cached_data()

    def detach(self):
        return Tensor.make_const(self.realize_cached_data())

    @property
    def shape(self):
        return self.realize_cached_data().shape

    @property
    def dtype(self):
        return self.realize_cached_data().dtype

    @property
    def device(self):
        return cpu()


    def backward(self, out_grad=None):
        def constant(*shape, c=1.0, device=None, dtype="float32", requires_grad=False):
            """Generate constant Tensor"""
            device = cpu() if device is None else device
            array = device.ones(*shape, dtype=dtype) * c  # note: can change dtype
            return Tensor(array, device=device, dtype=dtype, requires_grad=requires_grad)


        def ones(*shape, device=None, dtype="float32", requires_grad=False):
            """Generate all-ones Tensor"""
            return constant(
                *shape, c=1.0, device=device, dtype=dtype, requires_grad=requires_grad
            )
        out_grad = out_grad if out_grad is not None else ones(*self.shape, dtype=self.dtype, device=self.device)
                
        compute_gradient_of_variables(self, out_grad)
        

    def __repr__(self):
        return "Tensor(" + str(self.realize_cached_data()) + ")"

    def __str__(self):
        return self.realize_cached_data().__str__()

    def numpy(self):
        data = self.realize_cached_data()

        return data


    def __add__(self, other):
        if isinstance(other, Tensor):
            return EWiseAdd()(self, other)
        else:
            return AddScalar(other)(self)

    def __mul__(self, other):
        if isinstance(other, Tensor):
            return EWiseMul()(self, other)
        else:
            return MulScalar(other)(self)

    def __pow__(self, other):
        if isinstance(other, Tensor):
            return EWisePow()(self, other)
        else:
            return PowerScalar(other)(self)

    def __sub__(self, other):
        if isinstance(other, Tensor):
            return EWiseAdd()(self, Negate()(other))
        else:
            return AddScalar(-other)(self)

    def __truediv__(self, other):
        if isinstance(other, Tensor):
            return EWiseDiv()(self, other)
        else:
            return DivScalar(other)(self)

    def __matmul__(self, other):
        return MatMul()(self, other)

    def matmul(self, other):
        return MatMul()(self, other)

    def sum(self, axes=None):
        return Summation(axes)(self)

    def broadcast_to(self, shape):
        return BroadcastTo(shape)(self)

    def reshape(self, shape):
        return Reshape(shape)(self)

    def __neg__(self):
        return Negate()(self)

    def transpose(self, axes=None):
        return Transpose(axes)(self)

    __radd__ = __add__
    __rmul__ = __mul__
    __rsub__ = __sub__
    __rmatmul__ = __matmul__

class TensorOp(Op):
    def __call__(self, *args):
        return Tensor.make_from_op(self, args)


class EWiseAdd(TensorOp):
    def compute(self, a: np.ndarray, b: np.ndarray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: np.ndarray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: np.ndarray, b: np.ndarray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: np.ndarray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """逐点乘方，用标量做指数"""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: np.ndarray) -> np.ndarray:
        ## 请于此填写你的代码
        return np.power(a, self.scalar)
        

    def gradient(self, out_grad, node):
        ## 请于此填写你的代码
        a = node.inputs[0]
        grad_a = out_grad * self.scalar * a ** (self.scalar - 1)
        return grad_a
        


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWisePow(TensorOp):
    """逐点乘方"""

    def compute(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a**b

    def gradient(self, out_grad, node):
        if not isinstance(node.inputs[0], Tensor) or not isinstance(
            node.inputs[1], Tensor
        ):
            raise ValueError("Both inputs must be tensors.")

        a, b = node.inputs[0], node.inputs[1]
        grad_a = out_grad * b * (a ** (b - 1))
        grad_b = out_grad * (a**b) * log(a)
        return grad_a, grad_b

def power(a, b):
    return EWisePow()(a, b)


class EWiseDiv(TensorOp):
    """逐点相除"""

    def compute(self, a, b):
        ## 请于此填写你的代码
        return a / b
        

    def gradient(self, out_grad, node):
        ## 请于此填写你的代码
        a, b = node.inputs
        grad_a = out_grad / b
        grad_b = -out_grad * a / (b ** 2)
        return grad_a, grad_b
        


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ## 请于此填写你的代码
        return a / self.scalar
        

    def gradient(self, out_grad, node):
        ## 请于此填写你的代码
        return out_grad / self.scalar


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ## 请于此填写你的代码
        if self.axes is None:
            self.axes = tuple(reversed(np.arange(len(a.shape))[-2:]))
        old_axes = np.arange(len(a.shape))
        idx1 = self.axes[0]
        idx2 = self.axes[1]
        old_axes[idx1], old_axes[idx2] = old_axes[idx2], old_axes[idx1]
        self.axes = tuple(old_axes)
        return np.transpose(a, axes=self.axes)
        

    def gradient(self, out_grad, node):
        ## 请于此填写你的代码
        return out_grad.transpose(axes=self.axes)
        


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ## 请于此填写你的代码
        return np.reshape(a, self.shape)
        

    def gradient(self, out_grad, node):
        ## 请于此填写你的代码
        return out_grad.reshape(node.inputs[0].shape)
        


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ## 请于此填写你的代码
        return np.broadcast_to(a, self.shape)
        

    def gradient(self, out_grad, node):
        ## 请于此填写你的代码
        # 将处于原先被广播的维度的grad求和
        broadcasted_dim = []
        a = node.inputs[0]
        for i in range(len(a.shape)):
            if a.shape[i] == 1:
                broadcasted_dim.append(i)
        return out_grad.sum(axes=tuple(broadcasted_dim)).reshape(a.shape)
        


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ## 请于此填写你的代码
        return np.sum(a, axis=self.axes)

        

    def gradient(self, out_grad, node):
        ## 请于此填写你的代码
        adapt_shape = node.inputs[0].shape
        if self.axes is None:
            adapt_shape = np.ones(len(adapt_shape), dtype=int)
        else:
            for i in self.axes:
                adapt_shape[i] = 1
            
        return out_grad.reshape(adapt_shape).broadcast_to(node.inputs[0].shape)
        


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ## 请于此填写你的代码
        return np.matmul(a, b)
        

    def gradient(self, out_grad, node):
        ## 请于此填写你的代码
        a, b = node.inputs
        grad_a = matmul(out_grad, b.transpose())
        grad_b = matmul(a.transpose(), out_grad)
        return grad_a, grad_b
        


def matmul(a, b):
    return MatMul()(a, b)

class Conv2D(TensorOp):
    def __init__(self, pad, stride):
        self.pad = pad
        self.stride = stride
        
    def compute(self, kernel, image):
        # 用作业三的Tensor包装之后，正向运算
        tensor_x = MyCuda.Tensor(image, "GPU") # [N, Cin, H, W]
        tensor_weights = MyCuda.Tensor(kernel, "GPU") # [Cout, Cin, kH, kW]
        H_out = (tensor_x.shape[2] + 2 * self.pad - tensor_weights.shape[2]) / self.stride + 1
        W_out = (tensor_x.shape[3] + 2 * self.pad - tensor_weights.shape[3]) / self.stride + 1
        tensor_output = MyCuda.Tensor([tensor_x.shape[0],tensor_weights.shape[0],int(H_out),int(W_out)], "GPU")
        MyCuda.forward_conv(tensor_x, tensor_weights, tensor_output, self.pad, self.stride)
        return tensor_output.to_numpy()
    
    def gradient(self, out_grad, node):
        kernel, image = node.inputs
        # 用作业三的Tensor包装之后，反向传播
        tensor_x = MyCuda.Tensor(image.numpy(), "GPU") # [N, Cin, H, W]
        tensor_weights = MyCuda.Tensor(kernel.numpy(), "GPU") # [Cout, Cin, kH, kW]
        tensor_out_grad = MyCuda.Tensor(out_grad.numpy(), "GPU") # [N, Cout, H, W]
        
        input_grad = MyCuda.Tensor([tensor_x.shape[0],tensor_x.shape[1],tensor_x.shape[2],tensor_x.shape[3]], "GPU")
        weights_grad = MyCuda.Tensor([tensor_weights.shape[0],tensor_weights.shape[1],tensor_weights.shape[2],tensor_weights.shape[3]], "GPU")
        MyCuda.backward_conv(tensor_x, tensor_weights, tensor_out_grad, input_grad, weights_grad, self.pad, self.stride)
        
        return Tensor(weights_grad.to_numpy()), Tensor(input_grad.to_numpy())

def conv2d(kernel, image, pad=1, stride=1):
    return Conv2D(pad, stride)(kernel, image)

class MaxPooling(TensorOp):
    def __init__(self, kernel_h=2, kernel_w=2, stride_h=2, stride_w=2, pad_h=0, pad_w=0):
        self.kernel_h = kernel_h
        self.kernel_w = kernel_w
        self.stride_h = stride_h
        self.stride_w = stride_w
        self.pad_h = pad_h
        self.pad_w = pad_w
        self.output_mask = None
    
    def compute(self, input):
        out_h = int((input.shape[2] + 2 * self.pad_h - self.kernel_h) / self.stride_h + 1)
        out_w = int((input.shape[3] + 2 * self.pad_w - self.kernel_w) / self.stride_w + 1)
        # [N, Cin, H, W]
        tensor_x = MyCuda.Tensor(input, "GPU")
        # [N, Cin, out_h, out_w]
        tensor_output = MyCuda.Tensor([tensor_x.shape[0], tensor_x.shape[1], out_h, out_w], "GPU")
        # [N, Cin, out_h, out_w]
        output_mask = MyCuda.Tensor([tensor_x.shape[0], tensor_x.shape[1], out_h, out_w], "GPU")
        MyCuda.max_pooling(tensor_x, tensor_output, output_mask, self.kernel_h, self.kernel_w, self.stride_h, self.stride_w, self.pad_h, self.pad_w)
        
        self.output_mask = output_mask.to_numpy()
        return tensor_output.to_numpy()
    
    def gradient(self, out_grad, node):
        # st = time.perf_counter()
        # for-loop实现
        # N = node.inputs[0].shape[0]
        # Cin = node.inputs[0].shape[1]
        # in_grad = np.zeros(node.inputs[0].shape)
        # out_grad = out_grad.numpy()
        # for n in range(N):
        #     for c in range(Cin):
        #         for i in range(out_grad.shape[2]):
        #             for j in range(out_grad.shape[3]):
        #                 idx = self.output_mask[n, c, i, j]
        #                 in_grad[n, c, int(idx // in_grad.shape[3]), int(idx % in_grad.shape[3])] = out_grad[n, c, i, j]
        
        # CUDA实现
        # [N, Cin, out_h, out_w]
        tensor_out_grad = MyCuda.Tensor(out_grad.numpy(), "GPU")
        # [N, Cin, out_h, out_w]
        tensor_output_mask = MyCuda.Tensor(self.output_mask, "GPU")
        # [N, Cin, H, W]
        tensor_in_grad = MyCuda.Tensor(node.inputs[0].shape, "GPU")
        MyCuda.max_pooling_backward(tensor_out_grad, tensor_output_mask, tensor_in_grad)
        # print(f"back pool: {time.perf_counter() - st}s")
        return Tensor(tensor_in_grad.to_numpy())

def max_pooling(input, kernel_h=2, kernel_w=2, stride_h=2, stride_w=2, pad_h=0, pad_w=0):
    return MaxPooling(kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w)(input)

class Negate(TensorOp):
    def compute(self, a):
        ## 请于此填写你的代码
        return -a
        

    def gradient(self, out_grad, node):
        ## 请于此填写你的代码
        return -out_grad
        


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ## 请于此填写你的代码
        return np.log(a)
        

    def gradient(self, out_grad, node):
        ## 请于此填写你的代码
        a = node.inputs[0]
        return out_grad / a
        


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ## 请于此填写你的代码
        return np.exp(a)
        

    def gradient(self, out_grad, node):
        ## 请于此填写你的代码
        return out_grad * exp(node.inputs[0])
        


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ## 请于此填写你的代码
        return np.maximum(0, a)
        

    def gradient(self, out_grad, node):
        ## 请于此填写你的代码
        return out_grad * (node.inputs[0].realize_cached_data() > 0)
        


def relu(a):
    return ReLU()(a)


