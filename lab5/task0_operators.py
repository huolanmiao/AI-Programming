"""
本文件我们给出一个基本完善的Tensor类
你可以将lab5的对应代码复制到这里
"""

import numpy as np
from typing import List, Optional, Tuple, Union
from device import cpu, Device
from basic_operator import Op, Value
from task0_autodiff import compute_gradient_of_variables

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
        out_grad = out_grad if out_grad else ones(*self.shape, dtype=self.dtype, device=self.device)
                
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


