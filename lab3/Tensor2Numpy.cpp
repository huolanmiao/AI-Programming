#include <iostream>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // To support std::vector
#include <pybind11/numpy.h>

namespace py = pybind11;

py::array_t<float> Tensor2Numpy(const Tensor & t) {
    std::vector<int> shape = t.shape;
    size_t itemsize = sizeof(float);
    std::vector<py::ssize_t> strides(shape.size());
    strides[shape.size()-1] = sizeof(float);
    // 计算步长，从最后一个维度开始
    for (int i = shape.size() - 2; i >= 0  ; i--) {
        strides[i] = strides[i+1] * shape[i+1];
    }
    py::buffer_info buf(t.data, sizeof(float), py::format_descriptor<float>::format(), shape.size(), shape, strides);
    
    // 使用 buffer_info 创建一个 NumPy 数组
    return py::array_t<float>(buf);
}