#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // To support std::vector
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <iostream>
#include <cstddef>
#include "Tensor.h"
#include "activations.h"
#include "max_pooling.h"
#include "fully_connected.h"
#include "conv.h"
#include "CEloss.h"
#include <cublas_v2.h>
#include <curand.h>

namespace py = pybind11;



// Bind Tensor class to Python
PYBIND11_MODULE(tensor, m) {
    // m.def("print_array", &printArray, "Helper function for shoe_tensor.");
    py::class_<Tensor>(m, "Tensor")
        .def(py::init<const Tensor&>())
        .def(py::init<const std::vector<int> &, const std::string &>())
        .def(py::init<py::array_t<float>, const std::string &>())
        .def_readwrite("data_address", &Tensor::data)
        .def_readwrite("shape", &Tensor::shape)
        .def_readwrite("device", &Tensor::device)
        .def("set_data", &Tensor::set_data)
        .def("to_numpy", &Tensor::to_numpy)
        .def("show_tensor", &Tensor::show_tensor)
        // .def("get_strides", &Tensor::get_strides)
        .def("cpu", &Tensor::cpu)
        .def("gpu", &Tensor::gpu);
    // m.def("Tensor2Numpy", &Tensor2Numpy,"t2n");
    m.def("relu", &relu, "ReLU activation function");
    m.def("relu_grad", &relu_grad, "ReLU grad activation function");
    m.def("sigmoid_grad", &sigmoid_grad, "Sigmoid grad activation function");
    m.def("sigmoid", &sigmoid, "Sigmoid activation function");
    m.def("softmax", &softmax, "Softmax activation function" );

    m.def("cross_entropy_loss", &cross_entropy_loss, "cross_entropy_loss" );
    m.def("cross_entropy_loss_grad", &cross_entropy_loss_grad, "cross_entropy_loss_grad" );

    m.def("forward_conv", &forward_conv, "forward_conv" );
    m.def("backward_conv", &backward_conv, "backward_conv" );

    m.def("forward_fc", &forward_fc, "forward_fc" );
    m.def("backward_fc", &backward_fc, "backward_fc" );

    m.def("max_pooling", &max_pooling, "max_pooling" );
}
