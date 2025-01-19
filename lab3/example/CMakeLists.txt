cmake_minimum_required(VERSION 3.9)

project(tensor LANGUAGE CUDA CXX C)

find_package(pybind11 REQUIRED)

# Specify CUDA version if needed
find_package(CUDA REQUIRED)
add_subdirectory(pybind11)

# Add the source files
set(SOURCES
    TensorBindings.cpp
    Tensor.cu  # Add your .cu file
)

# Define the C++ standard
set(CMAKE_CXX_STANDARD 11)

# Create the pybind11 module
pybind11_add_module(tensor ${SOURCES})

# Link CUDA libraries
target_link_libraries(tensor PRIVATE CUDA::CUDA)
