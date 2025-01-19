from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext

# Define the extension
ext_modules = [
    Pybind11Extension(
        "example",  # Module name
        ["example.cpp"],  # Source files
    ),
]

# Setup configuration
setup(
    name="example",
    version="0.1",
    description="Pybind11 example module",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},  # Use Pybind11 build_ext
)
