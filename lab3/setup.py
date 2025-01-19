import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
__version__ = '0.0.1'
  
setup(
    name='tensor',
    version=__version__,
    packages=find_packages(),
    install_requires=['torch'],
    python_requires='>=3.8',
    ext_modules=[
        CUDAExtension(
            name='tensor',
            sources=[
                'Tensor.cu',
                'TensorBindings.cpp',
                'activations.cu',
                'max_pooling.cu',
                'fully_connected.cu',
                'conv.cu',
                'CEloss.cu'
            ],
            libraries=["cublas", "cudart", "curand"]
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension.with_options(use_ninja=False)
    },
    classifiers=[
        'license :: OSI Approved :: MIT License',
    ],
)