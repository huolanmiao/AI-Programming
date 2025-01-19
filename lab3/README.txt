链接程序：
python setup.py develop
运行unit test：
python -m unittest ./test.py
加载mnist数据集：
cd ./MNIST
python ./get_array_mnist.py

结构：
TensorBinding.cpp 定义PYBIND11_MODULE
setup.py 用cpp_extension编译cuda file
其他各个.cu和.h file与lab2大体相同
./MNINST 负责处理MNIST数据集，加载数据集，转换为numpy array，然后放到Tensor中
test.py 中定义各种unit test进行测试