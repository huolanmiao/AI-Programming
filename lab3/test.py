import torch 
import numpy as np
import unittest
import torch.nn.functional as F
from tensor import *



# 测试类：Tensor 类实现
class TestTensorOperations(unittest.TestCase):

    def test_tensor_cpu_gpu(self):
        x = np.random.randn(10, 10).astype(np.float32)
        tensor_x = Tensor(x, "CPU")
        self.assertEqual(tensor_x.device, "CPU")
        tensor_x.gpu()
        self.assertEqual(tensor_x.device, "GPU")
        
        
    def test_sigmoid(self):
        # 测试 Sigmoid
        x = np.random.randn(10, 10).astype(np.float32)
        tensor_x = Tensor(x, "CPU")
        result_x = sigmoid(tensor_x)
        target_x = F.sigmoid(torch.from_numpy(x))
        np.testing.assert_almost_equal(result_x.to_numpy().tolist(), target_x.tolist(), decimal=7)
        
        y = np.random.randn(10, 10).astype(np.float32)
        tensor_y = Tensor(y, "GPU")
        result_y = sigmoid(tensor_y)
        target_y = F.sigmoid(torch.from_numpy(y))
        np.testing.assert_almost_equal(result_y.to_numpy().tolist(), target_y.tolist(), decimal=7)

    def test_relu(self):
        # 测试 relu
        x = np.random.randn(10, 10).astype(np.float32)
        tensor_x = Tensor(x, "CPU")
        result_x = relu(tensor_x)
        target_x = F.relu(torch.from_numpy(x))
        np.testing.assert_almost_equal(result_x.to_numpy().tolist(), target_x.tolist(), decimal=7)
        
        y = np.random.randn(10, 10).astype(np.float32)
        tensor_y = Tensor(y, "GPU")
        result_y = relu(tensor_y)
        target_y = F.relu(torch.from_numpy(y))
        np.testing.assert_almost_equal(result_y.to_numpy().tolist(), target_y.tolist(), decimal=7)

    def test_softmax(self):
        # 测试 Softmax
        x = np.random.randn(10, 10).astype(np.float32).astype(np.float32)
        tensor_x = Tensor(x, "GPU")
        result_x = Tensor(x, "GPU")
        softmax(tensor_x, result_x)
        target_x = F.softmax(torch.from_numpy(x), dim=1)
        np.testing.assert_almost_equal(result_x.to_numpy().tolist(), target_x.tolist(), decimal=7)

    def test_cross_entropy_loss(self):
        # 测试 CrossEntropyLoss
        pred = np.random.rand(10, 5).astype(np.float32)
        target = np.random.randint(0, 5, size=10).astype(np.float32)
        tensor_pred = Tensor(pred, "GPU")
        tensor_target = Tensor(target.reshape(1,10), "GPU")
        tensor_loss = cross_entropy_loss(tensor_pred, tensor_target)
        target_loss = F.nll_loss(torch.log(torch.from_numpy(pred)), torch.from_numpy(target).long())
        np.testing.assert_almost_equal(tensor_loss, target_loss, decimal=7)

# 测试类：Fully Connected Layer (FC) 测试
class TestFullyConnectedLayer(unittest.TestCase):

    def test_fully_connected(self):
        # 测试 Fully Connected Layer
        in_features = 10
        out_features = 5
        x = np.random.randn(10, in_features).astype(np.float32)
        weights = np.random.randn(in_features, out_features).astype(np.float32)
        bias = np.random.randn(1,out_features).astype(np.float32)
        
        tensor_x = Tensor(x, "GPU")
        tensor_weights = Tensor(weights, "GPU")
        tensor_bias = Tensor(bias, "GPU")
        tensor_output = Tensor([10, out_features], "GPU")
        forward_fc(tensor_x, tensor_weights, tensor_bias, tensor_output)
        target_output = np.dot(x, weights) + bias
        np.testing.assert_almost_equal(tensor_output.to_numpy().tolist(), target_output.tolist(), decimal=6)
        
        
# 测试类：卷积层测试
class TestConvolutionLayer(unittest.TestCase):

    def test_convolution(self):
        # 测试 2D 卷积
        x = np.random.randn(2, 3, 4, 4).astype(np.float32)  # [N, Cin, H, W]
        weights = np.random.randn(2, 3, 3, 3).astype(np.float32)  # [Cout, Cin, kH, kW]
        
        tensor_x = Tensor(x, "GPU")
        tensor_weights = Tensor(weights, "GPU")
        tensor_output = Tensor([2,2,4,4], "GPU")
        forward_conv(tensor_x, tensor_weights, tensor_output)
        target_output = F.conv2d(torch.from_numpy(x), torch.from_numpy(weights), padding=1)
        np.testing.assert_almost_equal(tensor_output.to_numpy().tolist(), target_output.tolist(), decimal=5)
        
# 测试类：池化层测试
class TestMaxpoolingLayer(unittest.TestCase):

    def test_Maxpooling(self):
        x = np.random.randn(2, 3, 4, 4).astype(np.float32)  # [N, Cin, H, W]
        tensor_x = Tensor(x, "GPU")
        tensor_output = Tensor([2, 3, 2, 2], "GPU") # [N, Cin, H_out, W_out]
        tensor_output_mask = Tensor([2, 3, 2, 2], "GPU")
        max_pooling(tensor_x, tensor_output, tensor_output_mask, 2, 2, 2, 2, 0, 0)
        target_output = F.max_pool2d(torch.from_numpy(x), kernel_size=2, stride=2)
        self.assertEqual(tensor_output.to_numpy().tolist(), target_output.tolist())

# t = Tensor([2,2], "GPU")
# t.shape
# t.device
# t.show_tensor()

# t.set_data(np.array([[1,2],[3,4]]))
# t.show_tensor()

# t.set_data(np.array([[1,2],[3,4],[5,6]]))
# t.show_tensor()

# t.set_data(np.array([[[1,2],[3,4]],[[5,6],[7,8]]]))
# t.show_tensor()

# s = t.cpu()
# s.set_data(np.array([[-1,2],[3,4]]))
# s.show_tensor()
# t.show_tensor()

# u = sigmoid(s)
# u.show_tensor()

# v = relu(s)
# v.show_tensor()

# v.gpu()
# result = v
# softmax(v, result)
# v.show_tensor()
# result.show_tensor()


# test fc
# in_features = 5
# out_features = 6
# x = np.random.randn(10, in_features).astype(np.float32)
# weights = np.random.randn(in_features, out_features).astype(np.float32)
# bias = np.random.randn(1, out_features).astype(np.float32)
# # bias = np.zeros((1, out_features))
# print(bias.shape)
# # x = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
# # weights = np.array([[1,2],[3,4],[5,6]])
# # bias = np.array([[1,2]])

# tensor_x = Tensor(x, "GPU")
# # tensor_x.show_tensor()
# # print(x)
# tensor_weights = Tensor(weights, "GPU")
# # tensor_weights.show_tensor()
# # print(weights)
# tensor_bias = Tensor(bias, "GPU")
# tensor_output = Tensor([10, out_features], "GPU")
# forward_fc(tensor_x, tensor_weights, tensor_bias, tensor_output)
# tensor_output.show_tensor()
# target_output = np.dot(x, weights) + bias
# print(target_output)


# test conv
# x = np.random.randn(2, 3, 4, 4)  # [N, Cin, H, W]
# weights = np.random.randn(2, 3, 3, 3)  # [Cout, Cin, kH, kW]

# tensor_x = Tensor(x, "GPU")
# tensor_weights = Tensor(weights, "GPU")
# tensor_output = Tensor([2,2,4,4], "GPU")
# forward_conv(tensor_x, tensor_weights, tensor_output)
# target_output = F.conv2d(torch.from_numpy(x), torch.from_numpy(weights), padding=1)
# tensor_output.show_tensor()
# print(target_output)

# test pool
# x = np.random.randn(1, 1, 5, 5)  # [N, Cin, H, W]
# tensor_x = Tensor(x, "GPU")
# tensor_output = Tensor([1, 1, 2, 2], "GPU") # [N, Cin, H_out, W_out]
# tensor_output_mask = Tensor([1, 1, 2, 2], "GPU")
# max_pooling(tensor_x, tensor_output, tensor_output_mask, 2, 2, 2, 2, 0, 0)
# target_output = F.max_pool2d(torch.from_numpy(x), kernel_size=2, stride=2)
# # print(x)
# print(tensor_output.to_numpy())
# print(target_output)