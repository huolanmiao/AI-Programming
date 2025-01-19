"""
利用作业五中的Adam_epoch优化器，和作业四实现的自动微分。在网络结构中加入卷积层，在set_structure函数中增加kernel权重，相应修改forward函数，在Adam_epoch中增加更新kernel权重的代码。
"""
from task0_autodiff import *
from task0_operators import *
import numpy as np
import tensorflow as tf
import time
import argparse
from torchvision import transforms
from torchvision.datasets import ImageFolder

from torch.utils.data import DataLoader



def parse_mnist():
    """
    读取MNIST数据集，并进行简单的处理，如归一化
    你可以可以引入任何的库来帮你进行数据处理和读取
    所以不会规定你的输入的格式
    但需要使得输出包括X_tr, y_tr和X_te, y_te
    """
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_train, X_test = X_train / 255.0, X_test / 255.0  # Normalize to [0, 1]
    # 先不进行展平，在forward中先卷积再展平过线性层
    X_train = X_train.reshape(-1,1,28,28).astype(np.float32)
    X_test = X_test.reshape(-1,1,28,28).astype(np.float32)
    return X_train, y_train, X_test, y_test

def parse_imagenet():
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # LeNet 通常处理较小的图像，但这里将其调整为适合 ImageNet 的大小
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 数据加载
    train_data = ImageFolder(root='path_to_train_data', transform=transform)
    val_data = ImageFolder(root='path_to_val_data', transform=transform)
    
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_data, batch_size=64, shuffle=False, num_workers=4)


def set_structure(n, hidden_dim, k, model):
    """
    定义你的网络结构，并进行简单的初始化
    一个简单的网络结构为两个Linear层，中间加上ReLU
    Args:
        n: input dimension of the data.
        hidden_dim: hidden dimension of the network.
        k: output dimension of the network, which is the number of classes.
    Returns:
        List of Weights matrix wrapped by Tensor.
    """

    if model == "pure_linear":
        # 作业五两层线性层
        W1 = np.random.randn(n, hidden_dim).astype(np.float32) / np.sqrt(n)
        W2 = np.random.randn(hidden_dim, k).astype(np.float32) / np.sqrt(hidden_dim)
        w1 = Tensor(W1, requires_grad=True)
        w2 = Tensor(W2, requires_grad=True)
        return [w1, w2]
    if model == "simple_conv":
        # 加一层卷积层
        W_conv = np.random.randn(1, 1, 3, 3).astype(np.float32) / 3 # [Cout, Cin, kH, kW]
        W1 = np.random.randn(n, hidden_dim).astype(np.float32) / np.sqrt(n)
        W2 = np.random.randn(hidden_dim, k).astype(np.float32) / np.sqrt(hidden_dim)
        w_conv = Tensor(W_conv, requires_grad=True)
        w1 = Tensor(W1, requires_grad=True)
        w2 = Tensor(W2, requires_grad=True)
        return [w_conv, w1, w2]
    if model == "LeNet":
        # 实现LeNet结构
        W_conv1 = np.random.randn(6, 3, 9, 9).astype(np.float32) / 9 # [Cout, Cin, kH, kW]
        W_conv2 = np.random.randn(16, 6, 7, 7).astype(np.float32) / 7
        W1 = np.random.randn(16 * 12 * 12, 1000).astype(np.float32) / np.sqrt(16 * 12 * 12)
        W2 = np.random.randn(1000, 500).astype(np.float32) / np.sqrt(1000)
        W3 = np.random.randn(500, 100).astype(np.float32) / np.sqrt(500)
        # 用Tensor包装参数，以利用自动微分框架
        w_conv1 = Tensor(W_conv1, requires_grad=True)
        w_conv2 = Tensor(W_conv2, requires_grad=True)
        w1 = Tensor(W1, requires_grad=True)
        w2 = Tensor(W2, requires_grad=True)
        w3 = Tensor(W3, requires_grad=True)
        return [w_conv1, w_conv2, w1, w2, w3]

def forward(X, weights):
    """
    使用你的网络结构，来计算给定输入X的输出
    Args:
        X : 2D input array of size (num_examples, input_dim).
        weights : list of 2D array of layers weights, of shape [(input_dim, hidden_dim)]
    Returns:
        Logits calculated by your network structure.
    """
    
    if model == "pure_linear":
        w1, w2 = weights
        X = X.reshape([-1, 28 * 28])
        X = relu(matmul(X, w1))
        logits = matmul(X, w2)
        return logits
    if model == "simple_conv":
        w_conv, w1, w2 = weights
        X = conv2d(w_conv, X, 1, 1)
        X = X.reshape([-1, 28 * 28])
        X = relu(matmul(X, w1))
        logits = matmul(X, w2)
        return logits
    if model == "LeNet":
        w_conv1, w_conv2, w1, w2, w3 = weights
        X = max_pooling(relu((conv2d(w_conv1, X, 0, 1))), 4,4,4,4,0,0)
        X = max_pooling(relu((conv2d(w_conv2, X, 0, 1))), 4,4,4,4,0,0)
        X = X.reshape([-1, 16 * 12 * 12])
        X = relu(matmul(X, w1))
        X = relu(matmul(X, w2))
        logits = matmul(X, w3)
        return logits

def softmax_loss(Z, y):
    """ 
    一个写了很多遍的Softmax loss...

    Args:
        Z : 2D numpy array of shape (batch_size, num_classes), 
        containing the logit predictions for each class.
        y : 1D numpy array of shape (batch_size, )
            containing the true label of each example.

    Returns:
        Average softmax loss over the sample.
    """
    ## 请于此填写你的代码
    batch_size = Z.shape[0]
    exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))  # Numerically stable softmax
    probs = exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
    # print(min(probs[np.arange(batch_size), y]))
    log_probs = -np.log(probs[np.arange(batch_size), y]+ 1e-8)  # Select correct class
    return np.mean(log_probs)

def opti_epoch(train_loader, weights, lr = 0.1, batch=100, beta1=0.9, beta2=0.999):
    """
    优化一个epoch
    具体请参考SGD_epoch 和 Adam_epoch的代码
    """
    Adam_epoch(train_loader, weights, lr = lr, batch=batch, beta1=beta1, beta2=beta2)

def Adam_epoch(train_loader, weights, lr = 0.0001, batch=50, beta1=0.9, beta2=0.999):
    """ 
    ADAM优化一个
    本函数应该inplace地修改Weights矩阵来进行优化
    使用Adaptive Moment Estimation来进行更新Weights
    具体步骤可以是：
    1. 增加时间步 $t$。
    2. 计算当前梯度 $g$。
    3. 更新一阶矩向量：$m = \beta_1 \cdot m + (1 - \beta_1) \cdot g$。
    4. 更新二阶矩向量：$v = \beta_2 \cdot v + (1 - \beta_2) \cdot g^2$。
    5. 计算偏差校正后的一阶和二阶矩估计：$\hat{m} = m / (1 - \beta_1^t)$ 和 $\hat{v} = v / (1 - \beta_2^t)$。
    6. 更新参数：$\theta = \theta - \eta \cdot \hat{m} / (\sqrt{\hat{v}} + \epsilon)$。
    其中$\eta$表示学习率，$\beta_1$和$\beta_2$是平滑参数，
    $t$表示时间步，$\epsilon$是为了维持数值稳定性而添加的常数，如1e-8。
    
    Args:
        X : 2D input array of size (num_examples, input_dim).
        y : 1D class label array of size (num_examples,)
        weights : list of 2D array of layers weights, of shape [(input_dim, hidden_dim)]
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch
        beta1 (float): smoothing parameter for first order momentum
        beta2 (float): smoothing parameter for second order momentum

    Returns:
        None
    """
    n = X.shape[0]
    m1, m2 = [np.zeros_like(w) for w in weights], [np.zeros_like(w) for w in weights]
    t = 0
    epsilon = 1e-8
    loss = []
    err = []
    for inputs, labels in train_loader:
        X_batch = inputs.numpy()
        y_batch = labels.numpy()
        
        # Forward pass
        # st = time.perf_counter()
        logits = forward(Tensor(X_batch), weights)
        # print(f"forward: {time.perf_counter() - st}s")
        
        # Compute loss and err
        loss.append(softmax_loss(logits, y_batch))
        err.append(np.mean(logits.argmax(axis=1) != y_batch))
        
        # Compute gradients
        probs = np.exp(logits.numpy() - np.max(logits.numpy(), axis=1, keepdims=True))
        probs /= np.sum(probs, axis=1, keepdims=True)
        grads = probs
        grads[np.arange(batch), y_batch] -= 1
        grads /= batch
        
        # Backward pass: compute gradient w.r.t. weights
        # st = time.perf_counter()
        logits.backward(Tensor(grads))
        # print(f"backward: {time.perf_counter() - st}s")
        
        t += 1
        for i in range(len(weights)):
            dW = weights[i].grad.data.numpy()
            # Update moments
            m1[i] = beta1 * m1[i] + (1 - beta1) * dW
            m2[i] = beta2 * m2[i] + (1 - beta2) * dW**2
            # Bias correction
            m1_hat = m1[i] / (1 - beta1**t)
            m2_hat = m2[i] / (1 - beta2**t)
            # Update weights
            weights[i].data -= lr * m1_hat / (np.sqrt(m2_hat) + epsilon)
            # Zero gradients
            # 实际上不需要清零，因为一则Tensor定义了析构函数，当寿命终结时会删除对应的计算图
            # 一则compute_gradient_of_variables会覆盖先前的梯度
            weights[i].grad.data = Tensor(np.zeros_like(weights[i].grad.data).astype(np.float32))
    return sum(loss)/len(loss), sum(err)/len(err)

def loss_err(test_loader, weights):
    """ 
    计算给定预测结果h和真实标签y的loss和error
    """
    loss = []
    err = []
    for inputs, labels in train_loader:
        X_batch = Tensor(inputs.numpy())
        X_batch.requires_grad = False
        y_batch = labels.numpy()
        logits = forward(X_batch, weights)
        
        loss.append(softmax_loss(logits, y_batch))
        err.append(np.mean(logits.argmax(axis=1) != y_batch))
        
    return sum(loss)/len(loss), sum(err)/len(err)


def train_nn(train_loader, test_loader, hidden_dim = 500,
             epochs=10, lr=0.5, batch=100, beta1=0.9, beta2=0.999, model = "simple_conv"):
    """ 
    训练过程
    """
    # n,k为linear层需要的参数
    n, k = 28*28, y_tr.max() + 1
    weights_tensor = set_structure(n, hidden_dim, k, model = model)
    np.random.seed(0)
    
    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err | Epoch Time |")
    for epoch in range(epochs):
        st = time.perf_counter()
        # step decay 调整学习率
        lr = lr * 0.98**epoch
        
        train_loss, train_err = opti_epoch(train_loader, weights_tensor, lr=lr, batch=batch, beta1=beta1, beta2=beta2)
        # train_loss, train_err = loss_err(train_loader, weights_tensor)
        test_loss, test_err = loss_err(test_loader, weights_tensor)
        dur = time.perf_counter() - st
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} | {:.5f} |"\
              .format(epoch, train_loss, train_err, test_loss, test_err, dur))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='simple_conv', choices=['pure_linear', 'simple_conv', 'LeNet'], help='Model architecture')
    args = parser.parse_args()
    
    model = args.model
    train_loader, test_loader = parse_imagenet()
    ## using Adam optimizer
    train_nn(train_loader, test_loader, hidden_dim=100, epochs=20, lr = 0.001, batch=100, beta1=0.9, beta2=0.999, model = model)
    