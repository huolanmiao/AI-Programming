import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from tensor import Tensor

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_set = datasets.MNIST('./data', train=True, transform=transform, download=True)
test_set = datasets.MNIST('./data', train=False, transform=transform, download=True)

train_image = train_set.data.numpy()
train_labels = train_set.targets.numpy()
test_image = test_set.data.numpy()
test_labels = test_set.targets.numpy()

print(f"train_image shape {train_image.shape}")
print(f"test_image shape {test_image.shape}")

my_train_image = Tensor(train_image, "GPU")
my_train_labels = Tensor(train_labels, "GPU")
my_test_image = Tensor(test_image, "GPU")
my_test_labels = Tensor(test_labels, "GPU")

# 验证
print(my_train_image.to_numpy().shape)
print(my_train_labels.to_numpy().shape)