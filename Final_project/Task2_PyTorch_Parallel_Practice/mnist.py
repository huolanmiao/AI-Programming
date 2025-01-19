import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm
import time
import argparse

from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import os

# 数据增强
transform_train = transforms.Compose([
    # Random affine transformation
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.8, 1.2)),  
    transforms.ToTensor(),  
    transforms.Normalize((0.5,), (0.5,))  
])

transform_test = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize((0.5,), (0.5,)) 
])


def get_data_loaders(batch_size, local_rank, world_size, num_workers):
    train_sampler = torch.utils.data.DistributedSampler(
        train_dataset, num_replicas=world_size, rank=local_rank, shuffle=True
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size//world_size, sampler=train_sampler, num_workers=num_workers)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    
    return train_loader



# 定义LeNet模型
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)  
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5) 
        self.fc1 = nn.Linear(16 * 4 * 4, 120)  
        self.fc2 = nn.Linear(120, 84)  
        self.fc3 = nn.Linear(84, 10)   
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='base', choices=['base', 'DP', 'DDP'], help='Training mode: base, DP')
    parser.add_argument('--batch_size', type=int, default=64, help='Global batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=0, help='num_workers for DataLoader')
    args = parser.parse_args()

    mode = args.mode
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    num_workers = args.num_workers
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # is this a ddp run?
    ddp = int(os.environ.get('LOCAL_RANK', -1)) != -1 
    
        
    # 加载数据集
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)
    
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    model = LeNet().to(device)
    if ddp:
        # DDP
        mode = "DDP"
        init_process_group(backend='nccl')
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        print(ddp_world_size)
        device = f'cuda:{ddp_local_rank}'
        
        model.to(device)
        model = DDP(model, device_ids=[ddp_local_rank])
        trainloader = get_data_loaders(batch_size, ddp_local_rank, ddp_world_size, num_workers=num_workers)
    elif mode == "DP":
        # DP
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)
        ddp_rank = 0
    else:
        # base
        ddp_rank = 0
        
    master_process = ddp_rank == 0
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    st0 = time.perf_counter()
    # 模型训练
    model.train()
    num_epochs = 10
    losses = []
    for epoch in range(num_epochs):
        st = time.perf_counter()
        loss_accum = 0.0
        for inputs, labels in trainloader:
            # print(f"time of load: {time.perf_counter()-st}")
            # st = time.perf_counter()
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss_accum += loss.detach()
            loss.backward()
            # DDP会在所有进程backward结束时，执行all-reduce
            # 梯度求和后会对num_gpus取平均，每个卡上拿到的梯度是相同的
            optimizer.step()  
            # print(f"time of update: {time.perf_counter()-st}")
            # st = time.perf_counter()
        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
            avg_loss = loss_accum / len(trainloader)
            losses.append(avg_loss.item())
            dur = time.perf_counter() - st
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Dur: {dur}')

    
    if master_process:
        dur = time.perf_counter() - st0
        print(f'Total training time: {dur}s.')
        
        # 模型测试
        model.eval()
        correct = 0
        total = 0
        class_correct = [0] * 10
        class_total = [0] * 10

        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                for i in range(10):
                    mask = (labels == i)
                    class_correct[i] += (predicted[mask] == labels[mask]).sum().item()
                    class_total[i] += mask.sum().item()

        print(f'Accuracy of the network on the 10000 test images: {100 * correct / total:.2f}%')

        for i in range(10):
            if class_total[i] > 0:
                print(f'Accuracy of class {i}: {100 * class_correct[i] / class_total[i]:.2f}%')
            
        # save model
        # PATH = './Mnist_net.pth'
        # torch.save(model.state_dict(), PATH)

        # 绘制Loss Curve
        plt.plot(range(1, num_epochs + 1), losses, marker='.')
        plt.title('Loss Curve')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid()
        plt.savefig(f"./{mode}_curve.png")