from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader,Subset

def create_mnist_6_dataloaders(batch_size, image_size=28, num_workers=0):
    preprocess = transforms.Compose(
        [transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])]      # [0,1] to [-1,1]
        )
    train_dataset = MNIST(root=".",
                          train=True,
                          download=True,
                          transform=preprocess
                          )
    test_dataset = MNIST(root=".",
                         train=False,
                         download=True,
                         transform=preprocess
                         )
    label = 6
    train_indices = [i for i, target in enumerate(train_dataset.targets) if target == label]
    test_indices = [i for i, target in enumerate(test_dataset.targets) if target == label]
    train_dataset_filtered = Subset(train_dataset, train_indices)
    test_dataset_filtered = Subset(test_dataset, test_indices)

    train_loader = DataLoader(train_dataset_filtered, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset_filtered, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers)
    return (train_loader,test_loader)

def create_mnist_dataloaders(batch_size, image_size=28, num_workers=0):
    preprocess = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # [0,1] -> [-1,1]
    ])

    train_dataset = MNIST(root=".",
                          train=True,
                          download=True,
                          transform=preprocess)

    test_dataset = MNIST(root=".",
                         train=False,
                         download=True,
                         transform=preprocess)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers)

    return train_loader, test_loader

from torch.utils.data import DataLoader, TensorDataset, random_split
import torch
import numpy as np
def create_inverse_scatter_dataloaders(batch_size, test_ratio=0.1, num_workers=0, image_size=32):
    # 加载并预处理数据
    data = torch.tensor(np.load('cropped_inverse_scatter.npz')['arr_0'].astype(np.float32))  # (N, H, W)

    # 归一化到 [-1, 1]
    data = 2*(data - data.min()) / (data.max() - data.min()) -1

    # 增加 channel 维度，变成 (N, 1, 64, 64)，方便用于 CNN
    data = data.unsqueeze(1)

    # 划分训练集和测试集
    total_size = len(data)
    test_size = int(total_size * test_ratio)
    train_size = total_size - test_size

    train_data, test_data = random_split(data, [train_size, test_size])

    # 封装为 TensorDataset（无 label，仅图像）
    train_dataset = TensorDataset(train_data.dataset[train_data.indices])
    test_dataset = TensorDataset(test_data.dataset[test_data.indices])

    # 构建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader