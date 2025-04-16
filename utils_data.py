from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader,Subset

def create_mnist_dataloaders(batch_size, image_size=28, num_workers=0):
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