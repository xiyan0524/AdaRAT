from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms


def load(dataset_path, batch_size, drop_last):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
    ])

    train_set = torchvision.datasets.CIFAR10(root=dataset_path, train=True, download=True, transform=transform_train)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=drop_last)

    test_set = torchvision.datasets.CIFAR10(root=dataset_path, train=False, download=True, transform=transform_test)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=1)

    return train_loader, test_loader
