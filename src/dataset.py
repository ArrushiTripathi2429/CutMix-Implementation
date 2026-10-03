from torchvision import datasets


def load_cifar10(data_dir="./data/raw"):
    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True
    )

    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True
    )

    return train_dataset, test_dataset