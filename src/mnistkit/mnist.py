import torchvision.transforms.v2 as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader


def mnist(
    train_batch_size: int = 256,
    test_batch_size: int = 256,
    train_shuffle: bool = True,
    test_shuffle: bool = False,
    transforms=transforms.ToTensor(),
    root: str = ".",
):
    train_dataloader = DataLoader(
        MNIST(root=root, download=True, transform=transforms, train=True),
        batch_size=train_batch_size,
        shuffle=train_shuffle,
    )
    test_dataloader = DataLoader(
        MNIST(root=root, download=True, transform=transforms, train=False),
        batch_size=test_batch_size,
        shuffle=test_shuffle,
    )
    return (train_dataloader, test_dataloader)
