import torch
from torchvision.transforms import v2 as transforms

basic_transform = transforms.Compose(
    [transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)]
)
