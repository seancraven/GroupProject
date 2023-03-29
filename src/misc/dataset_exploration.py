# pylint: skip-file
import torchvision
import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader


def min_img_size(dataset):
    min_hight: int = 0
    min_width: int = 0
    mean_width: float = 0.0
    mean_height: float = 0.0
    msq_width: float = 0.0
    msq_height: float = 0.0
    for i in range(len(dataset)):
        img, _ = dataset[i]
        mean_height += float(img.shape[2])
        mean_width += float(img.shape[1])
        msq_height += float(img.shape[2]) ** 2
        msq_width += float(img.shape[1]) ** 2

        if min_width < img.shape[1]:
            min_width = img.shape[1]
        if min_hight < img.shape[2]:
            min_hight = img.shape[2]
    mean_height /= len(dataset)
    mean_width /= len(dataset)
    var_height = msq_height / len(dataset) - mean_height**2
    var_width = msq_width / len(dataset) - mean_width**2

    print(f"Height: mean {mean_height:.2f}, std {var_height**0.5:.2f}")
    print(f"Width: mean {mean_width:.2f}, std {var_width**0.5:.2f}")

    return min_width, min_hight


if __name__ == "__main__":
    dataset = datasets.OxfordIIITPet(
        "./",
        target_types="segmentation",
        download=True,
        transform=transforms.ToTensor(),
        target_transform=transforms.ToTensor(),
    )
    # w, h = min_img_size(dataset)
    # print(f"Max width {w}, Max hegiht {h}")
    loader = DataLoader(dataset, batch_size=1)
    net = nn.Sequential(*[nn.Conv2d(3, 3, 2), nn.ReLU(), nn.MaxPool2d(5)] * 5)
    for data, label in loader:
        out = net(data)
        print(out.shape)
