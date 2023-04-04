import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.utils as vutils


import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

from typing import List, Tuple


from PIL import Image

from pet_3.data import Pets

import time


def search_for_weird_labels(
    dataset_name: str,
) -> Tuple[List[int], List[torch.Tensor], List[torch.Tensor], Pets]:
    """
    Searches dataset for weird labels where the label is all 0's or all 1's

    Arguments:
    ----------
        dataset_name : str
                        > Name of the dataset to search

    Returns:
    ----------
        weird_idxs : List[int]
                    > List of indexes where the label is all 0's or all 1's
    """
    # define data loader
    dataset = Pets("./pet_3", dataset_name, binary_labels=True)
    loader = DataLoader(dataset, batch_size=1, num_workers=2)

    # search for weird labels
    weird_idxs: List[int] = []
    weird_imgs: List[torch.Tensor] = []
    weird_labels: List[torch.Tensor] = []

    # label search
    for idx, (image, label) in enumerate(loader):
        uniques = torch.unique(label)
        if torch.min(uniques) == torch.max(uniques):
            weird_idxs.append(idx)
            weird_imgs.append(image)
            weird_labels.append(label)

    return weird_idxs, weird_imgs, weird_labels, dataset


def save_image_grid(
    idxs: List,
    images: torch.Tensor,
    labels: torch.Tensor,
    dataset: Pets,
    file_name: str,
) -> None:
    """
    Saves a grid of images and labels to a file

    Arguments:
    ----------
        images : torch.Tensor
                > Tensor of images
        labels : torch.Tensor
                > Tensor of labels
        dataset : Pets
                > Dataset object
    """
    # image names
    image_names: List[str] = []

    # print out the first 8 images and labels using matplotlib
    fig, axs = plt.subplots(2, len(images), figsize=(20, 8))

    for i, ax in enumerate(axs.flat):

        if i < len(images):
            # get weird image
            img = images[i].squeeze().permute(1, 2, 0)
            ax.imshow(img)
            ax.set_title(f"Image Name: \n{dataset.images[idxs[i]]}")
            image_names.append(dataset.images[idxs[i]])
        else:
            # get weird label
            label_idx = i - len(images)
            seg_label = labels[label_idx].squeeze()
            ax.imshow(seg_label)
            ax.set_title(f"Image Label")

        ax.set_xticks([])
        ax.set_yticks([])

        # save the figure
    plt.suptitle("Images and Labels", fontsize=20, fontweight="bold")
    fig.savefig(f"{file_name}")
    plt.close(fig)

    print("Image Names: ", image_names)


# pylint: disable-all
if __name__ == "__main__":

    weird_idxs, weird_images, weird_labels, dataset = search_for_weird_labels(
        "test"
    )

    save_image_grid(
        weird_idxs,
        weird_images,
        weird_labels,
        dataset,
        "weirdo_segmentation_grid.png",
    )
