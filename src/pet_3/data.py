"""
Torch datasets for the pet dataset.
"""
from __future__ import annotations
import os
import random
import torch

from functools import cache
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.transforms import Resize, ToTensor
from typing import Tuple, Union
from copy import deepcopy

from .download_utils import (
    _populate_data,
    _move_files_according_to_file,
    return_to_train,
)


class Pets(Dataset):
    """
    Dataset for the pet dataset.

    The default implementation returns all the files as they are found on the
    pet_3 website. If foreground background binary classification is required,
    then set the binary_labels = True. This gives foreground = 1,background = 0.
    Additionally, edges are included into the foreground.


    Examples:
        Binary labeled dataset:
        train = pet_3.Data.Pets("./src/pet_3", split = "all_train", binary_labels = True)
        train[i][0] # the ith image
        train[i][1] # the ith label binary mode

        Binary unlabeled dataset 60% labeled
        unlabeled, labeled = pet_3.Data.Pets(
            "./src/pet_3",
            split = "labeled_unlabeled",
            labeled_fraction = 0.6,
            binary_labels=True,
        )

    """

    def __init__(
        self,
        root: str,
        split: str = "all_train",
        labeled_fraction: Union[float, None] = None,
        binary_labels: bool = True,
        shuffle: bool = False,
    ):
        # Check arguments are valid.
        assert split in [
            "labeled_unlabeled",
            "test",
            "all_train",
        ], """
        The split parameter should be one of:
         ("labeled_unlabeled"), ("test"), ("all_train").
        """

        assert (
            labeled_fraction in [i / 10 for i in range(9)] + [0.01, 0.02, 0.05]
            or labeled_fraction is None
        ), """
        Invalid fraction must be one of[0.01, 0.02, 0.05, 0.1, 0.2, 0.3, ... ,0.9]."""

        self.root = root
        self.labeled_fraction = labeled_fraction
        self.shuffle = shuffle

        if split == "test":
            self.parent_data_folder = os.path.join(root, "test_data")
        else:
            self.parent_data_folder = os.path.join(root, "train_data")

        # If the data is downloaded.
        if not os.path.isdir(self.parent_data_folder):
            _populate_data(self.root)

        if split in ("test", "all_train"):
            self.image_folder = os.path.join(self.parent_data_folder, "images")
            self.label_folder = os.path.join(self.parent_data_folder, "labels")
            self.images = os.listdir(self.image_folder)
            self.images = [img for img in self.images if _valid_images(img)]
            self.images.sort()
            if self.shuffle:
                random.shuffle(self.images)
            #  self.labels = os.listdir(self.label_folder)

        if split == "labeled_unlabeled":
            print("call get datasets method.")
        self.binary_labels = binary_labels

    def get_datasets(self) -> Tuple[PetsUnlabeled, Pets]:
        """Returns two new dataset objects, one of unlabeled data,
        the other of labeled."""

        if isinstance(self.labeled_fraction, float):
            unlabeled = PetsUnlabeled(self.root, self.labeled_fraction)
            return unlabeled, Pets(
                self.root,
                "all_train",
                self.labeled_fraction,
                binary_labels=self.binary_labels,
                shuffle=self.shuffle,
            )
        raise ValueError("labeled_fraction must be a float for a dataset split")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img = ToTensor()(
            Image.open(os.path.join(self.image_folder, self.images[idx])).convert("RGB")
        )
        label_name = self.images[idx].split(".")[0] + ".png"
        label = ToTensor()(Image.open(os.path.join(self.label_folder, label_name)))
        img = Resize((256, 256))(img)
        if self.binary_labels:
            label[label < 0.0075] = 1  # Only Edge
            label[label > 0.009] = 1  # Only foreground
            label[(0.0075 <= label) & (label <= 0.009)] = 0
            label = Resize((256, 256))(label).round().long()

        label = label.flatten(1, -1)

        return img, label

    @property
    def name(self) -> str:
        """Returns the dataset name"""
        if self.labeled_fraction is not None:
            return f"Pet 3 labeled fraction {self.labeled_fraction}"

        return "Pet 3"

    def validation_split(self, split_fraction: float) -> Tuple[Pets, Pets]:
        """Split the training data into a training and validation set. Note we return the images names,
        not the actual images."""
        # Load image names
        img_names = self.images.copy()
        # Make sure the two variables do not reference the same object in memory
        assert img_names is not self.images
        random.shuffle(img_names)
        n_img = len(img_names)
        val_img_names = img_names[: int(n_img * split_fraction)]
        train_img_names = img_names[int(n_img * split_fraction) :]
        # Check that the two sets are disjoint
        assert len(set(val_img_names).intersection(set(train_img_names))) == 0

        val_pets = deepcopy(self)
        val_pets.images = val_img_names
        self.images = train_img_names

        return val_pets, self


class PetsUnlabeled(Dataset):
    """Class to manage unlabeled data for pet dataset."""

    def __init__(self, root: str, labeled_fraction: float, shuffle: bool = True):
        move_files = os.path.join(root, f"unlabeled_train_{labeled_fraction}.txt")
        self.labeled_fraction = labeled_fraction
        self.train_dir = os.path.join(root, "train_data")
        self.unlabeled_dir = os.path.join(self.train_dir, "unlabeled")

        _move_files_according_to_file(
            move_files, self.train_dir, self.train_dir, "unlabeled"
        )

        self.images = os.listdir(self.unlabeled_dir)
        self.images = [img for img in self.images if _valid_images(img)]
        self.images.sort()
        if shuffle:
            random.shuffle(self.images)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = Image.open(os.path.join(self.unlabeled_dir, self.images[idx])).convert(
            "RGB"
        )
        img = Resize((256, 256))(ToTensor()(img))
        return img

    def __del__(self):
        """returns all items to images, when the object is garbage collected."""
        self.close()

    def close(self):
        """Moves images from train_data/unlabeled to train_data/images."""
        return_to_train(self.train_dir)

    @property
    def name(self) -> str:
        """Returns the dataset name"""
        return f"Pet 3 labeled fraction {self.labeled_fraction}"


def _valid_images(f_name: str) -> bool:
    """Returns true if the file is a valid image."""
    bad_names = [
        "Egyptian_Mau_162.jpg",
        "Egyptian_Mau_20.jpg",
        "japanese_chin_199.jpg",
        "miniature_pinscher_14.jpg",
        "saint_bernard_15.jpg",
        "staffordshire_bull_terrier_2.jpg",
    ]
    if f_name in bad_names or f_name[-3:] != "jpg":
        return False
    return True


class Oracle:
    """
    An oracle which can evaluate the accuracy of pseudolabel given the image,
    the pseudolabel, and the pseudolabel mask.
    """

    def __init__(self):
        # Build a cache of the dataset, and then delete the dataset object
        # so we don't mess up the files being moved around.
        # The cache will stay in memory, which might be problematic depending
        # on amounts of ram... I haven't tested this yet.
        # I'm not remotely proud of this.
        dataset = Pets("data", "all_train")
        for img, _ in Pets("data", "all_train"):
            self.__image_label(img)
        del dataset

    @cache
    @staticmethod
    def __image_label(self, image: torch.Tensor) -> torch.Tensor:
        """Returns the label of the image."""
        for img, label in self.dataset:
            if torch.equal(img, image):
                return label

    def __batch_labels(self, batch: torch.Tensor) -> torch.Tensor:
        """Returns the labels of the batch."""
        return torch.stack([self.__image_label(img) for img in batch])

    # def __call__(
    #     self, batch: torch.Tensor, pseudolabels: torch.Tensor, masks: torch.Tensor
    # ) -> float:
    #     """Returns the accuracy of the pseudolabels given the batch."""
    #     labels = self.__batch_labels(batch)
    #     pseudolabels = pseudolabels * masks
    #     labels = labels * masks
    #     # TODO: Implement this function.
    #     pass
