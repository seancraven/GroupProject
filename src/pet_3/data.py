import os
import random
from typing import List, Optional, Tuple, Union, Dict

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.transforms import Resize, ToTensor

from src.pet_3.download_utils import _populate_data

TrainValidatePseudoSplit = Tuple["PetsLabeled", "PetsLabeled", "PetsUnlabeled"]
TrainPseudoSplit = Tuple["PetsLabeled", "PetsUnlabeled"]
Named = Tuple[Union[TrainPseudoSplit, TrainValidatePseudoSplit], str]


class _BasePets(Dataset):
    INVALID_IMAGES = (
        "Egyptian_Mau_162.jpg",
        "Egyptian_Mau_20.jpg",
        "japanese_chin_199.jpg",
        "miniature_pinscher_14.jpg",
        "saint_bernard_15.jpg",
        "staffordshire_bull_terrier_2.jpg",
    )

    def __init__(
        self,
        filenames: List[str],
        image_folder: str,
        label_folder: Optional[str] = None,
    ):
        # Filenames must be provided shuffled.
        self.filenames = filenames
        self.image_folder = image_folder
        self.label_folder = label_folder

    def __len__(self) -> int:
        return len(self.filenames)

    def __get_image(self, idx) -> torch.Tensor:
        path = os.path.join(self.image_folder, self.filenames[idx])
        image = ToTensor()(Image.open(path).convert("RGB"))
        image = Resize((256, 256))(image)
        return image

    def __get_label(self, idx) -> torch.Tensor:
        if self.label_folder is None:
            raise ValueError("No label folder provided.")
        label_name = self.filenames[idx].split(".")[0] + ".png"
        path = os.path.join(self.label_folder, label_name)
        label = ToTensor()(Image.open(path))
        label[label < 0.0075] = 1  # Only edge
        label[label > 0.009] = 1
        label[(0.0075 <= label) & (label <= 0.009)] = 0
        label = Resize((256, 256))(label).round().long()
        label = label.squeeze(0).flatten()
        return label

    def __getitem__(self, idx: int) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        image = self.__get_image(idx)
        if self.label_folder is None:
            return image

        label = self.__get_label(idx)
        return image, label

    @staticmethod
    def _is_valid_image(file_name: str) -> bool:
        """Returns true if the file is a valid image."""
        if file_name in _BasePets.INVALID_IMAGES or file_name[-3:] != "jpg":
            return False
        return True


class PetsLabeled(_BasePets):
    def __init__(self, test: bool, filenames, image_folder, label_folder):
        super().__init__(filenames, image_folder, label_folder)
        self.test = test


class PetsUnlabeled(_BasePets):
    def __init__(self, filenames, image_folder):
        super().__init__(filenames, image_folder, None)


class PetsDataFetcher:
    def __init__(
        self,
        root: str,
    ) -> None:

        self.root = root
        self.test_path = os.path.join(self.root, "test_data")
        self.train_path = os.path.join(self.root, "train_data")

        if not all(os.path.isdir(x) for x in [self.test_path, self.train_path]):
            _populate_data(self.root)

    @staticmethod
    def _get_valid_files_from_txt(txt_file: str) -> List[str]:
        with open(txt_file, "r") as f:
            files = f.read().splitlines()
        return [file for file in files if _BasePets._is_valid_image(file)]

    def get_test_data(self) -> PetsLabeled:
        """Returns the test data."""
        test_txt = os.path.join(self.root, "test.txt")
        test_filenames = self._get_valid_files_from_txt(test_txt)
        return PetsLabeled(
            test=True,
            filenames=test_filenames,
            image_folder=os.path.join(self.root, "test_data", "images"),
            label_folder=os.path.join(self.root, "test_data", "labels"),
        )

    def get_train_data(
        self,
        label_proportion: float = 1.0,
        validation_proportion: float = 0.0,
        seed: int = 0,
        class_balance: bool = False,
    ) -> Union[TrainPseudoSplit, TrainValidatePseudoSplit]:
        """Returns the train data, generated randomly from the given seed.
        The class balance split option will not give the same random splits
        as with class_balance=False.
        Args:
            label_proportion: The proportion of the data that is labeled.
            validation_proportion: The proportion of the labeled data that is used for validation.
            seed: The seed used to generate the random split.
        Returns:
            A tuple of the train and validation data, and unlabeled data.
        """
        assert 0 <= label_proportion <= 1, "Label proportion must be between 0 and 1."
        assert (
            0 <= validation_proportion <= 1
        ), "Validation proportion must be between 0 and 1."

        random.seed(seed)
        train_txt = os.path.join(self.root, "train.txt")
        # Stop wierd behaviour across os.
        all_filenames = sorted(self._get_valid_files_from_txt(train_txt))
        random.shuffle(all_filenames)

        # Split into labeled and unlabeled
        if class_balance:
            (v, t, u) = _class_balanced_split(
                all_filenames, label_proportion, validation_proportion
            )
            validation_filenames = v
            train_filenames = t
            unlabeled_filenames = u
        else:
            num_labeled = int(len(all_filenames) * label_proportion)
            num_validation = int(len(all_filenames) * validation_proportion)

            validation_filenames = all_filenames[:num_validation]
            train_filenames = all_filenames[
                num_validation : num_labeled + num_validation
            ]
            unlabeled_filenames = all_filenames[num_labeled + num_validation :]

        assert len(set(train_filenames).intersection(set(validation_filenames))) == 0
        assert len(set(train_filenames).intersection(set(unlabeled_filenames))) == 0
        assert (
            len(set(validation_filenames).intersection(set(unlabeled_filenames))) == 0
        )

        if validation_proportion > 0 and len(validation_filenames) == 0:
            raise ValueError("Validation proportion is too small.")

        train, validate = map(
            lambda x: PetsLabeled(
                False,
                x,
                image_folder=os.path.join(self.train_path, "images"),
                label_folder=os.path.join(self.train_path, "labels"),
            ),
            (train_filenames, validation_filenames),
        )
        unlabeled = PetsUnlabeled(
            unlabeled_filenames, image_folder=os.path.join(self.train_path, "images")
        )

        if validation_proportion > 0:
            return train, validate, unlabeled
        return train, unlabeled

    def get_train_data_with_name(
        self,
        label_proportion: float = 1.0,
        validation_proportion: float = 0,
        seed: int = 0,
        class_balance: bool = False,
    ) -> Named:
        """Returns the train data, generated randomly from the given seed
        Args:
            label_proportion: The proportion of the data that is labeled.
            validation_proportion: The proportion of the labeled data that is used for validation.
            seed: The seed used to generate the random split.
        Returns:
            A tuple of the train and validation data, and unlabeled data.
        """
        return (
            self.get_train_data(label_proportion, validation_proportion, seed),
            f"pets_l_{label_proportion}_v_{validation_proportion}",
        )


def _deterministic_label_splits(root: str) -> List[float]:
    """Finds all of the predetermined validation splits."""
    splits: List[float] = []
    for file in os.listdir(root):
        if file.endswith(".txt"):
            split_fract_string = file.split("_")[1].split(".")[0]
            split_fract = float(split_fract_string)
            splits.append(split_fract)

    return splits


def _class_balanced_split(
    filenames: List[str],
    label_proportion: float,
    validation_proportion: float,
) -> Tuple[List[str], List[str], List[str]]:
    """Splits the data into labeled, unlabeled, and validation data,
    while maintaining class balance.
    Args:
        filenames: The filenames of the data.
        label_proportion: The proportion of the data that is labeled.
        validation_proportion: The proportion of the labeled data that is used for validation.
    Returns:
        A tuple of the validation, train, and unlabeled data.
    """

    class_dict: Dict[str, List[str]] = {}
    validation_filenames: List[str] = []
    train_filenames: List[str] = []
    unlabeled_filenames: List[str] = []
    # Split by class
    for file in filenames:
        class_name = "".join(file.split("_")[:-1])
        if class_name not in class_dict:
            class_dict[class_name] = []
        class_dict[class_name].append(file)

    for classed_files in class_dict.values():
        num_labeled = int(len(classed_files) * label_proportion)
        num_validation = int(len(classed_files) * validation_proportion)
        validation_filenames += classed_files[:num_validation]
        train_filenames += classed_files[num_validation : num_validation + num_labeled]
        unlabeled_filenames += classed_files[num_labeled + num_validation :]

        assert len(set(train_filenames).intersection(set(validation_filenames))) == 0
        assert len(set(train_filenames).intersection(set(unlabeled_filenames))) == 0
        assert (
            len(set(validation_filenames).intersection(set(unlabeled_filenames))) == 0
        )
    return validation_filenames, train_filenames, unlabeled_filenames
