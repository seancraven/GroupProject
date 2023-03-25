import os
import random
import torch

from abc import ABC, abstractmethod
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.transforms import Resize, ToTensor
from typing import List, Tuple, Optional

from src.pet_3.download_utils import _populate_data

TrainValidatePseudoSplit = Tuple['PetsLabeled', 'PetsLabeled', 'PetsUnlabeled']
TrainPseudoSplit = Tuple['PetsLabeled', 'PetsUnlabeled']


class _BasePets(Dataset):
    INVALID_IMAGES = (
        "Egyptian_Mau_162.jpg",
        "Egyptian_Mau_20.jpg",
        "japanese_chin_199.jpg",
        "miniature_pinscher_14.jpg",
        "saint_bernard_15.jpg",
        "staffordshire_bull_terrier_2.jpg",
    )
    
    def __init__(self, filenames: List[str], image_folder: str, label_folder: Optional[str]=None):
        # Filenames must be provided shuffled.
        self.filenames = filenames
        self.image_folder = image_folder
        self.label_folder = label_folder

    def __len__(self) -> int:
        return len(self.filenames)

    def __get_image(self, idx) -> torch.Tensor:
        path = os.path.join(self.image_folder, self.filenames[idx])
        image = ToTensor()(Image.open(path).convert("RGB"))
        image = Resize((256,256))(image)
        return image
    
    def __get_label(self, idx) -> torch.Tensor:
        label_name = self.filenames[idx].split(".")[0]+".png"
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
    def __init__(self, root: str) -> None:
        self.root = root
        self.test_path = os.path.join(self.root, "test_data")
        self.train_path = os.path.join(self.root, "train_data")

        if not all(
            os.path.isdir(x)
            for x in [self.test_path, self.train_path]
        ):
            _populate_data(self.root)

    @staticmethod
    def _get_valid_files_from_txt(txt_file: str) -> List[str]:
        with open(txt_file, 'r') as f:
            files = f.read().splitlines()
        return [file for file in files if _BasePets._is_valid_image(file)]

    def get_test_data(self) -> PetsLabeled:
        """Returns the test data."""
        test_txt = os.path.join(self.root, 'test.txt')
        test_filenames = self._get_valid_files_from_txt(test_txt)
        return PetsLabeled(
            test=True,
            filenames=test_filenames,
            image_folder=os.path.join(self.root, 'test_data', 'images'),
            label_folder=os.path.join(self.root, 'test_data', 'labels'),
        )

    def get_train_data(
        self,
        label_proportion: float=1.0,
        validation_proportion: float=0.0,
        seed: Optional[int] = None
    ) -> TrainPseudoSplit | TrainValidatePseudoSplit:
        """Returns the train data, generated randomly from the given seed """
        if seed is not None:
            random.seed(seed)
        train_txt = os.path.join(self.root, 'train.txt')
        all_filenames = sorted(self._get_valid_files_from_txt(train_txt))
        random.shuffle(all_filenames)
        
        # Split into labeled and unlabeled
        num_labeled = int(len(all_filenames) * label_proportion)
        num_validation = 500


        validation_filenames = all_filenames[:num_validation]
        train_filenames = all_filenames[num_validation:num_validation+num_labeled]
        unlabeled_filenames = all_filenames[num_validation+num_labeled:]

        assert len(set(train_filenames).intersection(set(validation_filenames))) == 0
        assert len(set(train_filenames).intersection(set(unlabeled_filenames))) == 0
        assert len(set(validation_filenames).intersection(set(unlabeled_filenames))) == 0

        if validation_proportion > 0 and len(validation_filenames) == 0:
            raise ValueError("Validation proportion is too small.")
        
        train, validate = map(
            lambda x: PetsLabeled(
                False,
                x,
                image_folder = os.path.join(self.train_path, 'images'),
                label_folder = os.path.join(self.train_path, 'labels')
            ),
            (train_filenames, validation_filenames)
        )
        unlabeled = PetsUnlabeled(
            unlabeled_filenames,
            image_folder = os.path.join(self.train_path, 'images')
        )

        if validation_proportion > 0:
            return train, validate, unlabeled
        return train, unlabeled