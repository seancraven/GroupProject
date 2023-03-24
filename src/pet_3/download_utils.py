"""
Download the pet data from the Oxford repo. Define an 80 / 20 test split write to
split files.
"""
import os
import shutil
import random
from typing import Tuple, Dict, List
from urllib import request
import tarfile
import torch


random.seed(0)
torch.manual_seed(0)


def _download_pet_from_url(target_dir: str) -> Tuple[str, str]:
    """Downloads the Oxford pet data from the urls."""
    images_url = "https://thor.robots.ox.ac.uk/~vgg/data/pets/images.tar.gz"
    annotations_url = "https://thor.robots.ox.ac.uk/~vgg/data/pets/annotations.tar.gz"

    images_download_loc = os.path.join(target_dir, "images.tar.gz")
    annotations_download_loc = os.path.join(target_dir, "annotations.tar.gz")

    request.urlretrieve(images_url, images_download_loc)
    request.urlretrieve(annotations_url, annotations_download_loc)

    return images_download_loc, annotations_download_loc


def _unzip_pet_data(tar_img_path: str, tar_ano_path: str, train_path: str):
    """
        Unzipts the tar files and moves the files to the train folders.
    Args:
        target_dir: Target directory

    """

    with tarfile.open(tar_img_path, "r") as img:
        members = img.getmembers()
        img.extractall(path=train_path, members=members[1:])
        img.close()

    with tarfile.open(tar_ano_path, "r") as ano:
        members = ano.getmembers()
        anotations = [img for img in members if ".png" in img.name]
        ano.extractall(path=train_path, members=anotations)
        ano.close()

    # move out of trimap
    ano_dest = os.path.join(train_path, "labels")
    os.makedirs(ano_dest)
    ano_ext_dest = os.path.join(
        train_path, "annotations", "trimaps"
    )  # Where files are extracted to.
    ano_files = os.listdir(ano_ext_dest)
    for file in ano_files:
        fname = os.path.join(ano_ext_dest, file)
        dest_name = os.path.join(ano_dest, file)

        os.rename(fname, dest_name)
    # cleaning up
    ano_path = os.path.join(train_path, "annotations")
    shutil.rmtree(ano_path)
    os.remove(tar_img_path)
    os.remove(tar_ano_path)


def _populate_data(target_dir):
    test_folder = os.path.join(target_dir, "test_data")
    train_folder = os.path.join(target_dir, "train_data")

    if os.path.isdir(test_folder) and os.path.isdir(train_folder):
        print(f"Train data is in {train_folder}")
        print(f"Test data is in {test_folder}")
    else:
        os.makedirs(test_folder)
        os.makedirs(train_folder)

        print("Downloading Files from https://thor.robots.ox.ac.uk/~vgg/data/pets")
        print("Takes a minute or so be patient please.")

        tar_img_path, tar_ano_path = _download_pet_from_url(target_dir)

        test_folder_data = os.path.join(test_folder, "images")
        test_folder_labels = os.path.join(test_folder, "labels")

        train_folder_unlabeled = os.path.join(train_folder, "unlabeled")

        os.makedirs(test_folder_data)
        os.makedirs(test_folder_labels)
        os.makedirs(train_folder_unlabeled)

        # Download data
        _unzip_pet_data(tar_img_path, tar_ano_path, train_folder)

        if os.path.isfile(os.path.join(target_dir, "train.txt")) and os.path.isfile(
            os.path.join(target_dir, "test.txt")
        ):
            print("Split files already exist")
        else:
            raise FileNotFoundError(
                """Split files do not exist, get them from the repo.
                Then place them in the pet_3 directory"""
            )
        print("Performing train test split")

        test_file = os.path.join(target_dir, "test.txt")
        _clean_labels(target_dir)
        _move_files_according_to_file(test_file, train_folder, test_folder)


def _clean_img(root: str):
    img_dir = os.path.join(root, "train_data", "images")
    img_list = os.listdir(img_dir)
    for img in img_list:
        if img[:-3] != "png":
            os.remove(os.path.join(img_dir, img))


def _class_balance_split(split_fraction: float, root: str, mode="test"):
    """Makes a dictionary, where keys are each class. The dictionary is used to write a
    file that describes the split."""
    images_loc = os.path.join(root, "train_data", "images")
    assert isinstance(split_fraction, float)
    images = os.listdir(images_loc)
    images_dict: Dict[str, List[str]] = {}
    # Populate each class
    for img in images:
        img_class = img.split("_")[0]

        if img_class in images_dict:
            images_dict[img_class].append(img)

        else:
            images_dict[img_class] = [img]
    if mode == "test":
        _write_train_test_file(images_dict, split_fraction, root)
    elif mode == "unlabeled":
        _write_unlabeled_file(images_dict, split_fraction, root)
    return images_dict


def _move_files_according_to_file(
    files_to_move: str, train_dir: str, destination: str, mode="test"
):
    move_files: List[str] = []

    with open(files_to_move, "r", encoding="UTF-8") as test:
        for files in test.readlines():
            move_files.append(files[:-1])
        test.close()

    img_dir = os.path.join(train_dir, "images")
    label_dir = os.path.join(train_dir, "labels")
    all_img = os.listdir(img_dir)

    for file in all_img:
        if file in move_files:
            if mode == "test":
                os.replace(
                    os.path.join(img_dir, file),
                    os.path.join(destination, "images", file),
                )
                label = file[:-4] + ".png"
                os.replace(  #
                    os.path.join(label_dir, label),
                    os.path.join(destination, "labels", label),
                )
            elif mode == "unlabeled":
                os.replace(
                    os.path.join(img_dir, file),
                    os.path.join(train_dir, "unlabeled", file),
                )


def return_to_train(train_dir):
    """Moves all of the unlabeled files back to train."""
    un_dir = os.path.join(train_dir, "unlabeled")
    img_dir = os.path.join(train_dir, "images")
    unlabeled_files = os.listdir(un_dir)
    for file in unlabeled_files:
        os.replace(os.path.join(un_dir, file), os.path.join(img_dir, file))
    print("All files returned to train_data/images.")

def _write_train_test_file(
    classed_files: Dict[str, List[str]], split_fraction: float, root: str
):
    """
    Splits each dictionary entry and writes these items to two files train.txt and test.txt.

    This function should not be called more than once. It will cause information
    leak between train and test.

    :param classed_files: Dict of file names where the key is the class.
    :type classed_files: Dict[str, List[str]]
    :param split_fraction: fraction of the data for train.
    :type split_fraction: float
    """
    train_file = os.path.join(root, "train.txt")
    test_file = os.path.join(root, "test.txt")
    with open(train_file, "w", encoding="UTF-8") as train, open(
        test_file, "w", encoding="UTF-8"
    ) as test:
        for _, f_names in classed_files.items():
            trains, tests = train_test_split(f_names, split_fraction)
            for f_train in trains:
                train.write(f_train + "\n")
            for f_test in tests:
                test.write(f_test + "\n")
        train.close()
        test.close()


def _write_unlabeled_file(
    classed_files: Dict[str, List[str]], split_fraction: float, root: str
):
    labeled_file = os.path.join(root, f"labeled_train_{split_fraction}.txt")
    unlabeled_file = os.path.join(root, f"unlabeled_train_{split_fraction}.txt")
    with open(labeled_file, "w", encoding="UTF-8") as l_train, open(
        unlabeled_file, "w", encoding="UTF-8"
    ) as u_train:
        for _, f_names in classed_files.items():
            lab, un_lab = train_test_split(f_names, split_fraction)
            for f_lab in lab:
                l_train.write(f_lab + "\n")
            for f_un_lab in un_lab:
                u_train.write(f_un_lab + "\n")
        l_train.close()
        u_train.close()


def _clean_labels(root: str):
    label_dir = os.path.join(root, "train_data", "labels")
    img_labels = os.listdir(label_dir)
    for img in img_labels:
        if img[0] == ".":
            os.remove(os.path.join(label_dir, img))


def train_test_split(
    class_file_names: List[str], fraction: float
) -> Tuple[List[str], List[str]]:
    """
    Splits a list of file names into two lists where the first list is the fraction of the data.

    :param class_file_names: List of file names, of one class.
    :type class_file_names: List[str]
    :param fraction: Fraction of data for training.
    :type fraction: float
    :return: train, test.
    :rtype: Tuple[List[str], List[str]]
    """
    file_count = len(class_file_names)
    n_train = int(file_count * fraction)
    random.shuffle(class_file_names)
    return class_file_names[:n_train], class_file_names[n_train:]


if __name__ == "__main__":
    # _populate_data("./")
    fractions = [i / 10 for i in range(10)]
