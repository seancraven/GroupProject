"""
Training U_nets on all partially labeled and unlabeled datasets.
"""
# pylint: disable-all
import time
import os

# You need to place # type: ignore after all wandb statements.
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split

import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

import wandb  # type: ignore

from ..src.pet_3.deprocated_data import Pets
from LSD_model import LSD


def load_u_net():
    """
    Load the U-Net model from the brain-segmentation-pytorch repo
    """
    model = torch.hub.load(
        "mateuszbuda/brain-segmentation-pytorch",
        "unet",
        in_channels=3,
        out_channels=1,
        init_features=32,
        pretrained=False,
    )
    return model


def mean_square_error_loss(outputs, targets):
    """
    Mean Square Error Loss
    """
    return F.mse_loss(outputs, targets)


def binary_cross_entropy_loss(outputs, targets):
    """
    binary cross entropy loss
    """
    return F.binary_cross_entropy(outputs, targets)


# def dice_loss(outputs, targets, smooth=1e-7):
#     """
#     Dice loss
#     """
#     num_classes = 2  # outputs.shape[1]
#     dice = 0
#     for i in range(num_classes):
#         output = outputs[:, 0, :, :]
#         target = (targets == i).float()

#         product = output * target
#         intersection = product.sum()

#         output_sum = output.sum()
#         target_sum = target.sum()
#         union = output_sum + target_sum

#         # intersection = torch.sum(output * target)
#         # union = torch.sum(output) + torch.sum(target)
#         dice_class = (2 * intersection + smooth) / (union + smooth)
#         dice += dice_class
#     return 1 - dice / num_classes


def train_model_wanb(
    model, criterion, learningrate, numepoch, train_loader, val_loader
):
    """
    Trains model
    """
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=learningrate)

    data_len = len(train_loader.dataset)
    for _ in range(numepoch):
        epoch_start_time = time.time()
        epoch_loss = 0.0

        # Train phase
        for img, labels in train_loader:
            img, labels = img.to(device), labels.to(device)
            opt.zero_grad()
            # forwards
            out = model(img)
            loss = criterion(out, labels)
            # backwards
            loss.backward()
            opt.step()
            epoch_loss += loss.item()

        # validation phase
        with torch.no_grad():
            val_loss = 0.0
            for img, labels in val_loader:
                img, labels = img.to(device), labels.to(device)
                out = model(img)
                loss = criterion(out, labels)
                val_loss += loss.item()

        # End batch data for wandb
        epoch_loss /= float(data_len)
        val_loss /= float(data_len)

        epoch_end_time = time.time()
        total_epoch_time = epoch_end_time - epoch_start_time

        # wandb logging
        wandb.log(
            {
                "Mean Epoch Loss": epoch_loss,
                "Mean Val Loss": val_loss,
                "Epoch Time": total_epoch_time,
            }
        )  # type: ignore


def multi_model_baseline():
    """Trains multiple models on different fractions of the dataset."""
    fractions = [i / 10 for i in range(1, 10)]
    train_set = Pets("./src/pet_3/", binary_labels=True)
    for split_fract in fractions:
        train_set.labeled_fraction = split_fract
        _, labeled = train_set.get_datasets()

        train_loader = DataLoader(
            labeled, batch_size=32, shuffle=True, num_workers=8
        )

        model = load_u_net()

        loss_name = "Mean_Squared_Error_Loss"
        wandb.init(  # type: ignore
            project="Mean Squared Error Loss",
            # track hyperparameters and run metadata
            config={
                "learning_rate": learning_rate,
                "architecture": architecture,
                "dataset": labeled.name,
                "epochs": num_epoch,
                "loss": loss_name,
                "optimizer": optimizer,
            },
        )

        train_model_wanb(
            model,
            mean_square_error_loss,
            learning_rate,
            num_epoch,
            train_loader,
            (),
        )

        file_name = os.path.join(
            "../../models",
            "u_net_supervised",
            f"{loss_name}_{num_epoch}_{labeled.name}.pt",
        )
        torch.save(model.state_dict(), file_name)

        wandb.finish()

        model = load_u_net()

        loss_name = "Categorical_Cross_Entropy_Loss"
        wandb.init(  # type: ignore
            project="Mean Squared Error Loss",
            # track hyperparameters and run metadata
            config={
                "learning_rate": learning_rate,
                "architecture": architecture,
                "dataset": train_set.name,
                "epochs": num_epoch,
                "loss": loss_name,
                "optimizer": optimizer,
            },
        )

        train_model_wanb(
            model,
            binary_cross_entropy_loss,
            learning_rate,
            num_epoch,
            train_loader,
            (),
        )

        file_name = os.path.join(
            "../../models",
            "u_net_supervised",
            f"{loss_name}_{num_epoch}_{labeled.name}.pt",
        )
        torch.save(model.state_dict(), file_name)

        wandb.finish()


def evaluate_model_wanb(model, criterion, test_loader):
    """
    Evaluates model
    """

    with torch.no_grad():
        model.to(device)
        epoch_loss = 0.0

        for img, labels in test_loader:
            img, labels = img.to(device), labels.to(device)

            # forwards
            out = model(img)
            loss = criterion(out, labels)

            epoch_loss += loss.item()

        # End batch data for wandb
        epoch_loss /= float(len(test_dataset))

        wandb.log({"Mean Epoch Loss": epoch_loss})  # type: ignore


def evalute_model_img(model, test_loader, save_path):
    """
    Saves images and predicted label to png file
    """
    with torch.no_grad():
        model.to("cpu")
        # print out the first batch of images and labels
        images, _ = next(iter(test_loader))
        guesses = model(images)

        # print out the first 8 images and labels using matplotlib
        fig = plt.figure(figsize=(20, 4))

        # images
        for idx in range(4):
            _ = fig.add_subplot(2, 4, idx + 1, xticks=[], yticks=[])
            imshow(images[idx].permute(1, 2, 0).numpy())

        # labels
        for idx in range(4, 8):
            _ = fig.add_subplot(2, 4, idx + 1, xticks=[], yticks=[])
            imshow(guesses[idx - 4].permute(1, 2, 0).numpy())

        # save the figure
        fig.savefig(save_path)
        plt.close(fig)
        model.to(device)


if __name__ == "__main__":
    # load models
    mse_model = load_u_net()
    cce_model = load_u_net()

    # cuda
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mse_model.to(device)
    cce_model.to(device)

    # meta
    learning_rate = 1e-3
    num_epoch = 20
    architecture = "Lysergic Acid Diethylamide"
    optimizer = "Adam"

    architecture = "U-Net_32_init"
    optimizer = "Adam"

    # datasets
    multi_model_baseline()
