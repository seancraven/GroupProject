"""
Experiment using 3 di
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

from pet_3.data import Pets
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
        epoch_loss /= float(len(train_dataset))
        val_loss /= float(len(val_dataset))

        epoch_end_time = time.time()
        total_epoch_time = epoch_end_time - epoch_start_time

        # wandb logging
        wandb.log({"Mean Epoch Loss": epoch_loss, "Mean Val Loss": val_loss, "Epoch Time": total_epoch_time})  # type: ignore

    file_name = os.path.join(
        "models", "u_net_supervised", f"{loss_name}_{num_epoch}.pt"
    )
    torch.save(model.state_dict(), file_name)


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
    dataset = Pets("./pet_3", "all_train", binary_labels=True)
    test_dataset = Pets("./pet_3", "test", binary_labels=True)

    # split dataset into train and validation sets
    train_size = int(len(dataset) * 0.8)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # create dataloaders
    trainloader = DataLoader(train_dataset, batch_size=32, num_workers=8)
    valloader = DataLoader(val_dataset, batch_size=32, num_workers=8)
    testloader = DataLoader(test_dataset, batch_size=32, num_workers=8)

    # experiment 1 : Mean Squared Error Loss
    loss_name = "Mean Squared Error Loss"

    wandb.init(  # type: ignore
        project="Mean Squared Error Loss",
        # track hyperparameters and run metadata
        config={
            "learning_rate": learning_rate,
            "architecture": architecture,
            # "dataset": train_dataset.name,
            "epochs": num_epoch,
            "loss": loss_name,
            "optimizer": optimizer,
        },
    )

    train_model_wanb(
        mse_model,
        mean_square_error_loss,
        learning_rate,
        num_epoch,
        trainloader,
        valloader,
    )
    evaluate_model_wanb(mse_model, mean_square_error_loss, testloader)
    evalute_model_img(mse_model, testloader, "mse_model.png")

    wandb.finish()  # type: ignore

    # experiment 2 : Categorical Cross Entropy Loss
    loss_name = "Categorical Cross Entropy Loss"

    wandb.init(  # type: ignore
        project="Categorical Cross Entropy Loss",
        # track hyperparameters and run metadata
        config={
            "learning_rate": learning_rate,
            "architecture": architecture,
            # "dataset": train_dataset.name,
            "epochs": num_epoch,
            "loss": loss_name,
            "optimizer": optimizer,
        },
    )

    train_model_wanb(
        cce_model,
        binary_cross_entropy_loss,
        learning_rate,
        num_epoch,
        trainloader,
        valloader,
    )

    train_model_wanb(
        cce_model,
        binary_cross_entropy_loss,
        learning_rate,
        num_epoch,
        trainloader,
        valloader,
    )
    evaluate_model_wanb(cce_model, binary_cross_entropy_loss, testloader)
    evalute_model_img(cce_model, testloader, "cce_model.png")

    wandb.finish()  # type: ignore

    # experiment 3 : Dice Loss
    # loss_name = "Dice Loss"

    # wandb.init(  # type: ignore
    #     project="Dice Loss",
    #     # track hyperparameters and run metadata
    #     config={
    #         "learning_rate": learning_rate,
    #         "architecture": architecture,
    #         "dataset": train_dataset.name,
    #         "epochs": num_epoch,
    #         "loss": loss_name,
    #         "optimizer": optimizer,
    #     },
    # )

    # train_model_wanb(dice_model, dice_loss, learning_rate, num_epoch, trainloader)
    # evaluate_model_wanb(dice_model, dice_loss, testloader)
    # evalute_model_img(dice_model, testloader, "dice_model.png")

    # wandb.finish()  # type: ignore
