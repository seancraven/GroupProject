"""
Before you run the script you must log into wanb, it should be installed on your environment.
in the console if you have wandb installed
console:
wandb login
After this you will be prompted for you api key. This is found when you log into wandb online
at: https://wandb.ai/home
Paste the key into the console.
"""
import time

# You need to place # type: ignore after all wandb statements.
import torch
from torch.utils.data import DataLoader
import wandb  # type: ignore
from src.pet_3.data import Pets


if __name__ == "__main__":
    model = torch.hub.load(
        "mateuszbuda/brain-segmentation-pytorch",
        "unet",
        in_channels=3,
        out_channels=1,
        init_features=32,
        pretrained=False,
    )

    learning_rate = 1e-3
    num_epoch = 10
    loss_name = "Mean Square Error"
    architecture = "U-Net_32_init"
    optimizer = "Adam"
    train_dataset = Pets("./pet_3", "all_train")

    train_loader = DataLoader(train_dataset, batch_size=32, num_workers=2)

    # set the wandb project where this run will be logged
    wandb.init(  # type: ignore
        project="Example-Project",
        # track hyperparameters and run metadata
        config={
            "learning_rate": learning_rate,
            "architecture": architecture,
            "dataset": train_dataset.name,
            "epochs": num_epoch,
            "loss": loss_name,
            "optimizer": optimizer,
        },
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Move Model to Cuda!
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epoch):
        epoch_start_time = time.time()
        epoch_loss = 0.0

        for img, labels in train_loader:
            # Move Data to Cuda !!!!
            img, labels = img.to(device), labels.to(device)
            opt.zero_grad()
            out = model(img)

            loss = (labels - out).pow(2).sum()
            loss.backward()
            opt.step()

            epoch_loss += loss.item()

        # End batch data for wandb
        epoch_loss /= float(len(train_dataset))
        epoch_end_time = time.time()
        total_epoch_time = epoch_end_time - epoch_start_time

        # Store the things you want in wanb
        # All these metrics are logged and plotted can be seen by the team.
        # All the values are plotted against the log number.
        wandb.log({"Mean Epoch Loss": epoch_loss, "Epoch Time": total_epoch_time})  # type: ignore

    # If you want to perform inference, make sure to move the
    # tensors to device.
    # If you want to do plotting you need to move the
    # model outputs must be moved to cpu with out.to("cpu") if you want to plot
    # Losses etc have to be loss.item(), this moves them to cpu also.

    wandb.finish()  # type: ignore
