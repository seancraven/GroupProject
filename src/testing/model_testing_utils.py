# pylint: disable-all
import os
from dataclasses import dataclass
import torch
import numpy as np
from torch import nn
from typing import Callable
from data import Pets
from torch.data.utils import DataLoader


def model_metric(
    loadmodel: LoadedModel, metric: Callable, test_dataset: Pets, output_path: str
):
    """
    evaluates model with given metric exports data in ....

    :param metric: Metric with sum reduction and scalar output.
    :param test_dataset: test dataset.
    """
    model = loadmodel.model
    model.to(device)

    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    with torch.no_grad():
        eval_metric = 0.0

        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            eval_metric += metric(y, y_pred).item()

    with open(output_path, "w") as f:
        pass


class LoadedModel:
    def __init__(self, model_path: str):
        self.model: nn.Module = torch.load(model_path)
        directory: str = os.path.dirname(model_path)
        file_name: str = os.path.basename(model_path)
        loss, self.num_epoch = file_name.split("_")
        num_epoch: int = int(self.num_epoch.split(".")[0])


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    LoadedModel("./models/u_net_supervised/Mean Squared Error_20.pt")
