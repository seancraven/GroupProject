# pylint: disable-all
import os
from dataclasses import dataclass
import torch
import numpy as np
from torch import nn
from typing import Callable

from torch.utils.data import DataLoader
from src.pet_3.data import Pets



class LoadedModel:
    def __init__(self, model_path: str):
        self.model: nn.Module = torch.load(model_path)
        directory: str = os.path.dirname(model_path)
        file_name: str = os.path.basename(model_path)
        loss, self.num_epoch = file_name.split("_")
        num_epoch: int = int(self.num_epoch.split(".")[0])


class ModelMetrics:
    """ Model metric evaluation class """
    def __init__(self, loadedmodel : LoadedModel, test_dataset : Pets):
        """ Add model metric to class here """
        self.model = loadedmodel
        self.test_dataset = test_dataset
        self.test_accuracy = self._accuracy()
        self.test_iou = self._intersection_over_union()
    

    def _accuracy(self):
        """ Calculate accuracy metric """
        # move model to device ??
        self.model.eval()
        # self.model.to(device) # ???
        test_loader = DataLoader(self.test_dataset, batch_size=1, num_workers=8)

        with torch.no_grad():
            correct_pixels = 0
            total_pixels = 0
            for _, (data, target) in enumerate(test_loader):
                data, target = data, target
                output = self.model.forward(data)
                pred = (output > .5).int()
                correct_pixels += (pred == target).sum().item()
                total_pixels += (target.shape[2] * target.shape[3]) * target.shape[0]

        return correct_pixels / total_pixels
    
    
    def _intersection_over_union(self):
        """ Calculate IoU metric """
         # move model to device ?
        self.model.eval()
        # self.model.to(device) # ???
        test_loader = DataLoader(self.test_dataset, batch_size=1, shuffle=True)

        with torch.no_grad():
            intersection_sum = 0.0
            union_sum = 0.0
            for _, (data, target) in enumerate(test_loader):
                output = self.model.forward(data)
                pred = (output > .5).int()
                intersection = (pred * target).sum().item()
                union = (pred + target).sum().item() - intersection
                intersection_sum += intersection
                union_sum += union
        
        iou = intersection_sum / union_sum
        return iou / len(test_loader)


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

