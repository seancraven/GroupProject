import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from src.pet_3.data import Pets
from typing import OrderedDict


class LoadedModel:
    def __init__(
        self,
        model_path: str,
    ):
        self.state_dict: OrderedDict = torch.load(model_path)

        model = torch.hub.load(
            "mateuszbuda/brain-segmentation-pytorch",
            "unet",
            in_channels=3,
            out_channels=1,
            init_features=32,
            pretrained=False,
        )
        model.load_state_dict(self.state_dict)
        self.model = model
        self.directory: str = os.path.dirname(model_path)
        self.file_name: str = os.path.basename(model_path)
        self.loss, self.num_epoch_str = self.file_name.split("_")
        self.num_epoch: int = int(self.num_epoch_str.split(".")[0])


class ModelMetrics:
    """Model metric evaluation class"""

    def __init__(self, model: nn.Module, test_dataset: Pets):
        """Add model metric to class here"""
        self.model = model
        self.test_dataset = test_dataset
        self.test_accuracy = self._accuracy()
        self.test_loss = self._binary_cross_entropy_loss()
        self.test_iou = self._intersection_over_union()

    def _accuracy(self):
        """Calculate accuracy metric"""

        self.model.eval()
        test_loader = DataLoader(self.test_dataset, batch_size=1, num_workers=8)

        with torch.no_grad():
            correct_pixels = 0
            total_pixels = 0
            for _, (data, target) in enumerate(test_loader):
                data, target = data, target
                output = self.model.forward(data)
                pred = (output > 0.5).int()
                correct_pixels += (pred == target).sum().item()
                total_pixels += (target.shape[2] * target.shape[3]) * target.shape[0]

        return correct_pixels / total_pixels

    def _binary_cross_entropy_loss(self):
        """Calculates the binary cross entropy loss"""

        self.model.eval()
        test_loader = DataLoader(self.test_dataset, batch_size=1, shuffle=True)

        with torch.no_grad():
            loss_sum = 0.0
            for _, (data, target) in enumerate(test_loader):
                output = self.model.forward(data)
                loss = F.binary_cross_entropy(output, target)
                loss_sum += loss.item()

        return loss_sum / len(test_loader)

    def _intersection_over_union(self):
        """Calculate IoU metric"""

        self.model.eval()
        test_loader = DataLoader(self.test_dataset, batch_size=1, shuffle=True)

        with torch.no_grad():
            intersection_sum = 0.0
            union_sum = 0.0
            for _, (data, target) in enumerate(test_loader):
                output = self.model.forward(data)
                pred = (output > 0.5).int()
                intersection = (pred * target).sum().item()
                union = (pred + target).sum().item() - intersection
                intersection_sum += intersection
                union_sum += union

        iou = intersection_sum / union_sum
        return iou / len(test_loader)


# def model_metric(
#     loadmodel: LoadedModel, metric: Callable, test_dataset: Pets, output_path: str
# ):
#     """
#     evaluates model with given metric exports data in ....

#     :param metric: Metric with sum reduction and scalar output.
#     :param test_dataset: test dataset.
#     """
#     model = loadmodel.model
#     model.to(device)

#     test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

#     with torch.no_grad():
#         eval_metric = 0.0
#         for X, y in test_loader:
#             X, y = X.to(device), y.to(device)
#             y_pred = model(X)
#             eval_metric += metric(y, y_pred).item()

#     with open(output_path, "w") as f:
#         pass