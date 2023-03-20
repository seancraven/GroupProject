# pylint: disable-all
import os
from dataclasses import dataclass
import torch
import numpy as np
from torch import nn
from typing import Callable

from torch.utils.data import DataLoader
from src.pet_3.data import Pets
from src.testing.model_testing_utils import LoadedModel, ModelMetrics

from LSD import LSD

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # model = LoadedModel("./models/u_net_supervised/Mean Squared Error_20.pt")
    model = LSD()
    testdataset = Pets("./src/pet_3", "test", binary_labels=True)
    model_metrics = ModelMetrics(model, testdataset)
    print('LSD accuracy: {:.5f}'.format(model_metrics.test_accuracy), )
    print('LSD loss: {:.5f}'.format(model_metrics.test_loss),)
    print('LSD IOU: {:.5f}'.format(model_metrics.test_iou))

