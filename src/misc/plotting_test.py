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
from src.plotting.model_plotting_utils import (
    plot_bar,
    plot_img,
    plot_img_row,
    plot_img_label_grid,
    plot_img_label_pred,
    plot_model_figures,
)

from LSD import LSD

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # model = LoadedModel("./models/u_net_supervised/Mean Squared Error_20.pt")
    model = LSD()
    testdataset = Pets("./src/pet_3", "test", binary_labels=True)

    model_metrics = ModelMetrics(model, testdataset)

    # PLOTTING BAR PLOT
    fig = plot_bar(
        title="LSD Test Metrics",
        x_label="Metric",
        y_label="Value",
        # named arguments
        accuracy=model_metrics.test_accuracy,
        loss=model_metrics.test_loss,
        iou=model_metrics.test_iou
        # ... etc
        # metric = model_metrics.metric,
    )

    fig.savefig("figures/barplot_test.png")

    # PLOTTING IMAGE
    fig = plot_img(
        image=testdataset[0][0],
        title="LSD Test Image",
        x_label="Image",
        y_label="Value",
    )

    fig.savefig("figures/plot_img_test.png")

    # PLOTTING ROW OF IMAGES
    fig = plot_img_row(
        title="LSD Test Images",
        x_label="Image",
        y_label="Value",
        # named arguments
        img1=testdataset[0][0],
        img2=testdataset[1][0],
        img3=testdataset[2][0],
        img4=testdataset[3][0],
        # ... etc
        # imgN = testdataset[N][0],
    )

    fig.savefig("figures/img_row_test.png")

    # PLOTTING GRID OF IMAGES AND LABELS
    fig = plot_img_label_grid(
        title="LSD Test Images and Labels",
        x_label="Image",
        y_label="Value",
        # named arguments
        image_dictionary={
            "Image 1": testdataset[0][0],
            "Image 2": testdataset[1][0],
            "Image 3": testdataset[2][0],
            "Image 4": testdataset[3][0],
            # ... etc
            # imgN = testdataset[N][0],
        },
        label_dictionary={
            "Label 1": testdataset[0][1],
            "Label 2": testdataset[1][1],
            "Label 3": testdataset[2][1],
            "Label 4": testdataset[3][1],
            # ... etc
            # imgN = testdataset[N][0],
        },
    )

    fig.savefig("figures/img_label_grid_test.png")

    # PLOTTING IMAGE, LABEL, AND PREDICTION
    fig = plot_img_label_pred(
        title="LSD Test Images, Labels, and Predictions",
        x_label="",
        y_label="",
        image=testdataset[0][0],
        label=testdataset[0][1],
        prediction=model.predict(testdataset[0][0]),
    )

    fig.savefig("figures/img_label_pred_test.png")

    # PLOTTING MODEL FIGURES
    plot_model_figures(
        model_path="./models/u_net_supervised/Mean Squared Error Loss_20.pt"
    )
